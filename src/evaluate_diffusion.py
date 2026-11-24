"""
Diffusion 重建评估: DDIM 采样 + 每步数据一致性引导 + 四口径指标。

引导机制(与训练归一化严格同构, 见 make_ddpm_cache.py):
  每个DDIM步的x0估计(384^2归一化幅度) -> 反归一化 -> 乘零填充图的相位(复数化)
  -> 按灵敏度展开回8线圈k-space -> DC替换已采样位置 -> IFFT+SENSE合并
  -> 取幅度再归一化, 作为该步的引导x0。

用法:
    python src/evaluate_diffusion.py --ddpm_checkpoint checkpoints/ddpm_brain/last.pth \
        --sme_checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth \
        --test_dir ../mri_data/subset_compressed/test --acceleration 8 \
        --save_dir results/results_diffusion_8x --max_slices 40
"""

import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.coil_compression import compress_coils
from data.mask import get_mask_for_acceleration
from data.transforms import ifft2c, center_crop
from data.roi_mask import compute_roi_mask, compute_background_mask
from models.sme_unet_model import SMEUNetModel
from models.ddpm import DDPM
from evaluate_multicoil import optimal_scale_align, compute_metrics


def fft2c_t(im):  # (B,C,H,W) complex tensor
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(im, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c_t(ks):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(ks, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


class GuidedSampler:
    """管理每切片的测量项(设备端张量)与引导步骤。

    归一化方案v3: x = gt/(scale*C), C为训练时标定的全局常数(norm json)。
    推理端 self.mu=0, self.sd=C, 反归一化 x0_norm*C 再乘scale。
    """

    def __init__(self, ddpm, us_norm, mask, sens, device, steps=50, norm_c=1.0):
        self.norm_c = float(norm_c)
        # us_norm: (8,H,W) complex(已除scale); mask: (H,W) float; sens: (8,H,W) complex
        self.ddpm, self.device, self.steps = ddpm, device, steps
        self.us = us_norm.unsqueeze(0).to(device)                 # (1,8,H,W)
        self.mask = mask.to(device).unsqueeze(0).unsqueeze(0)     # (1,1,H,W)
        self.sens = sens.unsqueeze(0).to(device)                  # (1,8,H,W)
        self.H, self.W = us_norm.shape[-2:]

    @torch.no_grad()
    def guide_from_norm(self, x0_norm):
        """x0归一化幅度(1,1,384,384) -> 数据一致性引导后的归一化幅度。"""
        # 首次调用时缓存零填充合并的统计与相位
        if not hasattr(self, "phase"):
            zf = torch.sum(torch.conj(self.sens) * ifft2c_t(self.us), dim=1, keepdim=True)
            mag = zf.abs()
            self.mu, self.sd = 0.0, self.norm_c    # v3: 全局常数C方案
            self.phase = zf / (mag + 1e-9)
        x0 = x0_norm * self.sd + self.mu
        canvas = torch.zeros(1, 1, self.H, self.W, dtype=x0.dtype, device=self.device)
        canvas[..., self.H // 2 - 192:self.H // 2 + 192,
               self.W // 2 - 192:self.W // 2 + 192] = x0
        x0_full = canvas * self.phase                             # 复数化
        ks_est = fft2c_t(self.sens * x0_full)
        ks_dc = self.mask * self.us + (1 - self.mask) * ks_est    # 数据一致性
        comb = torch.sum(torch.conj(self.sens) * ifft2c_t(ks_dc), dim=1, keepdim=True)
        mag384 = center_crop_t(comb.abs())
        return ((mag384 - self.mu) / self.sd).clamp(-1, 8)  # 数据范围适配: 缓存实测max~7.7


def center_crop_t(x, s=384):
    H, W = x.shape[-2:]
    return x[..., H // 2 - s // 2:H // 2 + s // 2, W // 2 - s // 2:W // 2 + s // 2]


@torch.no_grad()
def guided_sample(ddpm, sampler, steps=50):
    """DDIM 主循环(callback直接操作张量)。"""
    device = sampler.device
    ts = torch.linspace(ddpm.T - 1, 0, steps).long().tolist()
    x = torch.randn(1, 1, 384, 384, device=device)
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else 0
        x0, eps = ddpm.predict_x0(x, t)
        x0 = sampler.guide_from_norm(x0)                          # 引导
        x = ddpm.sqrt_ab[t_prev] * x0 + ddpm.sqrt_1mab[t_prev] * eps
    return x0 * sampler.sd + sampler.mu                            # 反归一化幅度


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm_checkpoint", type=str, required=True)
    ap.add_argument("--sme_checkpoint", type=str, required=True)
    ap.add_argument("--test_dir", type=str, required=True)
    ap.add_argument("--acceleration", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_slices", type=int, default=40)
    ap.add_argument("--norm_json", type=str, default="../mri_data/ddpm_cache_v3_norm.json",
                    help="训练缓存的全局常数C(与训练同口径)")
    ap.add_argument("--save_dir", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ddpm_checkpoint, map_location="cpu", weights_only=False)
    ddpm = DDPM(chans=ck["args"]["chans"], num_pools=ck["args"]["num_pools"])
    sd = ck.get("ema_state_dict") or ck["model_state_dict"]
    ddpm.load_state_dict(sd)
    ddpm.to(device).eval()
    print(f"DDPM 载入(step={ck.get('step')})")

    ck2 = torch.load(args.sme_checkpoint, map_location="cpu", weights_only=False)
    a = ck2["args"]
    sme = SMEUNetModel(a["sme_chans"], a["sme_pool_layers"], a["unet_chans"], a["unet_pool_layers"])
    sme.load_state_dict(ck2["model_state_dict"])
    sme.to(device).eval()

    nlf = {4: 32, 8: 16}[args.acceleration]
    files = sorted(glob.glob(os.path.join(args.test_dir, "*.h5")))
    os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    t_start = time.time()
    done = 0
    for f in files:
        if done >= args.max_slices:
            break
        with h5py.File(f, "r") as h:
            ks_full = h["kspace"][()]
            gt_full = h["reconstruction_rss"][()].astype(np.float32)
        for si in range(ks_full.shape[0]):
            if done >= args.max_slices:
                break
            ks = ks_full[si]
            C = ks.shape[0]
            ks8 = compress_coils(ks.astype(np.complex64), nlf, 8) if C > 8 else ks.astype(np.complex64)
            m2, _, _ = get_mask_for_acceleration(ks8.shape[1:], args.acceleration, seed=args.seed)
            us = ks8 * m2
            zf_rss = np.sqrt(np.sum(np.abs(np.stack([ifft2c(c) for c in us])) ** 2, 0))
            scale = zf_rss.std() + 1e-12
            kt = torch.from_numpy(np.stack([us.real, us.imag], 1)).float().unsqueeze(0).to(device) / scale
            mt = torch.from_numpy(m2.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                sens = sme(kt, mt, nlf)[0].cpu().numpy()
            sc = sens[:, 0] + 1j * sens[:, 1]
            us_norm = torch.from_numpy(us / scale)

            norm_c = json.load(open(args.norm_json))["C"] if os.path.exists(args.norm_json) else 1.0
            sampler = GuidedSampler(ddpm, us_norm, torch.from_numpy(m2.astype(np.float32)),
                                    torch.from_numpy(sc), device, steps=args.steps, norm_c=norm_c)
            t0 = time.time()
            recon = guided_sample(ddpm, sampler, steps=args.steps)  # (1,1,384,384)
            pred = recon.squeeze().cpu().numpy() * scale
            gt = center_crop(gt_full[si], (384, 384))

            bg = compute_background_mask(gt)
            roi = compute_roi_mask(gt, background_mask=bg)
            p_full, s_full = compute_metrics(pred, gt)
            p_bg, s_bg = compute_metrics(pred, gt, mask=bg)
            p_roi, s_roi = compute_metrics(pred, gt, mask=roi)
            rows.append([f"{os.path.basename(f)}:{si}", p_full, s_full, p_bg, s_bg, p_roi, s_roi])
            done += 1
            if done % 5 == 0:
                el = time.time() - t0
                print(f"  {done}/{args.max_slices} | 最近切片 {el:.1f}s | "
                      f"全图SSIM={s_full:.4f} ROI-SSIM={s_roi:.4f}", flush=True)

    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
    print(f"\n=== Diffusion({args.acceleration}x, {args.steps}步, {done}切片, 总{(time.time()-t_start)/60:.1f}min) ===")
    print(f"全图: PSNR={arr[:,0].mean():.2f} SSIM={arr[:,1].mean():.4f}")
    print(f"背景遮罩: PSNR={arr[:,2].mean():.2f} SSIM={arr[:,3].mean():.4f}")
    print(f"ROI: PSNR={arr[:,4].mean():.2f} SSIM={arr[:,5].mean():.4f}")
    with open(os.path.join(args.save_dir, "metrics_diffusion.csv"), "w") as fh:
        fh.write("slice,psnr_full,ssim_full,psnr_bg,ssim_bg,psnr_roi,ssim_roi\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n")
    print(f"已保存: {args.save_dir}/metrics_diffusion.csv")


if __name__ == "__main__":
    main()
