"""
方案C: 训练后量化(PTQ INT8) + 精度/速度/体积三维修计。

设计:
- 只量化卷积主干: SME+U-Net的recon_unet、VarNet每个级联的refine_net
  (QuantStub/DeQuantStub包装), FFT/数据一致性/SME归一化全部保持fp32
  (与AMP时期同一套数值纪律);
- 校准: 训练集若干切片走完整前向, 让observer看到真实激活分布;
- 对照: fp32 / fp16(autocast推理) / int8(PTQ) × (权重大小, 单切片时延,
  四口径指标), 在本地冻结test上评估, 输出一行式汇总。

用法:
    python src/quantize_eval.py --model sme   --checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth
    python src/quantize_eval.py --model varnet --checkpoint checkpoints/e2e_varnet_8x_lpips_roi_v3/best.pth
"""

import argparse
import copy
import glob
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.ao.quantization import QuantStub, DeQuantStub, get_default_qconfig, prepare, convert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.coil_compression import compress_coils
from data.mask import get_mask_for_acceleration
from data.transforms import ifft2c, center_crop
from data.roi_mask import compute_roi_mask, compute_background_mask
from models.sme_unet_model import SMEUNetModel
from models.e2e_varnet import E2EVarNet
from evaluate_multicoil import compute_metrics


class QuantWrapper(torch.nn.Module):
    """把卷积主干包成 量化输入->int8卷积->反量化输出。"""

    def __init__(self, unet):
        super().__init__()
        self.q, self.dq, self.unet = QuantStub(), DeQuantStub(), unet

    def forward(self, x):
        return self.dq(self.unet(self.q(x)))


def load_model(kind, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if kind == "sme":
        m = SMEUNetModel(a["sme_chans"], a["sme_pool_layers"], a["unet_chans"], a["unet_pool_layers"])
    else:
        m = E2EVarNet(a["num_cascades"], a["sme_chans"], a["sme_pool_layers"], a["cascade_chans"], a["cascade_pool_layers"])
    m.load_state_dict(ck["model_state_dict"])
    return m.eval()


def wrap_quant(model, kind):
    if kind == "sme":
        model.recon_unet = QuantWrapper(model.recon_unet)
    else:
        for c in model.cascades:
            c.refine_net = QuantWrapper(c.refine_net)
    return model


def get_inputs(test_dir, acceleration, n, seed=42):
    """返回 [(kspace_tensor, mask_tensor, nlf, gt, scale)]"""
    nlf = {4: 32, 8: 16}[acceleration]
    out = []
    files = sorted(glob.glob(os.path.join(test_dir, "*.h5")))
    for f in files:
        with h5py.File(f, "r") as h:
            ks_full = h["kspace"][()]
            gt_full = h["reconstruction_rss"][()].astype(np.float32)
        for si in range(0, ks_full.shape[0], max(1, ks_full.shape[0] // 2)):
            ks = ks_full[si]
            ks8 = compress_coils(ks.astype(np.complex64), nlf, 8) if ks.shape[0] > 8 else ks.astype(np.complex64)
            m2, _, _ = get_mask_for_acceleration(ks8.shape[1:], acceleration, seed=seed)
            us = ks8 * m2
            zf = np.sqrt(np.sum(np.abs(np.stack([ifft2c(c) for c in us])) ** 2, 0))
            scale = zf.std() + 1e-12
            kt = torch.from_numpy(np.stack([us.real, us.imag], 1)).float() / scale
            out.append((kt.unsqueeze(0), torch.from_numpy(m2.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                        nlf, center_crop(gt_full[si], (384, 384)), scale))
            if len(out) >= n:
                return out
    return out


def size_mb(model):
    p = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return p / 1024 ** 2


def latency(model, sample, device, warmup=3, rep=10, amp=False):
    model.to(device)
    kt, mt, nlf = sample[0].to(device), sample[1].to(device), sample[2]
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, enabled=amp):
                model(kt, mt, nlf)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            with torch.autocast(device_type=device.type, enabled=amp):
                model(kt, mt, nlf)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1000


def evaluate(model, samples, device, amp=False):
    model.to(device)
    res = []
    with torch.no_grad():
        for kt, mt, nlf, gt, scale in samples:
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(kt.to(device), mt.to(device), nlf)
            pred = center_crop(out.float().squeeze().cpu().numpy(), (384, 384)) * scale
            bg = compute_background_mask(gt)
            roi = compute_roi_mask(gt, background_mask=bg)
            res.append(compute_metrics(pred, gt) + compute_metrics(pred, gt, mask=bg)
                       + compute_metrics(pred, gt, mask=roi))
    return np.array(res).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sme", "varnet"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_dir", default="../mri_data/subset_compressed/train")
    ap.add_argument("--test_dir", default="../mri_data/subset_compressed/test")
    ap.add_argument("--acceleration", type=int, default=8)
    ap.add_argument("--n_calib", type=int, default=32)
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--backend", default="fbgemm")   # CPU int8
    ap.add_argument("--skip_int8", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    print(f"模型: {args.model} | 设备: {device} | backend: {args.backend}")

    calib = get_inputs(args.train_dir, args.acceleration, args.n_calib)
    test = get_inputs(args.test_dir, args.acceleration, args.n_eval)
    print(f"校准 {len(calib)} 切片 | 评估 {len(test)} 切片")

    base = load_model(args.model, args.checkpoint)

    # ---- fp32 基准 ----
    m_fp32 = copy.deepcopy(base)
    r0 = evaluate(m_fp32, test, device)
    lat_g = latency(m_fp32, test[0], device)
    lat_c = latency(m_fp32, test[0], cpu)
    print(f"[fp32] 体积={size_mb(m_fp32):.1f}MB GPU={lat_g:.0f}ms CPU={lat_c:.0f}ms "
          f"全图SSIM={r0[1]:.4f} ROI-SSIM={r0[5]:.4f}")

    # ---- fp16 (autocast 推理, 权重与FFT岛保持原样) ----
    try:
        m_fp16 = copy.deepcopy(base)
        r1 = evaluate(m_fp16, test, device, amp=True)
        lat_g16 = latency(m_fp16, test[0], device, amp=True)
        print(f"[fp16] GPU={lat_g16:.0f}ms 全图SSIM={r1[1]:.4f} ROI-SSIM={r1[5]:.4f}")
    except Exception as e:
        r1, lat_g16 = None, None
        print(f"[fp16] 失败: {e}")

    if args.skip_int8:
        print("[int8] 按参数跳过")
        return
    # ---- int8 PTQ (CPU) ----
    from torch.ao.quantization import QConfig
    from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver
    pt_qconfig = QConfig(
        activation=HistogramObserver.with_args(dtype=torch.quint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    import torch.nn as nn
    m_q = wrap_quant(copy.deepcopy(base), args.model)
    m_q.qconfig = None            # 根不量化: SME/FFT/DC 保持fp32
    wraps = [m_q.recon_unet] if args.model == "sme" else [c.refine_net for c in m_q.cascades]
    for w in wraps:
        w.qconfig = get_default_qconfig(args.backend)   # 普通卷积: per-channel(默认,精度更好)
        for name, mod in w.named_modules():             # 转置卷积: 降级 per-tensor(兼容)
            if isinstance(mod, nn.ConvTranspose2d):
                mod.qconfig = pt_qconfig
    prepare(m_q, inplace=True)
    with torch.no_grad():
        for kt, mt, nlf, *_ in calib:
            m_q(kt, mt, nlf)
    convert(m_q, inplace=True)
    m_q.eval()
    r2 = evaluate(m_q, test, cpu)
    lat_c8 = latency(m_q, test[0], cpu)
    print(f"[int8] 体积={size_mb(m_q):.1f}MB CPU={lat_c8:.0f}ms 全图SSIM={r2[1]:.4f} ROI-SSIM={r2[5]:.4f}")

    print("\n=== 汇总 (PSNR/SSIM 全图 | 背景 | ROI) ===")
    for name, r in [("fp32", r0), ("int8", r2)]:
        if r is not None:
            print(f"{name}: {r[0]:.2f}/{r[1]:.4f} | {r[2]:.2f}/{r[3]:.4f} | {r[4]:.2f}/{r[5]:.4f}")
    print(f"\n退化(int8-fp32): 全图SSIM {r2[1]-r0[1]:+.4f} | 背景SSIM {r2[3]-r0[3]:+.4f} | ROI-SSIM {r2[5]-r0[5]:+.4f}")


if __name__ == "__main__":
    main()
