"""
在测试集上评估完整E2E-VarNet模型，同时对比 RSS baseline 和(可选的)SME+U-Net，
产出三方定量指标(PSNR/SSIM)汇总表和可视化对比图，完整呈现
RSS -> SME+U-Net -> E2E-VarNet 这条技术演进链的提升幅度。

使用方法：
    python src/evaluate_e2e_varnet.py \
        --varnet_checkpoint checkpoints/e2e_varnet_4x/best.pth \
        --sme_unet_checkpoint checkpoints/sme_unet_4x/best.pth \
        --test_dir data/subset/test \
        --acceleration 4 \
        --target_num_coils 20 \
        --target_gt_h 384 --target_gt_w 384 \
        --save_dir results_e2e
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mask import get_mask_for_acceleration, ACCELERATION_CONFIGS
from data.transforms import center_crop, ifft2c
from data.coil_compression import compress_coils
from data.roi_mask import compute_roi_mask
from models.sme_unet_model import SMEUNetModel
from models.e2e_varnet import E2EVarNet

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def optimal_scale_align(pred, gt):
    pred_flat = pred.flatten().astype(np.float64)
    gt_flat = gt.flatten().astype(np.float64)
    denom = np.dot(pred_flat, pred_flat)
    scale = np.dot(pred_flat, gt_flat) / denom if denom > 0 else 1.0
    return pred * scale


def compute_background_mask(gt, threshold_ratio=0.05):
    """
    基于GT强度生成前景/背景遮罩。

    改用Otsu自适应阈值(而非固定的threshold_ratio比例)：多线圈RSS合并后，
    背景区域的噪声底噪并非趋近于0，而是服从非中心卡方分布，有一个不可忽略
    的均值(多个线圈噪声各自独立、平方和开根号后，不会互相抵消到0)。固定的
    "5%最大值"阈值在背景噪声底噪较高时会失效(背景大范围被误判为前景)，Otsu
    方法能根据每张图像素值的实际分布，自动找出前景/背景的最优分割点，更加
    鲁棒。threshold_ratio参数保留作为Otsu方法不可用时的备用方案。
    """
    try:
        from skimage.filters import threshold_otsu
        try:
            threshold = threshold_otsu(gt)
        except ValueError:
            # 极端情况(如全图数值完全相同)，Otsu无法计算，退回固定比例阈值
            threshold = threshold_ratio * gt.max()
    except ImportError:
        threshold = threshold_ratio * gt.max()

    mask = gt > threshold
    try:
        from scipy import ndimage
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        labeled, num_features = ndimage.label(mask)
        if num_features > 1:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            mask = (labeled == largest_label)
    except ImportError:
        pass
    return mask


def compute_metrics(pred, gt, mask=None):
    pred_aligned = optimal_scale_align(pred, gt)
    data_range = gt.max() - gt.min()
    if mask is None:
        psnr = peak_signal_noise_ratio(gt, pred_aligned, data_range=data_range)
        ssim = structural_similarity(gt, pred_aligned, data_range=data_range)
        return psnr, ssim
    diff_sq = (gt.astype(np.float64) - pred_aligned.astype(np.float64)) ** 2
    mse_masked = diff_sq[mask].mean()
    psnr = 10 * np.log10((data_range ** 2) / mse_masked) if mse_masked > 0 else float("inf")
    _, ssim_map = structural_similarity(gt, pred_aligned, data_range=data_range, full=True)
    ssim_masked = ssim_map[mask].mean()
    return psnr, ssim_masked


def rss_combine(coil_images):
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))


class LPIPSEvaluator:
    """纯评估用的LPIPS封装，不涉及梯度/训练，全局只初始化一次网络"""

    def __init__(self, net="alex", device="cpu"):
        if not LPIPS_AVAILABLE:
            raise ImportError("需要先安装lpips库: pip install lpips")
        self.device = device
        self.lpips_fn = lpips.LPIPS(net=net, verbose=False).to(device)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.lpips_fn.eval()

    @torch.no_grad()
    def compute(self, pred, gt):
        pred_aligned = optimal_scale_align(pred, gt)

        def to_tensor(img):
            img_min, img_max = img.min(), img.max()
            img_norm = (img - img_min) / (img_max - img_min + 1e-8) * 2 - 1
            t = torch.from_numpy(img_norm.astype(np.float32))
            t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            return t.to(self.device)

        pred_t = to_tensor(pred_aligned)
        gt_t = to_tensor(gt)
        return self.lpips_fn(pred_t, gt_t).item()


def load_sme_unet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    a = checkpoint.get("args", {})
    model = SMEUNetModel(
        sme_chans=a.get("sme_chans", 8), sme_pool_layers=a.get("sme_pool_layers", 2),
        unet_chans=a.get("unet_chans", 32), unet_pool_layers=a.get("unet_pool_layers", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载SME+U-Net checkpoint: {checkpoint_path} (epoch={checkpoint.get('epoch', '?')})")
    return model


def load_e2e_varnet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    a = checkpoint.get("args", {})
    model = E2EVarNet(
        num_cascades=a.get("num_cascades", 4),
        sme_chans=a.get("sme_chans", 8), sme_pool_layers=a.get("sme_pool_layers", 2),
        cascade_chans=a.get("cascade_chans", 12), cascade_pool_layers=a.get("cascade_pool_layers", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载E2E-VarNet checkpoint: {checkpoint_path} "
          f"(epoch={checkpoint.get('epoch', '?')}, num_cascades={a.get('num_cascades', 4)})")
    return model


@torch.no_grad()
def run_inference(model, undersampled_ks, mask_2d, num_low_freqs, target_shape, device):
    """对单张欠采样多线圈k-space做推理，两种模型(SMEUNetModel/E2EVarNet)接口一致，可复用"""
    num_coils, h, w = undersampled_ks.shape
    coil_images_zf = np.stack([ifft2c(undersampled_ks[c]) for c in range(num_coils)], axis=0)
    zf_rss = rss_combine(coil_images_zf)
    scale = zf_rss.std()
    if scale < 1e-12:
        scale = 1.0

    ks_norm = undersampled_ks / scale
    kspace_2ch = np.stack([ks_norm.real, ks_norm.imag], axis=1).astype(np.float32)
    mask_arr = mask_2d.astype(np.float32)[np.newaxis, :, :]

    kspace_tensor = torch.from_numpy(kspace_2ch).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).to(device)

    output = model(kspace_tensor, mask_tensor, num_low_freqs)
    output_np = output.squeeze().cpu().numpy() * scale

    return center_crop(output_np, target_shape), zf_rss


def evaluate_dataset(varnet_model, sme_unet_model, test_dir, acceleration,
                      target_num_coils, target_gt_shape, device,
                      use_coil_compression=True, fixed_seed=42,
                      compute_lpips=True, roi_top_fraction=0.5):
    files = sorted(Path(test_dir).glob("*.h5"))
    if len(files) == 0:
        raise FileNotFoundError(f"在 {test_dir} 下未找到.h5文件")

    center_fraction = ACCELERATION_CONFIGS[acceleration]["center_fraction"]
    records = []
    skipped = 0

    lpips_evaluator = None
    if compute_lpips:
        lpips_evaluator = LPIPSEvaluator(net="alex", device=device)
        print("LPIPS评估器已加载")

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            kspace = f["kspace"][()]
            gt = f["reconstruction_rss"][()]
            acquisition = f.attrs.get("acquisition", "unknown")

        num_slices, native_num_coils, h, w = kspace.shape

        if use_coil_compression:
            coil_ok = native_num_coils >= target_num_coils
        else:
            coil_ok = native_num_coils == target_num_coils

        if not coil_ok or tuple(gt.shape[1:]) != tuple(target_gt_shape):
            skipped += 1
            continue

        num_low_freqs = int(round(w * center_fraction))

        for slice_idx in range(num_slices):
            ks_slice = kspace[slice_idx]
            gt_slice = gt[slice_idx].astype(np.float32)
            target_shape = gt_slice.shape
            ks_slice_native = ks_slice  # 保留压缩前的版本，用于k-space可视化

            if native_num_coils > target_num_coils:
                ks_slice = compress_coils(ks_slice, num_low_freqs, target_num_coils)

            mask_2d, _, _ = get_mask_for_acceleration((h, w), acceleration=acceleration, seed=fixed_seed)
            undersampled_ks = ks_slice * mask_2d[np.newaxis, :, :]

            varnet_pred, zf_rss = run_inference(
                varnet_model, undersampled_ks, mask_2d, num_low_freqs, target_shape, device
            )
            zf_rss_cropped = center_crop(zf_rss, target_shape)

            bg_mask = compute_background_mask(gt_slice)
            roi_mask = compute_roi_mask(gt_slice, background_mask=bg_mask,
                                         top_fraction=roi_top_fraction)

            rss_psnr, rss_ssim = compute_metrics(zf_rss_cropped, gt_slice)
            rss_psnr_m, rss_ssim_m = compute_metrics(zf_rss_cropped, gt_slice, mask=bg_mask)
            rss_psnr_roi, rss_ssim_roi = compute_metrics(zf_rss_cropped, gt_slice, mask=roi_mask)

            varnet_psnr, varnet_ssim = compute_metrics(varnet_pred, gt_slice)
            varnet_psnr_m, varnet_ssim_m = compute_metrics(varnet_pred, gt_slice, mask=bg_mask)
            varnet_psnr_roi, varnet_ssim_roi = compute_metrics(varnet_pred, gt_slice, mask=roi_mask)

            full_kspace_vis = np.log(np.abs(ks_slice_native[0]) + 1e-9)
            undersampled_kspace_vis = np.log(np.abs(undersampled_ks[0]) + 1e-9)

            record = {
                "fname": fpath.name, "slice_idx": slice_idx, "acquisition": acquisition,
                "rss_psnr": rss_psnr, "rss_ssim": rss_ssim,
                "rss_psnr_masked": rss_psnr_m, "rss_ssim_masked": rss_ssim_m,
                "rss_psnr_roi": rss_psnr_roi, "rss_ssim_roi": rss_ssim_roi,
                "varnet_psnr": varnet_psnr, "varnet_ssim": varnet_ssim,
                "varnet_psnr_masked": varnet_psnr_m, "varnet_ssim_masked": varnet_ssim_m,
                "varnet_psnr_roi": varnet_psnr_roi, "varnet_ssim_roi": varnet_ssim_roi,
                "rss_image": zf_rss_cropped, "varnet_image": varnet_pred, "gt_image": gt_slice,
                "roi_mask": roi_mask,
                "full_kspace_vis": full_kspace_vis,
                "undersampled_kspace_vis": undersampled_kspace_vis,
            }

            if lpips_evaluator is not None:
                record["rss_lpips"] = lpips_evaluator.compute(zf_rss_cropped, gt_slice)
                record["varnet_lpips"] = lpips_evaluator.compute(varnet_pred, gt_slice)

            if sme_unet_model is not None:
                sme_pred, _ = run_inference(
                    sme_unet_model, undersampled_ks, mask_2d, num_low_freqs, target_shape, device
                )
                sme_psnr, sme_ssim = compute_metrics(sme_pred, gt_slice)
                sme_psnr_m, sme_ssim_m = compute_metrics(sme_pred, gt_slice, mask=bg_mask)
                sme_psnr_roi, sme_ssim_roi = compute_metrics(sme_pred, gt_slice, mask=roi_mask)
                record.update({
                    "sme_psnr": sme_psnr, "sme_ssim": sme_ssim,
                    "sme_psnr_masked": sme_psnr_m, "sme_ssim_masked": sme_ssim_m,
                    "sme_psnr_roi": sme_psnr_roi, "sme_ssim_roi": sme_ssim_roi,
                    "sme_image": sme_pred,
                })
                if lpips_evaluator is not None:
                    record["sme_lpips"] = lpips_evaluator.compute(sme_pred, gt_slice)

            records.append(record)

    if skipped > 0:
        print(f"(跳过 {skipped} 个不符合筛选条件的文件)")

    return records


def summarize(records, acceleration, has_sme):
    def avg(key):
        return np.mean([r[key] for r in records])

    has_lpips = "rss_lpips" in records[0]

    print(f"\n{'=' * 70}")
    print(f"{acceleration}x 加速 | 测试集共 {len(records)} 张切片")
    print(f"{'=' * 70}")

    for suffix, label in [("", "不加背景遮罩 - 全图评估"),
                           ("_masked", "加背景遮罩 - 仅前景区域评估"),
                           ("_roi", "加ROI遮罩 - 仅细节丰富区域评估(最贴近临床关注点)")]:
        print(f"\n【{label}】")
        print(f"{'方法':<20}{'PSNR (dB)':<15}{'SSIM':<10}")
        print(f"{'-' * 45}")
        print(f"{'RSS baseline':<20}{avg('rss_psnr'+suffix):<15.2f}{avg('rss_ssim'+suffix):<10.4f}")
        if has_sme:
            print(f"{'SME+U-Net':<20}{avg('sme_psnr'+suffix):<15.2f}{avg('sme_ssim'+suffix):<10.4f}")
        print(f"{'E2E-VarNet':<20}{avg('varnet_psnr'+suffix):<15.2f}{avg('varnet_ssim'+suffix):<10.4f}")
        print(f"{'-' * 45}")
        if has_sme:
            print(f"{'SME+U-Net 提升':<20}{avg('sme_psnr'+suffix)-avg('rss_psnr'+suffix):+.2f} dB{'':<7}"
                  f"{avg('sme_ssim'+suffix)-avg('rss_ssim'+suffix):+.4f}")
        print(f"{'E2E-VarNet 提升':<20}{avg('varnet_psnr'+suffix)-avg('rss_psnr'+suffix):+.2f} dB{'':<7}"
              f"{avg('varnet_ssim'+suffix)-avg('rss_ssim'+suffix):+.4f}")
        if has_sme:
            print(f"{'VarNet 相对SME+U-Net':<20}"
                  f"{avg('varnet_psnr'+suffix)-avg('sme_psnr'+suffix):+.2f} dB{'':<7}"
                  f"{avg('varnet_ssim'+suffix)-avg('sme_ssim'+suffix):+.4f}")

    result = {"acceleration": acceleration}
    for suffix in ["", "_masked", "_roi"]:
        result[f"rss_psnr{suffix}"] = avg(f"rss_psnr{suffix}")
        result[f"rss_ssim{suffix}"] = avg(f"rss_ssim{suffix}")
        result[f"varnet_psnr{suffix}"] = avg(f"varnet_psnr{suffix}")
        result[f"varnet_ssim{suffix}"] = avg(f"varnet_ssim{suffix}")
        if has_sme:
            result[f"sme_psnr{suffix}"] = avg(f"sme_psnr{suffix}")
            result[f"sme_ssim{suffix}"] = avg(f"sme_ssim{suffix}")

    if has_lpips:
        print(f"\n【LPIPS感知距离 - 数值越小越好】")
        print(f"{'方法':<20}{'LPIPS':<10}")
        print(f"{'-' * 30}")
        print(f"{'RSS baseline':<20}{avg('rss_lpips'):<10.4f}")
        if has_sme:
            print(f"{'SME+U-Net':<20}{avg('sme_lpips'):<10.4f}")
        print(f"{'E2E-VarNet':<20}{avg('varnet_lpips'):<10.4f}")
        print(f"{'-' * 30}")
        print(f"{'E2E-VarNet 改善':<20}{avg('rss_lpips')-avg('varnet_lpips'):+.4f}")
        if has_sme:
            print(f"{'VarNet 相对SME+U-Net':<20}{avg('sme_lpips')-avg('varnet_lpips'):+.4f} "
                  f"(正值表示VarNet更接近GT)")
        result["rss_lpips"] = avg("rss_lpips")
        result["varnet_lpips"] = avg("varnet_lpips")
        if has_sme:
            result["sme_lpips"] = avg("sme_lpips")

    return result


def save_qualitative_examples(records, save_dir, acceleration, has_sme, n_examples=3):
    save_dir = Path(save_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    sorted_records = sorted(records, key=lambda r: r["varnet_ssim"] - r["rss_ssim"])
    mid = len(sorted_records) // 2
    selected = sorted_records[max(0, mid - n_examples // 2): mid - n_examples // 2 + n_examples]

    n_cols = 4 if has_sme else 3

    for i, r in enumerate(selected):
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

        # 上排：k-space对比(完整 vs 降采样)
        axes[0, 0].imshow(r["full_kspace_vis"], cmap="gray")
        axes[0, 0].set_title("Full k-space (log magnitude)\n(single coil, for illustration)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(r["undersampled_kspace_vis"], cmap="gray")
        axes[0, 1].set_title(f"Undersampled k-space ({acceleration}x)")
        axes[0, 1].axis("off")

        for c in range(2, n_cols):
            axes[0, c].axis("off")

        # 下排：重建结果对比
        axes[1, 0].imshow(r["gt_image"], cmap="gray")
        axes[1, 0].contour(r["roi_mask"], colors="red", linewidths=0.8, levels=[0.5])
        axes[1, 0].set_title("Ground Truth (red = ROI boundary)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(r["rss_image"], cmap="gray")
        axes[1, 1].set_title(f"RSS ({acceleration}x)\n"
                              f"Full SSIM={r['rss_ssim']:.4f} | ROI SSIM={r['rss_ssim_roi']:.4f}")
        axes[1, 1].axis("off")

        col = 2
        if has_sme:
            axes[1, col].imshow(r["sme_image"], cmap="gray")
            axes[1, col].set_title(f"SME+U-Net ({acceleration}x)\n"
                                    f"Full SSIM={r['sme_ssim']:.4f} | ROI SSIM={r['sme_ssim_roi']:.4f}")
            axes[1, col].axis("off")
            col += 1

        axes[1, col].imshow(r["varnet_image"], cmap="gray")
        axes[1, col].set_title(f"E2E-VarNet ({acceleration}x)\n"
                                f"Full SSIM={r['varnet_ssim']:.4f} | ROI SSIM={r['varnet_ssim_roi']:.4f}")
        axes[1, col].axis("off")

        plt.tight_layout()
        save_path = save_dir / f"compare_{acceleration}x_{i}_{r['fname']}_s{r['slice_idx']}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存示例图: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="评估E2E-VarNet(可选与SME+U-Net三方对比)")
    parser.add_argument("--varnet_checkpoint", type=str, required=True)
    parser.add_argument("--sme_unet_checkpoint", type=str, default=None)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--acceleration", type=int, required=True, choices=[4, 8])
    parser.add_argument("--target_num_coils", type=int, default=8)
    parser.add_argument("--use_coil_compression", action="store_true", default=True)
    parser.add_argument("--no_coil_compression", dest="use_coil_compression",
                         action="store_false")
    parser.add_argument("--target_gt_h", type=int, default=384)
    parser.add_argument("--target_gt_w", type=int, default=384)
    parser.add_argument("--save_dir", type=str, default="results_e2e")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--compute_lpips", action="store_true", default=True)
    parser.add_argument("--no_lpips", dest="compute_lpips", action="store_false")
    parser.add_argument("--roi_top_fraction", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    varnet_model = load_e2e_varnet(args.varnet_checkpoint, device)
    sme_unet_model = load_sme_unet(args.sme_unet_checkpoint, device) if args.sme_unet_checkpoint else None
    has_sme = sme_unet_model is not None

    print(f"\n开始在测试集上评估...")
    records = evaluate_dataset(
        varnet_model, sme_unet_model, args.test_dir, args.acceleration,
        args.target_num_coils, (args.target_gt_h, args.target_gt_w), device,
        use_coil_compression=args.use_coil_compression, fixed_seed=args.seed,
        compute_lpips=args.compute_lpips, roi_top_fraction=args.roi_top_fraction,
    )

    summary = summarize(records, args.acceleration, has_sme)
    save_qualitative_examples(records, args.save_dir, args.acceleration, has_sme, args.n_examples)

    csv_path = Path(args.save_dir) / "metrics_e2e.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        base_cols = ["rss_psnr", "rss_ssim", "sme_psnr", "sme_ssim", "varnet_psnr", "varnet_ssim"] \
            if has_sme else ["rss_psnr", "rss_ssim", "varnet_psnr", "varnet_ssim"]
        all_cols = ["acceleration"]
        for suffix in ["", "_masked", "_roi"]:
            all_cols += [c + suffix for c in base_cols]
        if "rss_lpips" in summary:
            all_cols += ["rss_lpips", "varnet_lpips"]
            if has_sme:
                all_cols += ["sme_lpips"]
        if write_header:
            f.write(",".join(all_cols) + "\n")
        vals = [str(summary["acceleration"])]
        for c in all_cols[1:]:
            vals.append(f"{summary[c]:.4f}")
        f.write(",".join(vals) + "\n")
    print(f"\n汇总指标已追加至: {csv_path}")


if __name__ == "__main__":
    main()