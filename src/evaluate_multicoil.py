"""
在测试集上评估训练好的 SME+U-Net 模型，对比 RSS(Root Sum of Squares) baseline，
产出定量指标(PSNR/SSIM)汇总表和可视化对比图。

使用方法：
    python src/evaluate_multicoil.py \
        --checkpoint checkpoints/sme_unet_4x/best.pth \
        --test_dir data/subset/test \
        --acceleration 4 \
        --target_num_coils 20 \
        --target_gt_h 384 --target_gt_w 384 \
        --save_dir results
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
from data.roi_mask import compute_roi_mask, compute_background_mask
from models.sme_unet_model import SMEUNetModel

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def optimal_scale_align(pred, gt):
    """最小二乘最优尺度对齐，与singlecoil项目中的方法保持一致"""
    pred_flat = pred.flatten().astype(np.float64)
    gt_flat = gt.flatten().astype(np.float64)
    denom = np.dot(pred_flat, pred_flat)
    scale = np.dot(pred_flat, gt_flat) / denom if denom > 0 else 1.0
    return pred * scale



def compute_metrics(pred, gt, mask=None):
    """计算PSNR和SSIM，可选加背景遮罩"""
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
    """Root Sum of Squares 合并，作为baseline方法"""
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))


class LPIPSEvaluator:
    """
    纯评估用的LPIPS封装，与训练时losses_with_lpips.py中的用法类似，
    但这里只做前向推理打分，不涉及梯度/训练。全局只初始化一次网络，
    避免每次调用都重新加载模型。
    """

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
        """
        参数:
            pred, gt: 2D np.ndarray，单张灰度图像(数值范围任意)
        返回:
            lpips_score: float，越小越好(0表示完全一致)
        """
        pred_aligned = optimal_scale_align(pred, gt)

        def to_tensor(img):
            img_min, img_max = img.min(), img.max()
            img_norm = (img - img_min) / (img_max - img_min + 1e-8) * 2 - 1
            t = torch.from_numpy(img_norm.astype(np.float32))
            t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # (1,3,H,W)
            return t.to(self.device)

        pred_t = to_tensor(pred_aligned)
        gt_t = to_tensor(gt)
        score = self.lpips_fn(pred_t, gt_t).item()
        return score


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    model = SMEUNetModel(
        sme_chans=train_args.get("sme_chans", 8),
        sme_pool_layers=train_args.get("sme_pool_layers", 2),
        unet_chans=train_args.get("unet_chans", 32),
        unet_pool_layers=train_args.get("unet_pool_layers", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载checkpoint: {checkpoint_path} "
          f"(epoch={checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', float('nan')):.6f})")
    return model


@torch.no_grad()
def run_model_inference(model, undersampled_ks, mask_2d, num_low_freqs, target_shape, device):
    """
    对单张欠采样多线圈k-space做SME+U-Net推理。
    归一化流程需与训练时(dataset_multicoil.py)严格一致。
    """
    num_coils, h, w = undersampled_ks.shape

    coil_images_zf = np.stack(
        [ifft2c(undersampled_ks[c]) for c in range(num_coils)], axis=0
    )
    zf_rss = rss_combine(coil_images_zf)
    scale = zf_rss.std()
    if scale < 1e-12:
        scale = 1.0

    ks_norm = undersampled_ks / scale
    kspace_2ch = np.stack([ks_norm.real, ks_norm.imag], axis=1).astype(np.float32)
    mask_arr = mask_2d.astype(np.float32)[np.newaxis, :, :]

    kspace_tensor = torch.from_numpy(kspace_2ch).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).to(device)

    output = model(kspace_tensor, mask_tensor, num_low_freqs)  # (1, 1, H, W)
    output_np = output.squeeze().cpu().numpy() * scale

    return center_crop(output_np, target_shape), zf_rss


def evaluate_dataset(model, test_dir, acceleration, target_num_coils, target_gt_shape,
                      device, use_coil_compression=True, fixed_seed=42,
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

            # 线圈压缩：与训练时(dataset_multicoil.py)保持一致的处理顺序，
            # 先压缩到目标虚拟线圈数，再施加欠采样mask
            if native_num_coils > target_num_coils:
                ks_slice = compress_coils(ks_slice, num_low_freqs, target_num_coils)

            mask_2d, _, _ = get_mask_for_acceleration(
                (h, w), acceleration=acceleration, seed=fixed_seed
            )
            undersampled_ks = ks_slice * mask_2d[np.newaxis, :, :]

            model_pred, zf_rss = run_model_inference(
                model, undersampled_ks, mask_2d, num_low_freqs, target_shape, device
            )
            zf_rss_cropped = center_crop(zf_rss, target_shape)

            bg_mask = compute_background_mask(gt_slice)
            roi_mask = compute_roi_mask(gt_slice, background_mask=bg_mask,
                                         top_fraction=roi_top_fraction)

            rss_psnr, rss_ssim = compute_metrics(zf_rss_cropped, gt_slice)
            rss_psnr_m, rss_ssim_m = compute_metrics(zf_rss_cropped, gt_slice, mask=bg_mask)
            rss_psnr_roi, rss_ssim_roi = compute_metrics(zf_rss_cropped, gt_slice, mask=roi_mask)

            model_psnr, model_ssim = compute_metrics(model_pred, gt_slice)
            model_psnr_m, model_ssim_m = compute_metrics(model_pred, gt_slice, mask=bg_mask)
            model_psnr_roi, model_ssim_roi = compute_metrics(model_pred, gt_slice, mask=roi_mask)

            # k-space可视化(取第一个线圈作为代表，纯粹用于展示"原始输入长什么样"，
            # 不参与任何指标计算)：对数幅度谱，完整k-space vs 降采样(欠采样)后的k-space
            full_kspace_vis = np.log(np.abs(ks_slice[0]) + 1e-9)
            undersampled_kspace_vis = np.log(np.abs(undersampled_ks[0]) + 1e-9)

            record = {
                "fname": fpath.name, "slice_idx": slice_idx, "acquisition": acquisition,
                "rss_psnr": rss_psnr, "rss_ssim": rss_ssim,
                "rss_psnr_masked": rss_psnr_m, "rss_ssim_masked": rss_ssim_m,
                "rss_psnr_roi": rss_psnr_roi, "rss_ssim_roi": rss_ssim_roi,
                "model_psnr": model_psnr, "model_ssim": model_ssim,
                "model_psnr_masked": model_psnr_m, "model_ssim_masked": model_ssim_m,
                "model_psnr_roi": model_psnr_roi, "model_ssim_roi": model_ssim_roi,
                "rss_image": zf_rss_cropped, "model_image": model_pred, "gt_image": gt_slice,
                "roi_mask": roi_mask,
                "full_kspace_vis": full_kspace_vis,
                "undersampled_kspace_vis": undersampled_kspace_vis,
            }

            if lpips_evaluator is not None:
                record["rss_lpips"] = lpips_evaluator.compute(zf_rss_cropped, gt_slice)
                record["model_lpips"] = lpips_evaluator.compute(model_pred, gt_slice)

            records.append(record)

    if skipped > 0:
        print(f"(跳过 {skipped} 个不符合筛选条件(线圈数/GT尺寸)的文件)")

    return records


def summarize(records, acceleration):
    def avg(key):
        return np.mean([r[key] for r in records])

    has_lpips = "rss_lpips" in records[0]

    print(f"\n{'=' * 60}")
    print(f"{acceleration}x 加速 | 测试集共 {len(records)} 张切片")
    print(f"{'=' * 60}")

    print(f"\n【不加背景遮罩 - 全图评估】")
    print(f"{'方法':<20}{'PSNR (dB)':<15}{'SSIM':<10}")
    print(f"{'-' * 45}")
    print(f"{'RSS baseline':<20}{avg('rss_psnr'):<15.2f}{avg('rss_ssim'):<10.4f}")
    print(f"{'SME+U-Net':<20}{avg('model_psnr'):<15.2f}{avg('model_ssim'):<10.4f}")
    print(f"{'-' * 45}")
    print(f"{'提升幅度':<20}{avg('model_psnr')-avg('rss_psnr'):+.2f} dB{'':<7}"
          f"{avg('model_ssim')-avg('rss_ssim'):+.4f}")

    print(f"\n【加背景遮罩 - 仅前景区域评估】")
    print(f"{'方法':<20}{'PSNR (dB)':<15}{'SSIM':<10}")
    print(f"{'-' * 45}")
    print(f"{'RSS baseline':<20}{avg('rss_psnr_masked'):<15.2f}{avg('rss_ssim_masked'):<10.4f}")
    print(f"{'SME+U-Net':<20}{avg('model_psnr_masked'):<15.2f}{avg('model_ssim_masked'):<10.4f}")
    print(f"{'-' * 45}")
    print(f"{'提升幅度':<20}{avg('model_psnr_masked')-avg('rss_psnr_masked'):+.2f} dB{'':<7}"
          f"{avg('model_ssim_masked')-avg('rss_ssim_masked'):+.4f}")

    print(f"\n【加ROI遮罩 - 仅细节丰富区域评估(最贴近临床关注点)】")
    print(f"{'方法':<20}{'PSNR (dB)':<15}{'SSIM':<10}")
    print(f"{'-' * 45}")
    print(f"{'RSS baseline':<20}{avg('rss_psnr_roi'):<15.2f}{avg('rss_ssim_roi'):<10.4f}")
    print(f"{'SME+U-Net':<20}{avg('model_psnr_roi'):<15.2f}{avg('model_ssim_roi'):<10.4f}")
    print(f"{'-' * 45}")
    print(f"{'提升幅度':<20}{avg('model_psnr_roi')-avg('rss_psnr_roi'):+.2f} dB{'':<7}"
          f"{avg('model_ssim_roi')-avg('rss_ssim_roi'):+.4f}")

    result = {
        "acceleration": acceleration,
        "rss_psnr": avg("rss_psnr"), "rss_ssim": avg("rss_ssim"),
        "model_psnr": avg("model_psnr"), "model_ssim": avg("model_ssim"),
        "rss_psnr_masked": avg("rss_psnr_masked"), "rss_ssim_masked": avg("rss_ssim_masked"),
        "model_psnr_masked": avg("model_psnr_masked"), "model_ssim_masked": avg("model_ssim_masked"),
        "rss_psnr_roi": avg("rss_psnr_roi"), "rss_ssim_roi": avg("rss_ssim_roi"),
        "model_psnr_roi": avg("model_psnr_roi"), "model_ssim_roi": avg("model_ssim_roi"),
    }

    if has_lpips:
        print(f"\n【LPIPS感知距离 - 数值越小越好】")
        print(f"{'方法':<20}{'LPIPS':<10}")
        print(f"{'-' * 30}")
        print(f"{'RSS baseline':<20}{avg('rss_lpips'):<10.4f}")
        print(f"{'SME+U-Net':<20}{avg('model_lpips'):<10.4f}")
        print(f"{'-' * 30}")
        print(f"{'改善幅度':<20}{avg('rss_lpips')-avg('model_lpips'):+.4f} "
              f"(正值表示模型比baseline更接近GT)")
        result["rss_lpips"] = avg("rss_lpips")
        result["model_lpips"] = avg("model_lpips")

    return result


def save_qualitative_examples(records, save_dir, acceleration, n_examples=3):
    save_dir = Path(save_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    sorted_records = sorted(records, key=lambda r: r["model_ssim"] - r["rss_ssim"])
    mid = len(sorted_records) // 2
    selected = sorted_records[max(0, mid - n_examples // 2): mid - n_examples // 2 + n_examples]

    for i, r in enumerate(selected):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 上排：k-space本身(完整 vs 降采样)，帮助读者直观理解"模型的原始输入长什么样"
        axes[0, 0].imshow(r["full_kspace_vis"], cmap="gray")
        axes[0, 0].set_title("Full k-space (log magnitude)\n(single coil, for illustration)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(r["undersampled_kspace_vis"], cmap="gray")
        axes[0, 1].set_title(f"Undersampled k-space ({acceleration}x)\n(log magnitude)")
        axes[0, 1].axis("off")

        axes[0, 2].axis("off")  # 占位，保持网格对齐

        # 下排：重建结果对比(GT / RSS baseline / 模型输出)
        axes[1, 0].imshow(r["gt_image"], cmap="gray")
        axes[1, 0].contour(r["roi_mask"], colors="red", linewidths=0.8, levels=[0.5])
        axes[1, 0].set_title("Ground Truth (red = ROI boundary)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(r["rss_image"], cmap="gray")
        axes[1, 1].set_title(f"RSS baseline ({acceleration}x)\n"
                              f"Full SSIM={r['rss_ssim']:.4f} | ROI SSIM={r['rss_ssim_roi']:.4f}")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(r["model_image"], cmap="gray")
        axes[1, 2].set_title(f"SME+U-Net ({acceleration}x)\n"
                              f"Full SSIM={r['model_ssim']:.4f} | ROI SSIM={r['model_ssim_roi']:.4f}")
        axes[1, 2].axis("off")

        plt.tight_layout()
        save_path = save_dir / f"compare_{acceleration}x_{i}_{r['fname']}_s{r['slice_idx']}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存示例图: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="评估SME+U-Net模型效果")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--acceleration", type=int, required=True, choices=[4, 8])
    parser.add_argument("--target_num_coils", type=int, default=8)
    parser.add_argument("--use_coil_compression", action="store_true", default=True)
    parser.add_argument("--no_coil_compression", dest="use_coil_compression",
                         action="store_false")
    parser.add_argument("--target_gt_h", type=int, default=384)
    parser.add_argument("--target_gt_w", type=int, default=384)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--compute_lpips", action="store_true", default=True,
                         help="是否计算LPIPS感知距离(需要lpips库，且会增加评估耗时)")
    parser.add_argument("--no_lpips", dest="compute_lpips", action="store_false")
    parser.add_argument("--roi_top_fraction", type=float, default=0.5,
                         help="前景区域内，保留局部方差最高的比例作为ROI")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_model(args.checkpoint, device)

    print(f"\n开始在测试集上评估...")
    records = evaluate_dataset(
        model, args.test_dir, args.acceleration, args.target_num_coils,
        (args.target_gt_h, args.target_gt_w), device,
        use_coil_compression=args.use_coil_compression, fixed_seed=args.seed,
        compute_lpips=args.compute_lpips, roi_top_fraction=args.roi_top_fraction,
    )

    summary = summarize(records, args.acceleration)
    save_qualitative_examples(records, args.save_dir, args.acceleration, args.n_examples)

    csv_path = Path(args.save_dir) / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            cols = ["acceleration", "rss_psnr", "rss_ssim", "model_psnr", "model_ssim",
                    "rss_psnr_masked", "rss_ssim_masked", "model_psnr_masked", "model_ssim_masked",
                    "rss_psnr_roi", "rss_ssim_roi", "model_psnr_roi", "model_ssim_roi"]
            if "rss_lpips" in summary:
                cols += ["rss_lpips", "model_lpips"]
            f.write(",".join(cols) + "\n")
        vals = [str(summary["acceleration"])]
        for c in ["rss_psnr", "rss_ssim", "model_psnr", "model_ssim",
                  "rss_psnr_masked", "rss_ssim_masked", "model_psnr_masked", "model_ssim_masked",
                  "rss_psnr_roi", "rss_ssim_roi", "model_psnr_roi", "model_ssim_roi"]:
            vals.append(f"{summary[c]:.4f}")
        if "rss_lpips" in summary:
            vals += [f"{summary['rss_lpips']:.4f}", f"{summary['model_lpips']:.4f}"]
        f.write(",".join(vals) + "\n")
    print(f"\n汇总指标已追加至: {csv_path}")


if __name__ == "__main__":
    main()