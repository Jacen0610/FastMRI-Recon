"""
评估迭代式SME+U-Net，同时对比RSS baseline、原始SME+U-Net(单次)、E2E-VarNet，
完整呈现"是否更新灵敏度图"这个改动对最终效果的影响。

使用方法：
    python src/evaluate_iterative_sme_unet.py \
        --iter_checkpoint checkpoints/iter_sme_unet_8x/best.pth \
        --sme_unet_checkpoint checkpoints/sme_unet_8x/best.pth \
        --varnet_checkpoint checkpoints/e2e_varnet_8x/best.pth \
        --test_dir data/subset/test \
        --acceleration 8 \
        --target_num_coils 20 \
        --target_gt_h 384 --target_gt_w 384 \
        --save_dir results_iter_8x
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
from models.sme_unet_model import SMEUNetModel
from models.e2e_varnet import E2EVarNet
from models.iterative_sme_unet import IterativeSMEUNetModel


def optimal_scale_align(pred, gt):
    pred_flat = pred.flatten().astype(np.float64)
    gt_flat = gt.flatten().astype(np.float64)
    denom = np.dot(pred_flat, pred_flat)
    scale = np.dot(pred_flat, gt_flat) / denom if denom > 0 else 1.0
    return pred * scale


def compute_background_mask(gt, threshold_ratio=0.05):
    threshold = threshold_ratio * gt.max()
    mask = gt > threshold
    try:
        from scipy import ndimage
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
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


def load_sme_unet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    a = checkpoint.get("args", {})
    model = SMEUNetModel(
        sme_chans=a.get("sme_chans", 8), sme_pool_layers=a.get("sme_pool_layers", 2),
        unet_chans=a.get("unet_chans", 32), unet_pool_layers=a.get("unet_pool_layers", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载SME+U-Net checkpoint: {checkpoint_path}")
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
    print(f"已加载E2E-VarNet checkpoint: {checkpoint_path}")
    return model


def load_iterative_sme_unet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    a = checkpoint.get("args", {})
    model = IterativeSMEUNetModel(
        sme_chans=a.get("sme_chans", 8), sme_pool_layers=a.get("sme_pool_layers", 2),
        unet_chans=a.get("unet_chans", 32), unet_pool_layers=a.get("unet_pool_layers", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载迭代式SME+U-Net checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def run_inference(model, undersampled_ks, mask_2d, num_low_freqs, target_shape, device):
    """通用推理函数，三种模型(SMEUNetModel/E2EVarNet/IterativeSMEUNetModel)接口一致"""
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


def evaluate_dataset(models_dict, test_dir, acceleration, target_num_coils,
                      target_gt_shape, device, use_coil_compression=True, fixed_seed=42):
    """models_dict: {'rss': None, 'sme': model_or_None, 'varnet': model_or_None, 'iter_sme': model}"""
    files = sorted(Path(test_dir).glob("*.h5"))
    if len(files) == 0:
        raise FileNotFoundError(f"在 {test_dir} 下未找到.h5文件")

    center_fraction = ACCELERATION_CONFIGS[acceleration]["center_fraction"]
    records = []
    skipped = 0

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

            if native_num_coils > target_num_coils:
                ks_slice = compress_coils(ks_slice, num_low_freqs, target_num_coils)
            num_coils = ks_slice.shape[0]

            mask_2d, _, _ = get_mask_for_acceleration((h, w), acceleration=acceleration, seed=fixed_seed)
            undersampled_ks = ks_slice * mask_2d[np.newaxis, :, :]
            bg_mask = compute_background_mask(gt_slice)

            record = {"fname": fpath.name, "slice_idx": slice_idx, "acquisition": acquisition,
                      "gt_image": gt_slice}

            # RSS baseline
            coil_images_zf = np.stack([ifft2c(undersampled_ks[c]) for c in range(num_coils)], axis=0)
            zf_rss_cropped = center_crop(rss_combine(coil_images_zf), target_shape)
            record["rss_image"] = zf_rss_cropped
            record["rss_psnr"], record["rss_ssim"] = compute_metrics(zf_rss_cropped, gt_slice)
            record["rss_psnr_m"], record["rss_ssim_m"] = compute_metrics(zf_rss_cropped, gt_slice, mask=bg_mask)

            for key, model in models_dict.items():
                if model is None:
                    continue
                pred, _ = run_inference(model, undersampled_ks, mask_2d, num_low_freqs, target_shape, device)
                record[f"{key}_image"] = pred
                record[f"{key}_psnr"], record[f"{key}_ssim"] = compute_metrics(pred, gt_slice)
                record[f"{key}_psnr_m"], record[f"{key}_ssim_m"] = compute_metrics(pred, gt_slice, mask=bg_mask)

            records.append(record)

    if skipped > 0:
        print(f"(跳过 {skipped} 个不符合筛选条件的文件)")

    return records


def summarize(records, acceleration, method_keys):
    def avg(key):
        return np.mean([r[key] for r in records])

    print(f"\n{'=' * 75}")
    print(f"{acceleration}x 加速 | 测试集共 {len(records)} 张切片")
    print(f"{'=' * 75}")

    for suffix, label in [("", "不加背景遮罩 - 全图评估"), ("_m", "加背景遮罩 - 仅前景区域评估")]:
        print(f"\n【{label}】")
        print(f"{'方法':<20}{'PSNR (dB)':<15}{'SSIM':<10}")
        print(f"{'-' * 45}")
        print(f"{'RSS baseline':<20}{avg('rss_psnr'+suffix):<15.2f}{avg('rss_ssim'+suffix):<10.4f}")
        for key, label_name in method_keys:
            print(f"{label_name:<20}{avg(key+'_psnr'+suffix):<15.2f}{avg(key+'_ssim'+suffix):<10.4f}")

    summary = {"acceleration": acceleration}
    for suffix in ["", "_m"]:
        summary[f"rss_psnr{suffix}"] = avg(f"rss_psnr{suffix}")
        summary[f"rss_ssim{suffix}"] = avg(f"rss_ssim{suffix}")
        for key, _ in method_keys:
            summary[f"{key}_psnr{suffix}"] = avg(f"{key}_psnr{suffix}")
            summary[f"{key}_ssim{suffix}"] = avg(f"{key}_ssim{suffix}")
    return summary


def save_qualitative_examples(records, save_dir, acceleration, method_keys, n_examples=3):
    save_dir = Path(save_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    sort_key = method_keys[-1][0] if method_keys else "rss"
    sorted_records = sorted(records, key=lambda r: r.get(f"{sort_key}_ssim", 0) - r["rss_ssim"])
    mid = len(sorted_records) // 2
    selected = sorted_records[max(0, mid - n_examples // 2): mid - n_examples // 2 + n_examples]

    n_panels = 2 + len(method_keys)

    for i, r in enumerate(selected):
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

        axes[0].imshow(r["gt_image"], cmap="gray")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(r["rss_image"], cmap="gray")
        axes[1].set_title(f"RSS\nSSIM={r['rss_ssim']:.4f}")
        axes[1].axis("off")

        for j, (key, label_name) in enumerate(method_keys):
            axes[2 + j].imshow(r[f"{key}_image"], cmap="gray")
            axes[2 + j].set_title(f"{label_name}\nSSIM={r[f'{key}_ssim']:.4f}")
            axes[2 + j].axis("off")

        plt.tight_layout()
        save_path = save_dir / f"compare_{acceleration}x_{i}_{r['fname']}_s{r['slice_idx']}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  保存示例图: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="评估迭代式SME+U-Net(多方对比)")
    parser.add_argument("--iter_checkpoint", type=str, required=True)
    parser.add_argument("--sme_unet_checkpoint", type=str, default=None)
    parser.add_argument("--varnet_checkpoint", type=str, default=None)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--acceleration", type=int, required=True, choices=[4, 8])
    parser.add_argument("--target_num_coils", type=int, default=8)
    parser.add_argument("--use_coil_compression", action="store_true", default=True)
    parser.add_argument("--no_coil_compression", dest="use_coil_compression",
                         action="store_false")
    parser.add_argument("--target_gt_h", type=int, default=384)
    parser.add_argument("--target_gt_w", type=int, default=384)
    parser.add_argument("--save_dir", type=str, default="results_iter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_examples", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    models_dict = {}
    method_keys = []

    if args.sme_unet_checkpoint:
        models_dict["sme"] = load_sme_unet(args.sme_unet_checkpoint, device)
        method_keys.append(("sme", "SME+U-Net (single-pass)"))
    if args.varnet_checkpoint:
        models_dict["varnet"] = load_e2e_varnet(args.varnet_checkpoint, device)
        method_keys.append(("varnet", "E2E-VarNet"))

    models_dict["iter_sme"] = load_iterative_sme_unet(args.iter_checkpoint, device)
    method_keys.append(("iter_sme", "Iterative SME+U-Net"))

    print(f"\n开始在测试集上评估...")
    records = evaluate_dataset(
        models_dict, args.test_dir, args.acceleration, args.target_num_coils,
        (args.target_gt_h, args.target_gt_w), device,
        use_coil_compression=args.use_coil_compression, fixed_seed=args.seed,
    )

    summary = summarize(records, args.acceleration, method_keys)
    save_qualitative_examples(records, args.save_dir, args.acceleration, method_keys, args.n_examples)

    csv_path = Path(args.save_dir) / "metrics_iter.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        cols = ["acceleration", "rss_psnr", "rss_ssim", "rss_psnr_m", "rss_ssim_m"]
        for key, _ in method_keys:
            cols += [f"{key}_psnr", f"{key}_ssim", f"{key}_psnr_m", f"{key}_ssim_m"]
        f.write(",".join(cols) + "\n")
        vals = [str(summary["acceleration"])]
        for c in cols[1:]:
            vals.append(f"{summary[c]:.4f}")
        f.write(",".join(vals) + "\n")
    print(f"\n汇总指标已保存至: {csv_path}")


if __name__ == "__main__":
    main()