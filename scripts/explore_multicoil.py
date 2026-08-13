"""
探索 fastMRI brain multicoil 数据的内部结构，重点验证：
- k-space的shape是否包含"线圈"这一维度，具体有多少个线圈
- Ground Truth字段是什么(multicoil通常是 reconstruction_rss，不是singlecoil的 _esc)
- 用RSS(Root Sum of Squares)合并多线圈图像，验证合并逻辑是否正确

使用方法：
    python explore_multicoil.py --file /path/to/data/file_brain_AXT2_xxx.h5
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def ifft2c(kspace):
    """中心化正交归一化二维逆傅里叶变换，与singlecoil项目中保持一致的约定"""
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def rss_combine(multicoil_images):
    """
    Root Sum of Squares 合并：把多个线圈各自重建的图像，逐像素做平方和再开根号。
    这是最经典、最简单的多线圈合并方式，不需要额外估计线圈灵敏度。

    参数:
        multicoil_images: shape (num_coils, H, W)，每个线圈独立IFFT后的复数图像

    返回:
        combined: shape (H, W)，合并后的单张幅度图像
    """
    return np.sqrt(np.sum(np.abs(multicoil_images) ** 2, axis=0))


def main():
    parser = argparse.ArgumentParser(description="探索brain multicoil数据结构")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--slice_idx", type=int, default=None)
    parser.add_argument("--save", type=str, default="explore_multicoil_output.png")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        print("=" * 60)
        print(f"文件: {args.file}")
        print("=" * 60)

        print("\n[顶层 keys]")
        for key in f.keys():
            print(f"  - {key}: shape={f[key].shape}, dtype={f[key].dtype}")

        print("\n[文件 attrs]")
        for k, v in f.attrs.items():
            print(f"  - {k}: {v}")

        kspace = f["kspace"][()]

        # multicoil的GT字段通常是 reconstruction_rss，而不是singlecoil的 reconstruction_esc
        recon_key = None
        for candidate in ["reconstruction_rss", "reconstruction_esc"]:
            if candidate in f.keys():
                recon_key = candidate
                break
        target = f[recon_key][()] if recon_key else None

    print(f"\n[k-space 形状分析]")
    print(f"  完整shape: {kspace.shape}")
    if kspace.ndim == 4:
        num_slices, num_coils, h, w = kspace.shape
        print(f"  -> {num_slices}个切片, {num_coils}个线圈, 每个线圈k-space尺寸 {h}x{w}")
    else:
        print(f"  警告: k-space维度数({kspace.ndim})不是预期的4维，"
              f"可能不是multicoil数据，请确认文件来源")
        return

    if target is not None:
        print(f"\n[Ground Truth] 使用字段: {recon_key}, shape={target.shape}")

    slice_idx = args.slice_idx if args.slice_idx is not None else num_slices // 2
    print(f"\n可视化第 {slice_idx} 张切片（共 {num_slices} 张）")

    ks_slice = kspace[slice_idx]  # shape (num_coils, H, W)

    # 对每个线圈独立做IFFT，得到各自的复数图像
    coil_images = np.stack([ifft2c(ks_slice[c]) for c in range(num_coils)], axis=0)

    # RSS合并
    combined = rss_combine(coil_images)

    # 绘图：展示前4个线圈各自的图像 + RSS合并结果 + GT(如有)
    n_show_coils = min(4, num_coils)
    n_panels = n_show_coils + 1 + (1 if target is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    for i in range(n_show_coils):
        axes[i].imshow(np.abs(coil_images[i]), cmap="gray")
        axes[i].set_title(f"Coil {i}")
        axes[i].axis("off")

    axes[n_show_coils].imshow(combined, cmap="gray")
    axes[n_show_coils].set_title(f"RSS combined\n({num_coils} coils)")
    axes[n_show_coils].axis("off")

    if target is not None:
        target_slice = target[slice_idx]
        axes[-1].imshow(target_slice, cmap="gray")
        axes[-1].set_title(f"Ground Truth\n({recon_key})")
        axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\n可视化结果已保存至: {args.save}")

    if target is not None:
        print(f"\n[尺寸检查]")
        print(f"  RSS合并结果 shape: {combined.shape}")
        print(f"  Ground Truth shape: {target_slice.shape}")
        if combined.shape != target_slice.shape:
            print("  注意: 形状不一致，可能需要center crop对齐，"
                  "与singlecoil项目中遇到的情况类似")
        else:
            print("  形状一致，说明官方GT可能就是用RSS(或近似方法)对完整k-space合并得到的")


if __name__ == "__main__":
    main()