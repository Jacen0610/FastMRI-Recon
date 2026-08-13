"""
在"GT尺寸固定为384x384"这个前提下，扫描AXT2文件的线圈数量分布，
用于评估：如果用线圈压缩统一线圈数，理论上能额外捞回多少数据
(即目前因线圈数不是20而被排除、但GT尺寸已经匹配的那部分文件)。

使用方法：
    python scan_coil_within_gt_size.py --data_dir <目录> --acquisition AXT2 --target_gt_h 384 --target_gt_w 384
"""

import argparse
import glob
import os
from collections import Counter

import h5py


def main():
    parser = argparse.ArgumentParser(description="在固定GT尺寸下扫描线圈数量分布")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--acquisition", type=str, default="AXT2")
    parser.add_argument("--target_gt_h", type=int, default=384)
    parser.add_argument("--target_gt_w", type=int, default=384)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    print(f"共找到 {len(files)} 个.h5文件，开始扫描...")

    coil_count_dist = Counter()
    matched_count = 0

    for i, fpath in enumerate(files):
        try:
            with h5py.File(fpath, "r") as f:
                acquisition = f.attrs.get("acquisition", "unknown")
                if args.acquisition is not None and acquisition != args.acquisition:
                    continue
                gt_shape = f["reconstruction_rss"].shape[1:]
                if tuple(gt_shape) != (args.target_gt_h, args.target_gt_w):
                    continue
                num_coils = f["kspace"].shape[1]
                coil_count_dist[num_coils] += 1
                matched_count += 1
        except Exception as e:
            print(f"  跳过无法读取的文件 {fpath}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  已扫描 {i + 1}/{len(files)}")

    print(f"\n{'=' * 60}")
    print(f"筛选条件: acquisition={args.acquisition}, "
          f"gt_shape=({args.target_gt_h}, {args.target_gt_w})")
    print(f"符合GT尺寸条件的文件总数: {matched_count}")
    print(f"{'=' * 60}")

    print(f"\n该GT尺寸下的线圈数量分布:")
    for num_coils, count in sorted(coil_count_dist.items()):
        bar = "#" * min(count, 50)
        print(f"  {num_coils:>3}个线圈: {count:>4}个文件  {bar}")

    current_usable = coil_count_dist.get(20, 0)
    total_after_compression = matched_count  # 理论上线圈压缩后全部可用
    print(f"\n当前(不压缩，仅用20线圈)可用: {current_usable} 个文件")
    print(f"若用线圈压缩统一线圈数，理论上限可用: {total_after_compression} 个文件"
          f" (提升 {total_after_compression - current_usable} 个，"
          f"约 {(total_after_compression/current_usable - 1)*100:.0f}% 增幅)"
          if current_usable > 0 else "")


if __name__ == "__main__":
    main()