"""
扫描指定目录下所有(或指定序列类型)的.h5文件，统计各文件的线圈数量分布。
用于在训练前确认：应该筛选出线圈数量为多少的子集，能覆盖最多的可用文件。

使用方法：
    python scan_coil_counts.py --data_dir <目录路径> --acquisition AXT2
"""

import argparse
import glob
import os
from collections import Counter

import h5py


def main():
    parser = argparse.ArgumentParser(description="扫描线圈数量分布")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--acquisition", type=str, default=None,
                         help="只统计指定序列类型(如AXT2)，不指定则统计全部")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    print(f"共找到 {len(files)} 个.h5文件，开始扫描...")

    coil_count_dist = Counter()
    matched_files = []

    for i, fpath in enumerate(files):
        try:
            with h5py.File(fpath, "r") as f:
                acquisition = f.attrs.get("acquisition", "unknown")
                if args.acquisition is not None and acquisition != args.acquisition:
                    continue
                num_coils = f["kspace"].shape[1]
                coil_count_dist[num_coils] += 1
                matched_files.append((os.path.basename(fpath), num_coils))
        except Exception as e:
            print(f"  跳过无法读取的文件 {fpath}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  已扫描 {i + 1}/{len(files)}")

    print(f"\n{'=' * 50}")
    print(f"筛选条件: acquisition={args.acquisition or '全部'}")
    print(f"符合条件的文件总数: {len(matched_files)}")
    print(f"{'=' * 50}")

    print(f"\n线圈数量分布:")
    for num_coils, count in sorted(coil_count_dist.items()):
        bar = "#" * min(count, 50)
        print(f"  {num_coils:>3}个线圈: {count:>4}个文件  {bar}")

    if coil_count_dist:
        most_common_coils, most_common_count = coil_count_dist.most_common(1)[0]
        print(f"\n建议：筛选线圈数={most_common_coils}的文件作为训练子集，"
              f"可用文件数={most_common_count} "
              f"(占符合条件文件总数的 {most_common_count/len(matched_files)*100:.1f}%)")


if __name__ == "__main__":
    main()