"""
从brain multicoil数据集里，筛选出符合条件的文件(支持线圈压缩：线圈数>=目标值即可)，
再随机划分为 train/val/test，用软链接方式组织（不复制文件，节省磁盘空间）。

使用方法：
    python split_multicoil_subset.py \
        --source ~/mri_data_brain/raw/brain_multicoil_train \
        --output ~/mri_data_brain/subset \
        --acquisition AXT2 \
        --target_num_coils 8 \
        --use_coil_compression \
        --n_train 130 \
        --n_val 17 \
        --n_test 17 \
        --seed 42
"""

import argparse
import os
import random
from collections import Counter
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser(description="brain multicoil数据筛选+划分")
    parser.add_argument("--source", type=str, required=True, help="解压后.h5文件所在目录")
    parser.add_argument("--output", type=str, required=True, help="划分输出根目录")
    parser.add_argument("--acquisition", type=str, default="AXT2",
                         help="只使用指定序列类型，None表示不筛选")
    parser.add_argument("--target_num_coils", type=int, default=8,
                         help="目标虚拟线圈数(压缩后)。配合--use_coil_compression使用")
    parser.add_argument("--use_coil_compression", action="store_true", default=True,
                         help="线圈数>=target_num_coils的文件均视为符合条件(压缩使用)；"
                              "关闭则退回严格模式(线圈数必须恰好等于target_num_coils)")
    parser.add_argument("--no_coil_compression", dest="use_coil_compression",
                         action="store_false")
    parser.add_argument("--target_gt_h", type=int, default=None,
                         help="GT高度筛选标准，需与--target_gt_w同时指定。"
                              "不指定则自动选数据集内最常见的GT尺寸")
    parser.add_argument("--target_gt_w", type=int, default=None)
    parser.add_argument("--n_train", type=int, default=130)
    parser.add_argument("--n_val", type=int, default=17)
    parser.add_argument("--n_test", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"源目录不存在: {source}")

    all_files = sorted(source.glob("*.h5"))
    print(f"源目录共有 {len(all_files)} 个 .h5 文件，开始筛选...")

    candidates = []
    gt_shape_dist = Counter()
    native_coil_dist = Counter()
    for i, fpath in enumerate(all_files):
        try:
            with h5py.File(fpath, "r") as f:
                acquisition = f.attrs.get("acquisition", "unknown")
                if args.acquisition is not None and acquisition != args.acquisition:
                    continue
                num_coils = f["kspace"].shape[1]

                if args.use_coil_compression:
                    if num_coils < args.target_num_coils:
                        continue  # 线圈数不足，无法压缩到目标值
                else:
                    if num_coils != args.target_num_coils:
                        continue  # 严格模式

                gt_shape = f["reconstruction_rss"].shape[1:]
                gt_shape_dist[gt_shape] += 1
                candidates.append((fpath, gt_shape, num_coils))
        except Exception as e:
            print(f"  跳过无法读取的文件 {fpath.name}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  已扫描 {i + 1}/{len(all_files)}")

    if len(candidates) == 0:
        raise ValueError(
            f"筛选后没有符合条件的文件 (acquisition={args.acquisition}, "
            f"target_num_coils={args.target_num_coils}, "
            f"use_coil_compression={args.use_coil_compression})，请检查筛选条件"
        )

    if args.target_gt_h is not None and args.target_gt_w is not None:
        target_gt_shape = (args.target_gt_h, args.target_gt_w)
    else:
        target_gt_shape = gt_shape_dist.most_common(1)[0][0]
        print(f"自动检测到最常见GT尺寸: {target_gt_shape} "
              f"(完整分布: {dict(gt_shape_dist)})")

    matched = [(fpath, num_coils) for fpath, gt_shape, num_coils in candidates
               if tuple(gt_shape) == tuple(target_gt_shape)]
    matched_files = [fpath for fpath, _ in matched]

    used_coil_dist = Counter(num_coils for _, num_coils in matched)

    total = len(matched_files)
    print(f"\n筛选条件: acquisition={args.acquisition}, "
          f"target_num_coils={args.target_num_coils}, "
          f"use_coil_compression={args.use_coil_compression}, gt_shape={target_gt_shape}")
    print(f"符合条件的文件总数: {total}")
    print(f"原始线圈数分布: {dict(sorted(used_coil_dist.items()))}")

    needed = args.n_train + args.n_val + args.n_test
    if total < needed:
        raise ValueError(
            f"筛选后文件数量不足：现有 {total} 个，需要抽取 {needed} 个（train+val+test）。"
            f"请降低 --n_train/--n_val/--n_test，或放宽筛选条件。"
        )

    # 固定随机种子，保证可复现
    random.seed(args.seed)
    shuffled = matched_files.copy()
    random.shuffle(shuffled)

    train_files = shuffled[: args.n_train]
    val_files = shuffled[args.n_train : args.n_train + args.n_val]
    test_files = shuffled[args.n_train + args.n_val : args.n_train + args.n_val + args.n_test]

    splits = {"train": train_files, "val": val_files, "test": test_files}

    for split_name, files in splits.items():
        split_dir = output / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for existing in split_dir.glob("*.h5"):
            if existing.is_symlink():
                existing.unlink()

        for f in files:
            link_path = split_dir / f.name
            if not link_path.exists():
                os.symlink(f, link_path)

        print(f"[{split_name}] 已创建 {len(files)} 个软链接 -> {split_dir}")

    print(f"\n完成。抽样清单已固定（seed={args.seed}），如需复现实验请勿更改该参数。")
    print(f"训练集: {len(train_files)} | 验证集: {len(val_files)} | 测试集: {len(test_files)}")
    unused = total - needed
    if unused > 0:
        print(f"(筛选后还剩 {unused} 个符合条件但未使用的文件，如需扩大训练集可以调整参数重新划分)")


if __name__ == "__main__":
    main()