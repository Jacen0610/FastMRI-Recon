"""
数据协议 v2 切分: 三批 train 原始数据 -> 统一筛选/去重 -> 重切 train/val。

背景: 每个h5文件即一位独立患者(patient_id属性), 因此文件级切分天然患者级隔离。
full_test 到位后, 用 --exclude_patient_dirs 剔除与其患者重叠的训练池文件。

用法:
    python scripts/split_v2.py \
        --raw_dirs ../mri_data/raw/brain_multicoil_train \
                   ../mri_data/raw/brain_multicoil_train_1 \
                   ../mri_data/raw/brain_multicoil_train_2 \
        --out ../mri_data/subset_v2 --val_ratio 0.05 --seed 42
"""

import argparse
import glob
import os
import random

import h5py


def scan(paths, acquisition, gt_shape, min_coils):
    """递归扫描, 返回 [(path, patient_id, slices)]"""
    out = []
    for f in sorted(paths):
        try:
            with h5py.File(f, "r") as h:
                if acquisition != "all" and h.attrs.get("acquisition", "") != acquisition:
                    continue
                if tuple(h["reconstruction_rss"].shape[1:]) != gt_shape:
                    continue
                if h["kspace"].shape[1] < min_coils:
                    continue
                out.append((f, str(h.attrs.get("patient_id", f)), h["kspace"].shape[0]))
        except OSError:
            print(f"  跳过损坏: {f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--acquisition", type=str, default="AXT2")
    ap.add_argument("--gt_h", type=int, default=384)
    ap.add_argument("--gt_w", type=int, default=384)
    ap.add_argument("--min_coils", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--min_val_files", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_patient_dirs", type=str, nargs="*", default=[],
                    help="这些目录中出现过的患者将从训练池剔除(如 full_test)")
    args = ap.parse_args()

    # 需要剔除的患者
    excl = set()
    for d in args.exclude_patient_dirs:
        for f in glob.glob(os.path.join(d, "**", "*.h5"), recursive=True):
            try:
                with h5py.File(f, "r") as h:
                    excl.add(str(h.attrs.get("patient_id", "")))
            except OSError:
                pass
    if excl:
        print(f"剔除患者(来自 {len(args.exclude_patient_dirs)} 个目录): {len(excl)} 人")

    all_paths = []
    for d in args.raw_dirs:
        found = glob.glob(os.path.join(d, "**", "*.h5"), recursive=True)
        print(f"{d}: {len(found)} 个h5")
        all_paths += found

    print(f"共 {len(all_paths)} 文件, 开始筛选({args.acquisition}, GT {args.gt_h}x{args.gt_w}, 线圈>={args.min_coils})...")
    eligible = scan(all_paths, args.acquisition, (args.gt_h, args.gt_w), args.min_coils)
    print(f"  筛选通过: {len(eligible)} 个文件 / {sum(e[2] for e in eligible)} 切片")

    # 患者去重(保留首个) + 排除
    seen, pool = set(), []
    dropped_dup = dropped_excl = 0
    for f, p, n in eligible:
        if p in excl:
            dropped_excl += 1
            continue
        if p in seen:
            dropped_dup += 1
            continue
        seen.add(p)
        pool.append((f, n))
    print(f"  患者去重剔除 {dropped_dup} | 与排除目录重叠剔除 {dropped_excl} | 池: {len(pool)} 文件")

    random.Random(args.seed).shuffle(pool)
    n_val = max(args.min_val_files, int(len(pool) * args.val_ratio))
    val, train = pool[:n_val], pool[n_val:]

    for split, items in [("train", train), ("val", val)]:
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        for f, _ in items:
            dst = os.path.join(d, os.path.basename(f))
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(f), dst)
        print(f"{split}: {len(items)} 文件 / {sum(n for _, n in items)} 切片 -> {d}")


if __name__ == "__main__":
    main()
