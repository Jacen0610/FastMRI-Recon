"""
把新下载的 fastMRI 原始批次整合为增补训练集(方案A/B的数据准备)。

纪律:
- 本地 test/val 冻结不动, 只增补训练集;
- 患者级去重: 新文件的患者ID若出现在本地 test/val 中, 一律排除(防泄漏);
- 与既有筛选一致: 序列类型、GT尺寸、线圈数>=目标(压缩到8);
- 输出 train_aug/: 原train的全部文件 + 新增合格文件的软链接。

自检(用现有raw目录当"新数据"应得: 新增0个, train_aug=原130):
    python scripts/integrate_new_data.py --new_raw ../mri_data/raw/brain_multicoil_train
"""

import argparse
import glob
import os
from pathlib import Path

import h5py


def patient_of(fname):
    # 真实患者ID在h5 attrs(哈希)。文件名第4段不是患者ID(一个编号对应多个患者)。
    with h5py.File(fname, "r") as h:
        return str(h.attrs.get("patient_id", fname))


def scan_files(paths, acquisition, gt_shape, min_coils):
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
                out.append(f)
        except OSError:
            print(f"  跳过损坏文件: {f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_raw", type=str, nargs="+", required=True,
                    help="新批次解压目录(可多个)")
    ap.add_argument("--train", type=str, default="../mri_data/subset_compressed/train")
    ap.add_argument("--frozen", type=str, nargs="+",
                    default=["../mri_data/subset_compressed/test", "../mri_data/subset_compressed/val"],
                    help="冻结不动的划分(患者ID从这里去重)")
    ap.add_argument("--out", type=str, default="../mri_data/subset_compressed/train_aug")
    ap.add_argument("--acquisition", type=str, default="AXT2")
    ap.add_argument("--gt_h", type=int, default=384)
    ap.add_argument("--gt_w", type=int, default=384)
    ap.add_argument("--min_coils", type=int, default=8)
    args = ap.parse_args()

    frozen_patients = set()
    for d in args.frozen:
        for f in glob.glob(os.path.join(d, "*.h5")):
            frozen_patients.add(patient_of(f))
    existing_train = {os.path.basename(f) for f in glob.glob(os.path.join(args.train, "*.h5"))}
    print(f"冻结划分患者ID: {len(frozen_patients)} 个 | 既有训练文件: {len(existing_train)} 个")

    all_new_paths = []
    for d in args.new_raw:
        all_new_paths += glob.glob(os.path.join(d, "*.h5"))
    print(f"新原始文件: {len(all_new_paths)} 个, 开始筛选 ({args.acquisition}, GT {args.gt_h}x{args.gt_w}, 线圈>={args.min_coils})...")
    eligible = scan_files(all_new_paths, args.acquisition, (args.gt_h, args.gt_w), args.min_coils)
    print(f"  筛选通过: {len(eligible)} 个")

    added, skipped_patient, skipped_dup = [], 0, 0
    for f in eligible:
        base = os.path.basename(f)
        if base in existing_train:
            skipped_dup += 1
            continue
        if patient_of(f) in frozen_patients:
            skipped_patient += 1
            continue
        added.append(f)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 原train全部 + 新增
    n_old = 0
    for f in glob.glob(os.path.join(args.train, "*.h5")):
        dst = out / os.path.basename(f)
        if not dst.exists():
            dst.symlink_to(os.path.realpath(f))
        n_old += 1
    n_new = 0
    for f in added:
        dst = out / os.path.basename(f)
        if not dst.exists():
            dst.symlink_to(os.path.realpath(f))
        n_new += 1

    print(f"\n结果: 与既有训练重复跳过 {skipped_dup} | 患者与冻结划分重叠排除 {skipped_patient}")
    print(f"train_aug: 原 {n_old} + 新增 {n_new} = {n_old + n_new} 个文件")
    print("新增的患者ID分布(去重后):", sorted({patient_of(f) for f in added})[:20])


if __name__ == "__main__":
    main()
