"""
制备 DDPM 训练缓存: GT幅度 / (逐切片scale × 全局常数C), 打包单个npy + 常数json。

v3 设计(吸取两版教训):
- v1: GT自身统计归一化 -> 推理端无GT, 域失配;
- v2: 逐切片用零填充图mu/sd -> 分布重尾(max 95), 且实现曾有缩进bug(每卷只剩1切片);
- v3: x = gt / (scale * C)。scale=逐切片zf-RSS std(测量端可算); C=训练集一次性
  标定的全局常数(各切片gt/scale的p99的中位数)。推理端只需常数C, 训练/推理
  必然同口径, 分布近似有界(p99≈1)。

用法:
    python src/make_ddpm_cache.py --train_dir ../mri_data/subset_v3/train \
        --acceleration 8 --out ../mri_data/ddpm_cache_v3.npy
"""
import argparse, glob, json, os, sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.coil_compression import compress_coils
from data.mask import get_mask_for_acceleration
from data.transforms import ifft2c, center_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--acceleration", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_num_coils", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.train_dir, "*.h5")))
    print(f"{len(files)} 个文件, acceleration={args.acceleration}x")
    nlf = {4: 32, 8: 16}[args.acceleration]

    imgs, p99s = [], []
    for fi, f in enumerate(files):
        with h5py.File(f, "r") as h:
            n_sl = h["kspace"].shape[0]
            C_native = h["kspace"].shape[1]
            H, W = h["kspace"].shape[2:]
            if C_native < args.target_num_coils:
                continue
            m2, _, _ = get_mask_for_acceleration((H, W), args.acceleration, seed=args.seed)
            for si in range(n_sl):                      # 逐切片读取(内存友好)
                ks = h["kspace"][si]
                gt = h["reconstruction_rss"][si].astype(np.float32)
                ks8 = compress_coils(ks.astype(np.complex64), nlf, args.target_num_coils) \
                    if C_native > args.target_num_coils else ks.astype(np.complex64)
                us = ks8 * m2
                zf_rss = np.sqrt(np.sum(np.abs(np.stack([ifft2c(c) for c in us])) ** 2, 0))
                scale = zf_rss.std() + 1e-12
                raw = center_crop(gt, (384, 384)) / scale
                imgs.append(raw.astype(np.float32))
                p99s.append(float(np.percentile(raw, 99)))
        if (fi + 1) % 40 == 0:
            print(f"  {fi+1}/{len(files)} 卷, 已缓存 {len(imgs)} 切片", flush=True)

    C = float(np.median(p99s))
    arr = np.stack(imgs) / C
    np.save(args.out, arr)
    json.dump({"C": C, "scheme": "gt/(scale*C), C=median(p99)", "slices": int(arr.shape[0])},
              open(args.out.replace(".npy", "_norm.json"), "w"))
    print(f"完成: {arr.shape} C={C:.2f} | 归一后 mean={arr.mean():.3f} std={arr.std():.3f} "
          f"p99={np.percentile(arr,99):.2f} max={arr.max():.2f} -> {args.out}")


if __name__ == "__main__":
    main()
