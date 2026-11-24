"""
DDPM 训练脚本(基于 make_ddpm_cache.py 制备的缓存)。

缓存是「推理一致」归一化后的GT幅度图 (N, 384, 384), 全量载入内存训练。
EMA 权重随 checkpoint 保存(采样用 EMA 版本)。

正式训练(约3~4h, 4060Ti):
    python src/train_ddpm.py --cache ../mri_data/ddpm_cache_8x.npy \
        --steps 20000 --chans 64 --checkpoint_dir checkpoints/ddpm_brain
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ddpm import DDPM


class CacheDataset(Dataset):
    def __init__(self, cache_path):
        self.data = torch.from_numpy(np.load(cache_path)).unsqueeze(1)  # (N,1,384,384)
        print(f"缓存: {self.data.shape} (已载入内存)")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--chans", type=int, default=64)
    ap.add_argument("--num_pools", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints/ddpm_brain")
    ap.add_argument("--log_interval", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ds = CacheDataset(args.cache)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                          num_workers=2, pin_memory=(device.type == "cuda"),
                                          drop_last=True, persistent_workers=True)
    model = DDPM(chans=args.chans, num_pools=args.num_pools).to(device)
    print(f"设备: {device} | DDPM 参数量: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.train()
    step, t0, running = 0, time.time(), 0.0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            x = batch.to(device, non_blocking=True)
            loss = model.loss(x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            running += loss.item()
            step += 1
            if step % args.log_interval == 0:
                print(f"step {step}/{args.steps} | loss={running/args.log_interval:.4f} "
                      f"| {step/(time.time()-t0):.2f} it/s", flush=True)
                running = 0.0

    torch.save({"model_state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow.state_dict(),
                "step": step, "args": vars(args)},
               os.path.join(args.checkpoint_dir, "last.pth"))
    print(f"已保存: {args.checkpoint_dir}/last.pth")


if __name__ == "__main__":
    main()
