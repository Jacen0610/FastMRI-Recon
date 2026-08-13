"""
迭代式 SME+U-Net (Brain Multicoil) 训练脚本。

使用方法：
    python src/train_iterative_sme_unet.py \
        --train_dir data/subset/train \
        --val_dir data/subset/val \
        --acceleration 8 \
        --batch_size 1 \
        --grad_accum_steps 4 \
        --epochs 30 \
        --checkpoint_dir checkpoints/iter_sme_unet_8x

显存提醒：本模型包含两个完整重建U-Net，显存开销比单次SME+U-Net(约6.35GB)更高，
建议从batch_size=1开始测试。
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM

from data.dataset_multicoil import BrainMulticoilDataset
from data.transforms import center_crop
from models.iterative_sme_unet import IterativeSMEUNetModel


class L1SSIMLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=10.0, size_average=True, channel=1)

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        pred_shifted = pred - pred.min() + 1e-3
        target_shifted = target - target.min() + 1e-3
        ssim_loss = 1 - self.ssim(pred_shifted, target_shifted)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss


def parse_args():
    parser = argparse.ArgumentParser(description="迭代式SME+U-Net Brain Multicoil训练")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--acceleration", type=int, default=4, choices=[4, 8])
    parser.add_argument("--target_num_coils", type=int, default=8,
                         help="目标虚拟线圈数(压缩后)，配合--use_coil_compression使用")
    parser.add_argument("--use_coil_compression", action="store_true", default=True,
                         help="线圈数>=target_num_coils的文件均纳入使用(压缩到目标值)；"
                              "关闭则退回严格模式")
    parser.add_argument("--no_coil_compression", dest="use_coil_compression",
                         action="store_false")
    parser.add_argument("--acquisition_filter", type=str, default="AXT2")
    parser.add_argument("--target_gt_h", type=int, default=384)
    parser.add_argument("--target_gt_w", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=1,
                         help="模型含两个完整U-Net，显存开销高于单次SME+U-Net，建议从1开始")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sme_chans", type=int, default=8)
    parser.add_argument("--sme_pool_layers", type=int, default=2)
    parser.add_argument("--unet_chans", type=int, default=32)
    parser.add_argument("--unet_pool_layers", type=int, default=4)
    parser.add_argument("--ssim_alpha", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/iter_sme_unet")
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            kspace = batch["kspace"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            num_low_freqs = batch["num_low_freqs"]

            output = model(kspace, mask, num_low_freqs)
            output_cropped = center_crop(output, target.shape[-2:])
            loss = criterion(output_cropped, target)

            total_loss += loss.item() * kspace.size(0)
            num_samples += kspace.size(0)

    model.train()
    return total_loss / num_samples


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset = BrainMulticoilDataset(
        args.train_dir, acceleration=args.acceleration,
        target_num_coils=args.target_num_coils,
        use_coil_compression=args.use_coil_compression,
        acquisition_filter=args.acquisition_filter,
        target_gt_shape=(args.target_gt_h, args.target_gt_w),
        is_train=True,
    )
    val_dataset = BrainMulticoilDataset(
        args.val_dir, acceleration=args.acceleration,
        target_num_coils=args.target_num_coils,
        use_coil_compression=args.use_coil_compression,
        acquisition_filter=args.acquisition_filter,
        target_gt_shape=(args.target_gt_h, args.target_gt_w),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = IterativeSMEUNetModel(
        sme_chans=args.sme_chans, sme_pool_layers=args.sme_pool_layers,
        unet_chans=args.unet_chans, unet_pool_layers=args.unet_pool_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")

    criterion = L1SSIMLoss(alpha=args.ssim_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    print(f"\n训练配置: acceleration={args.acceleration}x, "
          f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps} "
          f"(等效batch_size={args.batch_size * args.grad_accum_steps}), "
          f"amp={args.amp and device.type == 'cuda'}\n")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            kspace = batch["kspace"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            num_low_freqs = batch["num_low_freqs"]

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                output = model(kspace, mask, num_low_freqs)
                output_cropped = center_crop(output, target.shape[-2:])
                loss = criterion(output_cropped, target) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum_steps

            if (i + 1) % args.log_interval == 0:
                avg_loss = running_loss / (i + 1)
                print(f"  Epoch {epoch} [{i+1}/{len(train_loader)}] "
                      f"train_loss={avg_loss:.6f}")

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} 完成 | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"lr={current_lr:.2e} | 耗时={epoch_time:.1f}s")

        scheduler.step(val_loss)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }, os.path.join(args.checkpoint_dir, "last.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> 新的最优模型已保存 (val_loss={val_loss:.6f})")

    print(f"\n训练完成。最优 val_loss={best_val_loss:.6f}，"
          f"checkpoint保存于 {args.checkpoint_dir}/best.pth")


if __name__ == "__main__":
    main()