"""
完整 E2E-VarNet (Brain Multicoil) 训练脚本。

使用方法：
    python src/train_e2e_varnet.py \
        --train_dir data/subset/train \
        --val_dir data/subset/val \
        --acceleration 4 \
        --batch_size 1 \
        --num_cascades 4 \
        --epochs 30 \
        --checkpoint_dir checkpoints/e2e_varnet_4x

显存提醒：多级联+20线圈的显存开销远高于此前的SME+U-Net单次版本(该版本batch_size=2
时已占用约6.35GB)，建议从batch_size=1开始测试，用grad_accum_steps模拟更大等效batch。
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
from models.e2e_varnet import E2EVarNet
from losses_with_lpips import L1SSIMLPIPSLoss
from losses_with_lpips_roi import L1SSIMLPIPSROILoss


class L1SSIMLoss(nn.Module):
    """L1 + SSIM 混合损失，与此前脚本中使用的定义保持一致"""

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
    parser = argparse.ArgumentParser(description="完整E2E-VarNet Brain Multicoil训练")
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
                         help="多级联+多线圈显存开销很大，建议从1开始测试")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                         help="梯度累积步数，用于在小batch_size下模拟更大等效batch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_cascades", type=int, default=4)
    parser.add_argument("--sme_chans", type=int, default=8)
    parser.add_argument("--sme_pool_layers", type=int, default=2)
    parser.add_argument("--cascade_chans", type=int, default=12,
                         help="每个级联内部CNN的基础通道数，比singlecoil VarNet(18)更保守，"
                              "因为multicoil的20个线圈本身已带来更大显存压力")
    parser.add_argument("--cascade_pool_layers", type=int, default=2)
    parser.add_argument("--ssim_alpha", type=float, default=0.5,
                         help="loss_type=l1_ssim时生效：L1权重(SSIM权重为1-此值)")
    parser.add_argument("--loss_type", type=str, default="l1_ssim",
                         choices=["l1_ssim", "l1_ssim_lpips", "l1_ssim_lpips_roi"],
                         help="l1_ssim: 原始L1+SSIM混合loss；"
                              "l1_ssim_lpips: 加入LPIPS感知损失；"
                              "l1_ssim_lpips_roi: 再加入ROI加权L1，让模型对细节丰富"
                              "区域(如脑沟脑回)的重建精度更敏感")
    parser.add_argument("--lpips_alpha_l1", type=float, default=0.4)
    parser.add_argument("--lpips_alpha_ssim", type=float, default=0.3)
    parser.add_argument("--lpips_alpha_lpips", type=float, default=0.3)
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg"])
    parser.add_argument("--roi_alpha", type=float, default=0.2)
    parser.add_argument("--roi_top_fraction", type=float, default=0.5)
    parser.add_argument("--roi_kernel_size", type=int, default=7)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/e2e_varnet")
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

    model = E2EVarNet(
        num_cascades=args.num_cascades,
        sme_chans=args.sme_chans, sme_pool_layers=args.sme_pool_layers,
        cascade_chans=args.cascade_chans, cascade_pool_layers=args.cascade_pool_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sme_params = sum(p.numel() for p in model.sme.parameters() if p.requires_grad)
    cascade_params = sum(
        p.numel() for c in model.cascades for p in c.parameters() if p.requires_grad
    )
    print(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M) "
          f"[SME: {sme_params:,}, {args.num_cascades}个级联合计: {cascade_params:,}]")

    if args.loss_type == "l1_ssim_lpips_roi":
        criterion = L1SSIMLPIPSROILoss(
            alpha_l1=args.lpips_alpha_l1, alpha_ssim=args.lpips_alpha_ssim,
            alpha_lpips=args.lpips_alpha_lpips, alpha_roi=args.roi_alpha,
            roi_kernel_size=args.roi_kernel_size, roi_top_fraction=args.roi_top_fraction,
            lpips_net=args.lpips_net,
        ).to(device)
        print(f"使用 L1+SSIM+LPIPS+ROI加权 组合loss")
    elif args.loss_type == "l1_ssim_lpips":
        criterion = L1SSIMLPIPSLoss(
            alpha_l1=args.lpips_alpha_l1, alpha_ssim=args.lpips_alpha_ssim,
            alpha_lpips=args.lpips_alpha_lpips, lpips_net=args.lpips_net,
        ).to(device)
        print(f"使用 L1+SSIM+LPIPS 组合loss")
    else:
        criterion = L1SSIMLoss(alpha=args.ssim_alpha)
        print(f"使用 L1+SSIM loss (alpha={args.ssim_alpha})")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    print(f"\n训练配置: acceleration={args.acceleration}x, "
          f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps} "
          f"(等效batch_size={args.batch_size * args.grad_accum_steps}), "
          f"num_cascades={args.num_cascades}, "
          f"amp={args.amp and device.type == 'cuda'}\n")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        num_valid_batches = 0
        num_skipped_nan = 0

        for i, batch in enumerate(train_loader):
            kspace = batch["kspace"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            num_low_freqs = batch["num_low_freqs"]

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                output = model(kspace, mask, num_low_freqs)
                output_cropped = center_crop(output, target.shape[-2:])
                loss = criterion(output_cropped, target) / args.grad_accum_steps

            loss_value = loss.item()
            if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                scaler.scale(loss).backward()
                running_loss += loss_value * args.grad_accum_steps
                num_valid_batches += 1
            else:
                num_skipped_nan += 1
                print(f"  警告: Epoch {epoch} [{i+1}/{len(train_loader)}] "
                      f"loss为NaN/Inf，已跳过此batch(不参与反向传播和统计)")

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (i + 1) % args.log_interval == 0 and num_valid_batches > 0:
                avg_loss = running_loss / num_valid_batches
                print(f"  Epoch {epoch} [{i+1}/{len(train_loader)}] "
                      f"train_loss={avg_loss:.6f}"
                      + (f" (已跳过{num_skipped_nan}个NaN batch)" if num_skipped_nan > 0 else ""))

        train_loss = running_loss / max(num_valid_batches, 1)
        if num_skipped_nan > 0:
            print(f"  本epoch共跳过 {num_skipped_nan}/{len(train_loader)} 个NaN/Inf batch")
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