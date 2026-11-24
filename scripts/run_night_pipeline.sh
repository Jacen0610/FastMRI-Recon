#!/bin/bash
# 夜间自驱动流水线: batch_1解压结束 -> 切分 -> DDPM缓存 -> SME重训+评估 -> DDPM训练+评估 -> VarNet重训+评估
set -u
cd /home/jacen/workspace/brain_reconstruction/multicoil-varnet-brain
PY=/home/jacen/miniconda3/envs/monai/bin/python
LOG=/tmp/night_pipeline.log
STATUS=../RUN_STATUS.md

note() { echo "[$(date '+%H:%M')] $1" | tee -a "$LOG"; echo "- ${1}" >> "$STATUS"; }

note "等待 batch_1 解压进程结束..."
while pgrep -f "brain_multicoil_train_batch_1" > /dev/null 2>&1; do sleep 120; done
note "batch_1 解压完成: $(find ../mri_data/raw/brain_multicoil_train_1 -name '*.h5' 2>/dev/null | wc -l) 个h5"

note "S1: split_v2 三源切分..."
$PY scripts/split_v2.py \
  --raw_dirs ../mri_data/raw/brain_multicoil_train ../mri_data/raw/brain_multicoil_train_1 ../mri_data/raw/brain_multicoil_train_2 \
  --exclude_patient_dirs ../mri_data/subset_compressed/test ../mri_data/subset_compressed/val \
  --out ../mri_data/subset_v2 --val_ratio 0.06 >> "$LOG" 2>&1 || note "S1 失败!"

note "S2: DDPM 缓存制备..."
$PY src/make_ddpm_cache.py --train_dir ../mri_data/subset_v2/train \
  --sme_checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth \
  --acceleration 8 --out ../mri_data/ddpm_cache_v2.npy >> "$LOG" 2>&1 || note "S2 失败!"

note "A1: 重训 SME+U-Net (等总训练量: 14 epochs)..."
$PY src/train_multicoil_sme.py --train_dir ../mri_data/subset_v2/train --val_dir ../mri_data/subset_v2/val \
  --acceleration 8 --epochs 14 --batch_size 4 --loss_type l1_ssim_lpips_roi \
  --checkpoint_dir checkpoints/sme_unet_8x_datav2 >> "$LOG" 2>&1 || note "A1 失败!"

note "E-A1: 旧冻结test上评估重训SME..."
$PY src/evaluate_multicoil.py --checkpoint checkpoints/sme_unet_8x_datav2/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 \
  --save_dir results/results_sme_8x_datav2 >> "$LOG" 2>&1 || note "E-A1 失败!"

note "B1: DDPM 训练 18k steps..."
$PY src/train_ddpm.py --cache ../mri_data/ddpm_cache_v2.npy --steps 18000 --chans 64 \
  --checkpoint_dir checkpoints/ddpm_brain_v2 >> "$LOG" 2>&1 || note "B1 失败!"

note "E-B1: Diffusion 引导采样评估(60切片)..."
$PY src/evaluate_diffusion.py --ddpm_checkpoint checkpoints/ddpm_brain_v2/last.pth \
  --sme_checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 --steps 50 \
  --max_slices 60 --save_dir results/results_diffusion_8x_v2 >> "$LOG" 2>&1 || note "E-B1 失败!"

note "A2: 重训 VarNet (14 epochs, 最重)..."
$PY src/train_e2e_varnet.py --train_dir ../mri_data/subset_v2/train --val_dir ../mri_data/subset_v2/val \
  --acceleration 8 --epochs 14 --batch_size 2 --grad_accum_steps 4 --loss_type l1_ssim_lpips_roi \
  --checkpoint_dir checkpoints/e2e_varnet_8x_datav2 >> "$LOG" 2>&1 || note "A2 失败!"

note "E-A2: 旧冻结test上三方评估(重训SME vs 重训VarNet)..."
$PY src/evaluate_e2e_varnet.py --varnet_checkpoint checkpoints/e2e_varnet_8x_datav2/best.pth \
  --sme_unet_checkpoint checkpoints/sme_unet_8x_datav2/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 \
  --save_dir results/results_e2e_8x_datav2 --n_examples 0 >> "$LOG" 2>&1 || note "E-A2 失败!"

note "全部流水线步骤结束"
