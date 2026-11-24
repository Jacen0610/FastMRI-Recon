#!/bin/bash
# 阶段1流水线: batch_0+batch_2 数据(2.4x) —— 立即执行, 无等待门
set -u
cd /home/jacen/workspace/brain_reconstruction/multicoil-varnet-brain
PY=/home/jacen/miniconda3/envs/monai/bin/python
LOG=/tmp/pipeline_v2.log
STATUS=../RUN_STATUS.md
note() { echo "[$(date '+%H:%M')] $1" | tee -a "$LOG"; echo "- ${1}" >> "$STATUS"; }

note "S1: split_v2 (batch_0+batch_2, 2.4x数据点)..."
$PY scripts/split_v2.py \
  --raw_dirs ../mri_data/raw/brain_multicoil_train ../mri_data/raw/brain_multicoil_train_2 \
  --exclude_patient_dirs ../mri_data/subset_compressed/test ../mri_data/subset_compressed/val \
  --out ../mri_data/subset_v2 --val_ratio 0.06 >> "$LOG" 2>&1 || note "S1 失败!"

note "S2: DDPM 缓存制备(2.4x)..."
$PY src/make_ddpm_cache.py --train_dir ../mri_data/subset_v2/train \
  --sme_checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth \
  --acceleration 8 --out ../mri_data/ddpm_cache_v2.npy >> "$LOG" 2>&1 || note "S2 失败!"

note "A1: 重训 SME+U-Net (2.4x, 等总训练量: 33 epochs)..."
$PY src/train_multicoil_sme.py --train_dir ../mri_data/subset_v2/train --val_dir ../mri_data/subset_v2/val \
  --acceleration 8 --epochs 33 --batch_size 4 --loss_type l1_ssim_lpips_roi \
  --checkpoint_dir checkpoints/sme_unet_8x_data24 >> "$LOG" 2>&1 || note "A1 失败!"

note "E-A1: 评估 2.4x SME (旧冻结test)..."
$PY src/evaluate_multicoil.py --checkpoint checkpoints/sme_unet_8x_data24/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 \
  --save_dir results/results_sme_8x_data24 >> "$LOG" 2>&1 || note "E-A1 失败!"

note "B1: DDPM 训练 18k steps (2.4x缓存)..."
$PY src/train_ddpm.py --cache ../mri_data/ddpm_cache_v2.npy --steps 18000 --chans 64 \
  --checkpoint_dir checkpoints/ddpm_brain_v2 >> "$LOG" 2>&1 || note "B1 失败!"

note "E-B1: Diffusion 引导评估(60切片)..."
$PY src/evaluate_diffusion.py --ddpm_checkpoint checkpoints/ddpm_brain_v2/last.pth \
  --sme_checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 --steps 50 \
  --max_slices 60 --save_dir results/results_diffusion_8x_v2 >> "$LOG" 2>&1 || note "E-B1 失败!"

note "A2: 重训 VarNet (2.4x, 33 epochs)..."
$PY src/train_e2e_varnet.py --train_dir ../mri_data/subset_v2/train --val_dir ../mri_data/subset_v2/val \
  --acceleration 8 --epochs 33 --batch_size 2 --grad_accum_steps 4 --loss_type l1_ssim_lpips_roi \
  --checkpoint_dir checkpoints/e2e_varnet_8x_data24 >> "$LOG" 2>&1 || note "A2 失败!"

note "E-A2: 三方评估(2.4x SME vs 2.4x VarNet)..."
$PY src/evaluate_e2e_varnet.py --varnet_checkpoint checkpoints/e2e_varnet_8x_data24/best.pth \
  --sme_unet_checkpoint checkpoints/sme_unet_8x_data24/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 \
  --save_dir results/results_e2e_8x_data24 --n_examples 0 >> "$LOG" 2>&1 || note "E-A2 失败!"

note "阶段1(2.4x数据点)全部结束"
