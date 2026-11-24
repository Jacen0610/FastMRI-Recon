#!/bin/bash
# 全量流水线(3.2x数据): 等batch_1标记文件 -> 切分 -> 缓存 -> SME重训+评估 -> DDPM+评估 -> VarNet+评估
set -u
cd /home/jacen/workspace/brain_reconstruction/multicoil-varnet-brain
PY=/home/jacen/miniconda3/envs/monai/bin/python
LOG=/tmp/pipeline_full.log
STATUS=../RUN_STATUS.md
note() { echo "[$(date '+%H:%M')] $1" | tee -a "$LOG"; echo "- ${1}" >> "$STATUS"; }

note "等待 batch_1 完成标记文件..."
while [ ! -f /tmp/batch1_done ]; do sleep 60; done
note "batch_1 就绪: $(find ../mri_data/raw/brain_multicoil_train_1 -name '*.h5' | wc -l) 个h5"

note "S1: split_v2 (batch_0+1+2 全量)..."
$PY scripts/split_v2.py \
  --raw_dirs ../mri_data/raw/brain_multicoil_train ../mri_data/raw/brain_multicoil_train_1 ../mri_data/raw/brain_multicoil_train_2 \
  --exclude_patient_dirs ../mri_data/subset_compressed/test ../mri_data/subset_compressed/val \
  --out ../mri_data/subset_v2 --val_ratio 0.06 >> "$LOG" 2>&1 || note "S1 失败!"
N_TRAIN=$(ls ../mri_data/subset_v2/train/*.h5 2>/dev/null | wc -l)
EPOCHS=$(( (6500 + N_TRAIN - 1) / N_TRAIN ))
note "S1完成: train=${N_TRAIN}文件 -> 等总训练量协议 epochs=${EPOCHS}"

note "S2: DDPM 缓存制备(全量)..."
$PY src/make_ddpm_cache.py --train_dir ../mri_data/subset_v2/train \
  --sme_checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth \
  --acceleration 8 --out ../mri_data/ddpm_cache_full.npy >> "$LOG" 2>&1 || note "S2 失败!"

note "A1: 重训 SME+U-Net (全量, ${EPOCHS} epochs)..."
$PY src/train_multicoil_sme.py --train_dir ../mri_data/subset_v2/train --val_dir ../mri_data/subset_v2/val \
  --acceleration 8 --epochs ${EPOCHS} --batch_size 4 --loss_type l1_ssim_lpips_roi \
  --checkpoint_dir checkpoints/sme_unet_8x_datafull >> "$LOG" 2>&1 || note "A1 失败!"

note "E-A1: 评估全量SME (旧冻结test)..."
$PY src/evaluate_multicoil.py --checkpoint checkpoints/sme_unet_8x_datafull/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 \
  --save_dir results/results_sme_8x_datafull >> "$LOG" 2>&1 || note "E-A1 失败!"

note "B1: DDPM 训练 18k steps (全量缓存)..."
$PY src/train_ddpm.py --cache ../mri_data/ddpm_cache_full.npy --steps 18000 --chans 64 \
  --checkpoint_dir checkpoints/ddpm_brain_full >> "$LOG" 2>&1 || note "B1 失败!"

note "E-B1: Diffusion 引导评估(60切片)..."
$PY src/evaluate_diffusion.py --ddpm_checkpoint checkpoints/ddpm_brain_full/last.pth \
  --sme_checkpoint checkpoints/sme_unet_8x_lpips_roi_v3/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 --steps 50 \
  --max_slices 60 --save_dir results/results_diffusion_8x_full >> "$LOG" 2>&1 || note "E-B1 失败!"

note "A2: 重训 VarNet (全量, ${EPOCHS} epochs)..."
$PY src/train_e2e_varnet.py --train_dir ../mri_data/subset_v2/train --val_dir ../mri_data/subset_v2/val \
  --acceleration 8 --epochs ${EPOCHS} --batch_size 2 --grad_accum_steps 4 --loss_type l1_ssim_lpips_roi \
  --checkpoint_dir checkpoints/e2e_varnet_8x_datafull >> "$LOG" 2>&1 || note "A2 失败!"

note "E-A2: 三方评估(全量SME vs 全量VarNet)..."
$PY src/evaluate_e2e_varnet.py --varnet_checkpoint checkpoints/e2e_varnet_8x_datafull/best.pth \
  --sme_unet_checkpoint checkpoints/sme_unet_8x_datafull/best.pth \
  --test_dir ../mri_data/subset_compressed/test --acceleration 8 \
  --save_dir results/results_e2e_8x_datafull --n_examples 0 >> "$LOG" 2>&1 || note "E-A2 失败!"

note "全量流水线结束"
