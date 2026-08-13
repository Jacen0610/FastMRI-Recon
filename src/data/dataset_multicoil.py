"""
Brain Multicoil 数据集的 PyTorch Dataset 封装，用于 SME + U-Net / E2E-VarNet 训练。

关键设计：
- 线圈数量：不再要求"必须等于目标值"，而是用线圈压缩(SVD)把任意数量(>=目标值)的
  物理线圈，统一投影压缩到固定的虚拟线圈数(target_num_coils)。这能把原本因
  "线圈数不是20"而被排除的16/14/12线圈等文件重新纳入训练，显著扩大可用数据规模。
  压缩矩阵基于每个文件自己的ACS区域计算(全采样、信噪比高)，是确定性的数学变换，
  不涉及训练，风险与center_pad这类工具函数相当。压缩在欠采样mask应用之前进行，
  数学上二者可交换顺序(压缩只在线圈维度做线性组合，与频率维度的mask相互独立)。
- 只筛选Ground Truth尺寸一致的文件：实测brain数据集内不同文件的reconstruction_rss
  尺寸并不统一(如320x320和384x384并存)，这很可能反映不同扫描的实际视野(FOV)差异，
  而不是单纯的"缺失数据"，直接补零可能错误地把"视野外真实存在的解剖结构"当作背景，
  因此依然采用筛选策略，而不是像k-space尺寸差异那样用center_pad处理
- 复用singlecoil项目验证过的mask.py(欠采样mask对每个线圈广播使用，因为欠采样
  模式由脉冲序列/梯度时序决定，不是线圈相关的，所有线圈共享同一个mask)
- 复用transforms.py的center_pad处理"k-space尺寸(H,W)"不一致的问题(这个和线圈数量/
  GT尺寸不一致是不同性质的问题，k-space的频率维度差异可以安全padding)
"""

import glob
import os
from collections import Counter

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .mask import get_mask_for_acceleration, apply_mask, ACCELERATION_CONFIGS
    from .transforms import ifft2c, center_pad, center_crop
    from .coil_compression import compress_coils
except ImportError:
    from mask import get_mask_for_acceleration, apply_mask, ACCELERATION_CONFIGS
    from transforms import ifft2c, center_pad, center_crop
    from coil_compression import compress_coils


class BrainMulticoilDataset(Dataset):
    """
    参数:
        data_dir: 存放.h5文件的目录
        acceleration: 欠采样加速倍数，4 或 8
        target_num_coils: 目标虚拟线圈数(压缩后的线圈数)。原始线圈数>=此值的文件
                           都会被压缩到这个数量后使用；原始线圈数<此值的文件会被跳过
                           (压缩只能减少线圈数，无法凭空增加)
        use_coil_compression: True时对线圈数大于target_num_coils的文件做压缩；
                               False则退回严格模式，只使用线圈数恰好等于target_num_coils
                               的文件(即之前版本的行为，用于对比实验)
        acquisition_filter: 只使用指定序列类型(如'AXT2')的文件，None则不筛选
        target_gt_shape: 只使用GT尺寸等于此值(如(384,384))的文件；
                          None则自动检测数据集内最常见的GT尺寸并据此筛选
        is_train: True时每个样本用随机mask；False时用固定seed
        fixed_seed: is_train=False时使用的固定随机种子
    """

    def __init__(self, data_dir, acceleration=4, target_num_coils=8,
                 use_coil_compression=True,
                 acquisition_filter="AXT2", target_gt_shape=None,
                 is_train=True, fixed_seed=42):
        self.acceleration = acceleration
        self.target_num_coils = target_num_coils
        self.use_coil_compression = use_coil_compression
        self.is_train = is_train
        self.fixed_seed = fixed_seed

        all_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if len(all_files) == 0:
            raise FileNotFoundError(f"在 {data_dir} 下未找到任何.h5文件")

        # 第一遍扫描：筛选 序列类型 + 线圈数量条件，同时统计GT尺寸分布和原始线圈数
        candidates = []
        gt_shape_dist = Counter()
        native_coil_dist = Counter()
        for fpath in all_files:
            with h5py.File(fpath, "r") as f:
                acquisition = f.attrs.get("acquisition", "unknown")
                if acquisition_filter is not None and acquisition != acquisition_filter:
                    continue
                num_slices, num_coils, h, w = f["kspace"].shape

                if use_coil_compression:
                    if num_coils < target_num_coils:
                        continue  # 线圈数不足，无法压缩到目标值
                else:
                    if num_coils != target_num_coils:
                        continue  # 严格模式：必须恰好相等

                gt_shape = f["reconstruction_rss"].shape[1:]
                gt_shape_dist[gt_shape] += 1
                native_coil_dist[num_coils] += 1
                candidates.append((fpath, h, w, gt_shape, num_coils))

        if len(candidates) == 0:
            raise ValueError(
                f"筛选后没有符合条件的文件 (acquisition={acquisition_filter}, "
                f"target_num_coils={target_num_coils}, "
                f"use_coil_compression={use_coil_compression})，请检查筛选条件"
            )

        if target_gt_shape is None:
            target_gt_shape = gt_shape_dist.most_common(1)[0][0]
            print(f"[BrainMulticoilDataset] 自动检测到最常见GT尺寸: {target_gt_shape} "
                  f"(完整分布: {dict(gt_shape_dist)})")
        self.target_gt_shape = tuple(target_gt_shape)

        # 第二遍筛选：只保留GT尺寸也匹配的文件
        self.files = []
        max_h, max_w = 0, 0
        used_coil_dist = Counter()
        for fpath, h, w, gt_shape, num_coils in candidates:
            if tuple(gt_shape) != self.target_gt_shape:
                continue
            self.files.append(fpath)
            used_coil_dist[num_coils] += 1
            max_h, max_w = max(max_h, h), max(max_w, w)

        if len(self.files) == 0:
            raise ValueError(
                f"按GT尺寸{self.target_gt_shape}筛选后没有符合条件的文件，"
                f"请检查target_gt_shape参数"
            )

        self.target_shape = (max_h, max_w)
        print(f"[BrainMulticoilDataset] {data_dir}: 筛选出 {len(self.files)} 个文件 "
              f"(acquisition={acquisition_filter}, target_num_coils={target_num_coils}, "
              f"use_coil_compression={use_coil_compression}, "
              f"gt_shape={self.target_gt_shape})")
        print(f"[BrainMulticoilDataset] 实际使用的原始线圈数分布: {dict(sorted(used_coil_dist.items()))}")
        print(f"[BrainMulticoilDataset] k-space统一padding尺寸: {self.target_shape}")

        self.index_map = []
        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                num_slices = f["kspace"].shape[0]
            for slice_idx in range(num_slices):
                self.index_map.append((fpath, slice_idx))

        print(f"[BrainMulticoilDataset] 共 {len(self.index_map)} 张切片 "
              f"(acceleration={acceleration}x, is_train={is_train})")

        center_fraction = ACCELERATION_CONFIGS[acceleration]["center_fraction"]
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fpath, slice_idx = self.index_map[idx]

        with h5py.File(fpath, "r") as f:
            ks_slice = f["kspace"][slice_idx]  # (native_num_coils, H, W) complex
            gt_slice = f["reconstruction_rss"][slice_idx].astype(np.float32)
            acquisition = f.attrs.get("acquisition", "unknown")

        native_num_coils, h, w = ks_slice.shape
        num_low_freqs = int(round(w * self.center_fraction))

        # 线圈压缩：若原始线圈数大于目标值，先压缩到目标虚拟线圈数
        # (压缩矩阵基于该文件自己的ACS区域计算，全采样、信噪比高)
        if native_num_coils > self.target_num_coils:
            ks_slice = compress_coils(ks_slice, num_low_freqs, self.target_num_coils)
        num_coils = ks_slice.shape[0]  # 压缩后统一为target_num_coils

        seed = None if self.is_train else self.fixed_seed
        mask_2d, _, _ = get_mask_for_acceleration((h, w), acceleration=self.acceleration, seed=seed)

        # mask对所有(虚拟)线圈广播使用
        undersampled_ks = ks_slice * mask_2d[np.newaxis, :, :]  # (num_coils, H, W)

        coil_images_zf = np.stack(
            [ifft2c(undersampled_ks[c]) for c in range(num_coils)], axis=0
        )
        zf_rss = np.sqrt(np.sum(np.abs(coil_images_zf) ** 2, axis=0))
        scale = zf_rss.std()
        if scale < 1e-12:
            scale = 1.0

        ks_norm = undersampled_ks / scale
        target_norm = gt_slice / scale

        kspace_2ch = np.stack([ks_norm.real, ks_norm.imag], axis=1).astype(np.float32)
        mask_tensor = mask_2d.astype(np.float32)[np.newaxis, :, :]

        kspace_2ch = center_pad(kspace_2ch, self.target_shape)
        mask_tensor = center_pad(mask_tensor, self.target_shape)

        return {
            "kspace": torch.from_numpy(kspace_2ch),
            "mask": torch.from_numpy(mask_tensor),
            "target": torch.from_numpy(target_norm).unsqueeze(0),
            "num_low_freqs": num_low_freqs,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "fname": os.path.basename(fpath),
            "slice_idx": slice_idx,
            "acquisition": acquisition,
            "native_num_coils": native_num_coils,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python dataset_multicoil.py <data_dir>")
        sys.exit(1)

    dataset = BrainMulticoilDataset(sys.argv[1], acceleration=4, target_num_coils=8,
                                     use_coil_compression=True)
    sample = dataset[0]
    print(f"\n单个样本检查:")
    print(f"  kspace shape: {sample['kspace'].shape}")
    print(f"  mask shape: {sample['mask'].shape}")
    print(f"  target shape: {sample['target'].shape}")
    print(f"  num_low_freqs: {sample['num_low_freqs']}")
    print(f"  scale: {sample['scale'].item():.6e}")
    print(f"  acquisition: {sample['acquisition']}")
    print(f"  native_num_coils(压缩前原始线圈数): {sample['native_num_coils']}")