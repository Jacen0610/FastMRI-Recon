"""
图像裁剪工具函数。

fastMRI官方Ground Truth经过center crop（如640x368 -> 320x320），
模型的中间输出（IFFT重建结果等）在与GT对比或计算指标前，需要做同样的裁剪对齐。
"""

import numpy as np


def ifft2c(kspace):
    """
    中心化的二维逆傅里叶变换，采用正交归一化(norm='ortho')约定。

    背景：诊断发现raw IFFT重建和fastMRI官方reconstruction_esc之间存在一个
    固定的sqrt(H*W)倍尺度差异(在knee singlecoil数据上实测约485.3，
    恰好等于sqrt(640*368))。这是FFT归一化约定不一致导致的：
    numpy默认约定(逆变换除以1/N) vs 官方使用的正交归一化约定(每次变换除以1/sqrt(N))。
    这不是数据内容导致的差异，是纯数学约定问题，因此用norm='ortho'
    从根源统一，让我们计算的重建结果和官方GT处于同一物理尺度，
    不再需要在训练/评估时做额外的尺度对齐或独立归一化。
    """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho"))


def fft2c(image):
    """中心化的二维傅里叶变换，与ifft2c对应的正变换，同样采用正交归一化"""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image), norm="ortho"))


def center_crop(data, shape):
    """
    对图像做中心裁剪，只作用于最后两个维度(H, W)，兼容2D和更高维输入。

    参数:
        data: np.ndarray, 最后两维为(H, W)，H/W需 >= 目标shape对应维度
        shape: tuple, (target_h, target_w)，目标裁剪尺寸

    返回:
        cropped: np.ndarray, 最后两维变为 shape
    """
    target_h, target_w = shape
    h, w = data.shape[-2], data.shape[-1]

    if h < target_h or w < target_w:
        raise ValueError(
            f"输入尺寸 ({h}, {w}) 小于目标裁剪尺寸 {shape}，无法裁剪"
        )

    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2

    return data[..., start_h : start_h + target_h, start_w : start_w + target_w]


def normalize_instance(data, eps=1e-11):
    """
    用输入自身的均值/标准差做归一化(instance normalization)，
    这是fastMRI官方U-Net baseline采用的标准做法：
    每个样本各自归一化到均值0、方差1，避免不同样本间数值尺度差异过大导致训练不稳定。

    参数:
        data: np.ndarray 或 torch.Tensor
        eps: 防止除零的小常数

    返回:
        normalized: 归一化后的数据
        mean: 该样本的均值(用于反归一化，还原真实尺度)
        std: 该样本的标准差(用于反归一化)
    """
    mean = data.mean()
    std = data.std()
    normalized = (data - mean) / (std + eps)
    return normalized, mean, std


def unnormalize(data, mean, std, eps=1e-11):
    """normalize_instance的逆操作，把归一化后的数据还原回原始数值尺度"""
    return data * (std + eps) + mean


def center_pad(data, target_shape):
    """
    对数据做中心补零，把最后两维(H, W)填充到目标尺寸。
    与center_crop相对应：fastMRI不同volume的k-space原始尺寸并不完全一致
    (实测同一数据集内出现过640x368和640x372两种)，VarNet需要保留完整k-space
    分辨率(不能像U-Net那样直接裁剪到固定尺寸)，因此用补零的方式统一尺寸，
    保证同一个batch内所有样本形状一致，同时不丢失任何真实测量到的数据。

    参数:
        data: np.ndarray，最后两维为(H, W)
        target_shape: tuple, (target_h, target_w)，目标尺寸，需 >= 原尺寸

    返回:
        padded: 最后两维变为 target_shape，原数据在中心，四周补0
    """
    h, w = data.shape[-2], data.shape[-1]
    target_h, target_w = target_shape

    if h > target_h or w > target_w:
        raise ValueError(
            f"输入尺寸 ({h}, {w}) 超过目标padding尺寸 {target_shape}，无法补零"
        )

    pad_h = target_h - h
    pad_w = target_w - w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    pad_width = [(0, 0)] * (data.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
    return np.pad(data, pad_width, mode="constant", constant_values=0)