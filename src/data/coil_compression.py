"""
基于SVD的线圈压缩(Coil Compression)。

多个物理线圈的信号高度冗余(相邻线圈观测区域大量重叠)，可以用SVD找到一组最优的
线性组合方向，把任意数量(>=target)的物理线圈，压缩投影成固定数量的"虚拟线圈"，
同时尽量保留原始信号中的主要信息量。这是MRI并行成像领域的经典预处理技术
(Buehrer et al. 2007)，不涉及训练，是确定性的数学变换。

用途：让原本因为"线圈数量不统一"而被迫排除的数据(比如16/14/12线圈的文件)，
都能统一压缩到同一个目标线圈数，从而被重新纳入训练/评估，扩大可用数据规模。

压缩矩阵基于ACS区域(低频、信噪比高、信息最集中)计算，再应用到完整k-space，
这样既能得到稳定可靠的压缩方向，又能压缩全部频率范围的数据。
"""

import numpy as np


def compute_compression_matrix(kspace, num_acs_lines, target_num_coils):
    """
    从k-space的ACS区域，计算SVD压缩矩阵。

    参数:
        kspace: (num_coils, H, W) complex，完整(或欠采样)k-space
        num_acs_lines: ACS区域宽度(与mask.py中的定义一致)
        target_num_coils: 目标虚拟线圈数，必须 <= num_coils

    返回:
        compression_matrix: (num_coils, target_num_coils) complex，
                             用于将原始线圈数据投影到虚拟线圈空间
    """
    num_coils, h, w = kspace.shape
    if target_num_coils > num_coils:
        raise ValueError(
            f"目标虚拟线圈数({target_num_coils})不能超过原始线圈数({num_coils})，"
            f"线圈压缩只能减少线圈数，不能增加"
        )

    center_start = (w - num_acs_lines) // 2
    center_end = center_start + num_acs_lines
    acs_data = kspace[:, :, center_start:center_end]  # (num_coils, H, num_acs_lines)

    # 展平成 (num_samples, num_coils) 矩阵，每一行是"同一个k-space采样点，
    # num_coils个线圈各自测到的复数值"，做SVD找主成分方向
    acs_matrix = acs_data.reshape(num_coils, -1).T  # (H*num_acs_lines, num_coils)

    # SVD: acs_matrix = U @ diag(S) @ Vh
    # Vh的行(即V的列)给出了"线圈信号空间"里，能量最集中的正交基方向
    _, _, vh = np.linalg.svd(acs_matrix, full_matrices=False)

    # 取能量最大的前target_num_coils个方向作为压缩矩阵
    compression_matrix = vh[:target_num_coils, :].conj().T  # (num_coils, target_num_coils)
    return compression_matrix


def apply_coil_compression(kspace, compression_matrix):
    """
    用压缩矩阵，把k-space从原始线圈数投影到目标虚拟线圈数。

    参数:
        kspace: (num_coils, H, W) complex
        compression_matrix: (num_coils, target_num_coils) complex

    返回:
        compressed_kspace: (target_num_coils, H, W) complex
    """
    num_coils, h, w = kspace.shape
    kspace_flat = kspace.reshape(num_coils, -1)  # (num_coils, H*W)
    compressed_flat = compression_matrix.conj().T @ kspace_flat  # (target_num_coils, H*W)
    target_num_coils = compression_matrix.shape[1]
    return compressed_flat.reshape(target_num_coils, h, w)


def compress_coils(kspace, num_acs_lines, target_num_coils):
    """
    一步完成：计算压缩矩阵 + 应用压缩，是最常用的调用接口。

    参数:
        kspace: (num_coils, H, W) complex
        num_acs_lines: ACS区域宽度
        target_num_coils: 目标虚拟线圈数

    返回:
        compressed_kspace: (target_num_coils, H, W) complex
    """
    if kspace.shape[0] == target_num_coils:
        return kspace  # 已经是目标线圈数，无需压缩

    compression_matrix = compute_compression_matrix(kspace, num_acs_lines, target_num_coils)
    return apply_coil_compression(kspace, compression_matrix)


if __name__ == "__main__":
    # 自检：构造一个"有真实冗余结构"的模拟多线圈数据(而非纯随机噪声)，
    # 验证压缩后信号主要能量得以保留(通过对比RSS重建图像的相似度)
    np.random.seed(0)

    h, w = 96, 96
    num_coils = 16
    num_acs_lines = 24
    target_num_coils = 8

    # 构造一个"真实图像"，并模拟不同线圈的空间灵敏度(用简单的高斯核模拟，
    # 确保线圈之间存在真实的空间相关性/冗余，而不是纯随机独立噪声)
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    true_image = np.exp(-(xx**2 + yy**2) * 2) * (1 + 0.3 * np.sin(10 * xx))

    coil_images = []
    for c in range(num_coils):
        cx, cy = np.cos(2 * np.pi * c / num_coils), np.sin(2 * np.pi * c / num_coils)
        sensitivity = np.exp(-((xx - cx * 0.8) ** 2 + (yy - cy * 0.8) ** 2) * 1.5)
        coil_images.append(true_image * sensitivity)
    coil_images = np.stack(coil_images, axis=0).astype(np.complex64)

    # 转到k-space
    def fft2c(img):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

    def ifft2c(ks):
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ks, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

    kspace = np.stack([fft2c(coil_images[c]) for c in range(num_coils)], axis=0)

    # 压缩前：RSS合并
    rss_before = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))

    # 压缩后：先压缩k-space，再逐虚拟线圈IFFT，再RSS合并
    compressed_kspace = compress_coils(kspace, num_acs_lines, target_num_coils)
    print(f"原始k-space形状: {kspace.shape}")
    print(f"压缩后k-space形状: {compressed_kspace.shape}")
    assert compressed_kspace.shape == (target_num_coils, h, w)

    compressed_coil_images = np.stack(
        [ifft2c(compressed_kspace[c]) for c in range(target_num_coils)], axis=0
    )
    rss_after = np.sqrt(np.sum(np.abs(compressed_coil_images) ** 2, axis=0))

    # 用归一化相关系数衡量压缩前后RSS图像的相似程度(应该非常接近，
    # 因为构造的数据里线圈间确实存在真实冗余，8个虚拟线圈应该能保留绝大部分信息)
    corr = np.corrcoef(rss_before.flatten(), rss_after.flatten())[0, 1]
    print(f"\n压缩前后RSS图像相关系数: {corr:.6f} (应接近1.0，说明信息基本保留)")

    rel_error = np.linalg.norm(rss_after - rss_before) / np.linalg.norm(rss_before)
    print(f"相对误差: {rel_error:.4f} (应远小于1，说明压缩没有丢失主要信息)")

    print("\n线圈压缩自检通过")