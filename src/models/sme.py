"""
SME (Sensitivity Map Estimation) 模块：从欠采样多线圈k-space中，估计每个线圈的
空间灵敏度图。这是对经典并行成像算法(如SENSE)里"预先标定线圈灵敏度"这一步的
深度学习替代——SENSE要求提前用专门的标定扫描测出灵敏度图，SME则直接从当前这次
欠采样数据本身学习估计出一个够用的近似值，不需要额外的标定扫描，且整个过程
和下游重建网络一起端到端训练(没有独立的监督信号，梯度完全来自最终重建效果)。

设计要点：
- 输入用k-space的中心ACS区域(mask.py中始终保留为全采样的低频区域)，
  对每个线圈独立做IFFT得到低分辨率图像，这是估计灵敏度图最常用、最稳定的输入源
  (高频区域欠采样、噪声更大，不适合用来估计灵敏度这种应该平滑变化的量)
- 用同一个小型CNN处理所有线圈(共享权重)，因为"如何从低分辨率图像估计灵敏度"
  这个任务对每个线圈是同质的，不需要为每个线圈单独训练一套参数
- 输出需要做归一化，保证 sum_c |S_c|^2 = 1，这是灵敏度加权合并公式成立的前提
  (否则图像的整体尺度会和灵敏度图的尺度混在一起，产生歧义)
"""

import torch
import torch.nn as nn

from .unet import UNet


def extract_acs_kspace(masked_kspace, num_low_freqs):
    """
    从(已欠采样的)k-space中，提取中心ACS区域，其余位置置0。
    因为mask.py的设计里，中心区域本来就是100%采样的，这里直接从
    已有的masked_kspace中央裁出这部分即可，不需要额外的数据来源。

    参数:
        masked_kspace: (B, num_coils, 2, H, W) 实部/虚部堆叠
        num_low_freqs: 中心ACS区域的宽度(列数)

    返回:
        acs_kspace: 同shape，只保留中心ACS区域，其余为0
    """
    b, c, ch, h, w = masked_kspace.shape
    acs_kspace = torch.zeros_like(masked_kspace)
    center_start = (w - num_low_freqs) // 2
    center_end = center_start + num_low_freqs
    acs_kspace[..., center_start:center_end] = masked_kspace[..., center_start:center_end]
    return acs_kspace


def ifft2c_torch(kspace_2ch):
    """
    对(B, num_coils, 2, H, W)格式的k-space做正交归一化IFFT，返回同格式的图像域结果。
    与transforms.py中numpy版本的ifft2c保持相同的数学约定(norm='ortho')。
    """
    complex_ks = torch.complex(kspace_2ch[..., 0, :, :], kspace_2ch[..., 1, :, :]) \
        if kspace_2ch.shape[-3] != 2 else torch.complex(kspace_2ch[:, :, 0], kspace_2ch[:, :, 1])
    image = torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.ifftshift(complex_ks, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    return torch.stack([image.real, image.imag], dim=2)  # (B, num_coils, 2, H, W)


class SensitivityModel(nn.Module):
    """
    参数:
        chans: SME内部小型U-Net的基础通道数(比重建网络更小，任务相对简单)
        num_pool_layers: SME内部U-Net的下采样层数
    """

    def __init__(self, chans=8, num_pool_layers=2):
        super().__init__()
        # 输入输出都是2通道(实部/虚部)，与重建网络中处理复数图像的方式一致
        self.norm_net = UNet(in_channels=2, out_channels=2, chans=chans, num_pool_layers=num_pool_layers)

    def forward(self, masked_kspace, num_low_freqs):
        """
        参数:
            masked_kspace: (B, num_coils, 2, H, W) 欠采样k-space
            num_low_freqs: 中心ACS区域宽度

        返回:
            sens_maps: (B, num_coils, 2, H, W)，归一化后的线圈灵敏度图(实部/虚部堆叠)
        """
        b, num_coils, ch, h, w = masked_kspace.shape

        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            masked_kspace = masked_kspace.float()
            acs_kspace = extract_acs_kspace(masked_kspace, num_low_freqs)
            # (B, num_coils, 2, H, W) -> IFFT -> 低分辨率的每线圈图像
            coil_images = ifft2c_torch(acs_kspace)

        # 用同一个共享CNN处理每个线圈：把batch和coil维度合并，一次性过网络，
        # 相当于对每个线圈独立、但用同一套参数做处理(权重共享)
        coil_images_flat = coil_images.reshape(b * num_coils, ch, h, w)
        refined_flat = self.norm_net(coil_images_flat)
        refined = refined_flat.reshape(b, num_coils, ch, h, w)

        # 归一化：保证 sum_c |S_c|^2 = 1，这是灵敏度加权合并公式成立的前提
        refined_complex = torch.complex(refined[:, :, 0], refined[:, :, 1])  # (B, num_coils, H, W)
        rss_norm = torch.sqrt(torch.sum(refined_complex.abs() ** 2, dim=1, keepdim=True) + 1e-12)
        normalized_complex = refined_complex / rss_norm

        sens_maps = torch.stack([normalized_complex.real, normalized_complex.imag], dim=2)
        return sens_maps  # (B, num_coils, 2, H, W)


def coil_combine(coil_images, sens_maps):
    """
    用估计出的灵敏度图，对多线圈图像做加权合并(SENSE风格合并公式)：
        x = sum_c conj(S_c) * image_c
    与RSS(简单平方和开根号)不同，这种合并方式利用了灵敏度图的相位信息，
    保留合并结果的相位(不是只取幅度)，是更精确的物理建模方式。

    参数:
        coil_images: (B, num_coils, 2, H, W) 每个线圈的复数图像(实部/虚部堆叠)
        sens_maps: (B, num_coils, 2, H, W) 归一化后的灵敏度图

    返回:
        combined: (B, 2, H, W) 合并后的单张复数图像
    """
    img_complex = torch.complex(coil_images[:, :, 0], coil_images[:, :, 1])
    sens_complex = torch.complex(sens_maps[:, :, 0], sens_maps[:, :, 1])
    combined_complex = torch.sum(torch.conj(sens_complex) * img_complex, dim=1)  # (B, H, W)
    return torch.stack([combined_complex.real, combined_complex.imag], dim=1)  # (B, 2, H, W)


if __name__ == "__main__":
    # 自检：验证SME模块前向传播形状正确，梯度能正常回传
    torch.manual_seed(0)
    sme = SensitivityModel(chans=8, num_pool_layers=2)

    batch_size, num_coils, h, w = 2, 6, 320, 320
    dummy_kspace = torch.randn(batch_size, num_coils, 2, h, w)
    num_low_freqs = 24

    sens_maps = sme(dummy_kspace, num_low_freqs)
    print(f"输入k-space形状: {dummy_kspace.shape}")
    print(f"输出灵敏度图形状: {sens_maps.shape}")
    assert sens_maps.shape == dummy_kspace.shape

    # 验证归一化性质: sum_c |S_c|^2 应该处处接近1
    sens_complex = torch.complex(sens_maps[:, :, 0], sens_maps[:, :, 1])
    norm_check = torch.sum(sens_complex.abs() ** 2, dim=1)
    print(f"归一化检查 sum_c|S_c|^2: mean={norm_check.mean().item():.6f} "
          f"(应接近1.0), std={norm_check.std().item():.6e}")

    # 测试coil_combine
    dummy_coil_images = torch.randn(batch_size, num_coils, 2, h, w)
    combined = coil_combine(dummy_coil_images, sens_maps)
    print(f"合并后图像形状: {combined.shape}")
    assert combined.shape == (batch_size, 2, h, w)

    # 梯度检查
    loss = combined.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in sme.parameters() if p.requires_grad)
    has_nan = any(torch.isnan(p.grad).any() for p in sme.parameters() if p.grad is not None)
    print(f"梯度正常回传: {has_grad}, 梯度含NaN: {has_nan}")
    print("SME模块自检通过")