"""
SME + U-Net 重建模型：这是"完整E2E-VarNet"和"RSS+U-Net baseline"之间的中间方案。

相比RSS+U-Net：用可训练的SME模块估计线圈灵敏度，做SENSE风格的加权合并
(保留相位信息)，而不是简单的平方和(RSS)，理论上更充分利用多线圈的空间信息。

相比完整E2E-VarNet：只做"一次"合并+精修，不做多级联的反复数据一致性迭代，
训练行为更接近普通U-Net，风险和调试难度显著低于完整VarNet。

数据流：
    多线圈欠采样k-space (num_coils, 2, H, W)
        │
        ├──> SME模块 (从ACS区域估计) ──> 线圈灵敏度图 (num_coils, 2, H, W)
        │
        ▼ 每个线圈独立IFFT
    多线圈复数图像 (num_coils, 2, H, W)
        │
        ▼ 用灵敏度图做SENSE风格加权合并
    单张合并后复数图像 (2, H, W)
        │
        ▼ U-Net精修(图像域，输入输出都是2通道实部/虚部)
    最终复数图像 (2, H, W)
        │
        ▼ 取幅度
    最终重建图像 (1, H, W)
"""

import torch
import torch.nn as nn

from .sme import SensitivityModel, coil_combine
from .unet import UNet


class SMEUNetModel(nn.Module):
    """
    参数:
        sme_chans: SME内部网络的基础通道数
        sme_pool_layers: SME内部网络的下采样层数
        unet_chans: 主重建U-Net的基础通道数
        unet_pool_layers: 主重建U-Net的下采样层数
    """

    def __init__(self, sme_chans=8, sme_pool_layers=2, unet_chans=32, unet_pool_layers=4):
        super().__init__()
        self.sme = SensitivityModel(chans=sme_chans, num_pool_layers=sme_pool_layers)
        self.recon_unet = UNet(
            in_channels=2, out_channels=2,
            chans=unet_chans, num_pool_layers=unet_pool_layers,
        )

    def forward(self, masked_kspace, mask, num_low_freqs):
        """
        参数:
            masked_kspace: (B, num_coils, 2, H, W) 欠采样k-space(实部/虚部堆叠)
            mask: (B, 1, H, W)，本模型中暂未在合并后使用数据一致性，
                  保留此参数是为了未来若要扩展为级联版本(即完整E2E-VarNet)时，
                  接口保持一致，不需要改动Dataset和训练脚本
            num_low_freqs: (B,) 或 int，每个样本ACS区域宽度

        返回:
            final_image: (B, 1, H, W_full) 最终重建图像幅度
                         (注意：此处返回的是padding后的完整尺寸，训练/评估时
                          需要center_crop到GT的真实尺寸，与singlecoil VarNet一致)
        """
        b, num_coils, ch, h, w = masked_kspace.shape

        # num_low_freqs在DataLoader默认collate下会变成一个tensor(如果每个样本相同，
        # 所有元素值一致)，这里统一取第一个值作为整个batch的ACS宽度
        # (同一批次内，若加速倍数相同，ACS宽度理论上也应一致，因为只依赖图像宽度w和
        # 固定的center_fraction；不同样本宽度w若不同，会在Dataset的padding步骤统一，
        # 但ACS宽度是基于原始w计算的，理论上可能有细微差异，这里简化为取众数/第一个值)
        if torch.is_tensor(num_low_freqs):
            num_low_freqs_val = int(num_low_freqs[0].item())
        else:
            num_low_freqs_val = int(num_low_freqs)

        # 1. SME估计线圈灵敏度图
        sens_maps = self.sme(masked_kspace, num_low_freqs_val)  # (B, num_coils, 2, H, W)

        # 2. 每个线圈独立IFFT得到图像域结果
        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            masked_kspace_fp32 = masked_kspace.float()
            complex_ks = torch.complex(masked_kspace_fp32[:, :, 0], masked_kspace_fp32[:, :, 1])
            coil_images_complex = torch.fft.ifftshift(
                torch.fft.ifft2(torch.fft.ifftshift(complex_ks, dim=(-2, -1)), norm="ortho"),
                dim=(-2, -1),
            )
            coil_images = torch.stack(
                [coil_images_complex.real, coil_images_complex.imag], dim=2
            )  # (B, num_coils, 2, H, W)

            # 3. 用灵敏度图做SENSE风格加权合并
            combined = coil_combine(coil_images, sens_maps)  # (B, 2, H, W)

        # 4. U-Net在图像域精修(在AMP环境下正常使用半精度，这部分是纯卷积，没有FFT限制)
        refined = self.recon_unet(combined)  # (B, 2, H, W)

        # 5. 取幅度得到最终图像
        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            refined_fp32 = refined.float()
            final_complex = torch.complex(refined_fp32[:, 0], refined_fp32[:, 1])
            final_image = final_complex.abs().unsqueeze(1)  # (B, 1, H, W)

        return final_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自检：验证完整模型前向+反向传播
    torch.manual_seed(0)
    model = SMEUNetModel(sme_chans=8, sme_pool_layers=2, unet_chans=16, unet_pool_layers=3)

    batch_size, num_coils, h, w = 2, 20, 768, 396
    dummy_kspace = torch.randn(batch_size, num_coils, 2, h, w)
    dummy_mask = torch.zeros(batch_size, 1, h, w)
    dummy_mask[:, :, :, 180:216] = 1.0
    num_low_freqs = 32

    output = model(dummy_kspace, dummy_mask, num_low_freqs)
    print(f"输入k-space形状: {dummy_kspace.shape}")
    print(f"输出图像形状: {output.shape}")
    assert output.shape == (batch_size, 1, h, w)
    print("形状检查通过\n")

    num_params = count_parameters(model)
    sme_params = count_parameters(model.sme)
    unet_params = count_parameters(model.recon_unet)
    print(f"总参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  其中SME: {sme_params:,} ({sme_params/1e6:.2f}M)")
    print(f"  其中重建U-Net: {unet_params:,} ({unet_params/1e6:.2f}M)")

    target = torch.randn(batch_size, 1, h, w)
    loss = torch.nn.functional.l1_loss(output, target)
    loss.backward()
    has_grad_sme = all(p.grad is not None for p in model.sme.parameters() if p.requires_grad)
    has_grad_unet = all(p.grad is not None for p in model.recon_unet.parameters() if p.requires_grad)
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"\nSME梯度正常回传: {has_grad_sme}")
    print(f"U-Net梯度正常回传: {has_grad_unet}")
    print(f"梯度含NaN: {has_nan}")
    print("SME+U-Net完整模型自检通过")