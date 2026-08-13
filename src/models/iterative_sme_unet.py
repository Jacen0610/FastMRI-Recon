"""
迭代式 SME + U-Net：在"单次SME+U-Net"基础上，引入灵敏度图的二次估计更新，
直接对应文献(JDSI, arXiv:2210.12723)中指出的E2E-VarNet局限性——
"仅用ACS区域估计一次灵敏度图，且重建过程中不再更新"，并针对性改进。

核心思路：
    第一轮：用ACS区域估计初始灵敏度图(信息有限，尤其在高加速倍数下ACS更窄)
        -> 合并 -> U-Net精修 -> 得到一张"初步去除部分伪影"的中间图像

    第二轮：用这张已经更清晰的中间图像作为新的参考信息，重新估计一次灵敏度图
        (这次的输入不再局限于原始ACS区域，而是基于更完整的图像内容，
         这正是文献中提出的"应该更新灵敏度图"的核心思路)
        -> 用更新后的灵敏度图重新合并原始欠采样图像 -> U-Net精修 -> 最终输出

这个设计只做"一次"更新(而非JDSI那样在每一步都持续更新)，
是一个复杂度可控、但直接验证"灵敏度图更新是否有效"这一核心假设的实现。
"""

import torch
import torch.nn as nn

from .sme import SensitivityModel, coil_combine
from .unet import UNet


class IterativeSMEUNetModel(nn.Module):
    """
    参数:
        sme_chans / sme_pool_layers: SME内部网络配置(初始估计和更新估计共用同一个SME网络，
                                       权重共享——这样"更新"学到的是"如何根据更清晰的图像
                                       修正灵敏度图"，而不是引入一套全新的、未经初始估计
                                       训练过的参数)
        unet_chans / unet_pool_layers: 每一轮图像域精修网络的配置(两轮各自独立的U-Net，
                                         因为"从粗糙输入精修"和"从已初步去伪影的输入精修"
                                         是难度不同的两个任务，用独立参数更合理)
    """

    def __init__(self, sme_chans=8, sme_pool_layers=2, unet_chans=32, unet_pool_layers=4):
        super().__init__()
        self.sme = SensitivityModel(chans=sme_chans, num_pool_layers=sme_pool_layers)
        self.recon_unet_1 = UNet(in_channels=2, out_channels=2, chans=unet_chans, num_pool_layers=unet_pool_layers)
        self.recon_unet_2 = UNet(in_channels=2, out_channels=2, chans=unet_chans, num_pool_layers=unet_pool_layers)

    def _multicoil_ifft(self, kspace_2ch):
        with torch.autocast(device_type=kspace_2ch.device.type, enabled=False):
            kspace_fp32 = kspace_2ch.float()
            complex_ks = torch.complex(kspace_fp32[:, :, 0], kspace_fp32[:, :, 1])
            image = torch.fft.ifftshift(
                torch.fft.ifft2(torch.fft.ifftshift(complex_ks, dim=(-2, -1)), norm="ortho"),
                dim=(-2, -1),
            )
            return torch.stack([image.real, image.imag], dim=2)

    def forward(self, masked_kspace, mask, num_low_freqs):
        """
        参数:
            masked_kspace: (B, num_coils, 2, H, W)
            mask: (B, 1, H, W)，本模型未使用数据一致性，保留接口一致性
            num_low_freqs: ACS区域宽度

        返回:
            final_image: (B, 1, H, W)
        """
        if torch.is_tensor(num_low_freqs):
            num_low_freqs_val = int(num_low_freqs[0].item())
        else:
            num_low_freqs_val = int(num_low_freqs)

        # ---- 第一轮：用ACS区域估计初始灵敏度图 ----
        sens_maps_v1 = self.sme(masked_kspace, num_low_freqs_val)

        coil_images = self._multicoil_ifft(masked_kspace)
        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            combined_v1 = coil_combine(coil_images, sens_maps_v1.float())

        refined_v1 = self.recon_unet_1(combined_v1)  # 第一轮的中间重建结果

        # ---- 第二轮：用更清晰的中间结果，重新估计灵敏度图 ----
        # 把第一轮精修结果重新变换到k-space域，作为SME的新输入
        # (SME的设计本身是从k-space的ACS区域估计，这里我们用"精修后图像重新变换回
        #  的k-space"里同样位置的ACS区域，此时因为图像已经初步去噪，这部分频谱信息
        #  比原始欠采样数据更干净，理论上能支持更准的灵敏度估计)
        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            refined_v1_fp32 = refined_v1.float()
            refined_complex = torch.complex(refined_v1_fp32[:, 0], refined_v1_fp32[:, 1])
            refined_kspace_complex = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.fftshift(refined_complex, dim=(-2, -1)), norm="ortho"),
                dim=(-2, -1),
            )
            # 展开回多线圈k-space(用第一轮的灵敏度图做展开，作为SME第二轮的输入基础)
            sens_complex_v1 = torch.complex(sens_maps_v1[:, :, 0], sens_maps_v1[:, :, 1])
            expanded_complex = sens_complex_v1 * refined_kspace_complex.unsqueeze(1)
            pseudo_kspace = torch.stack([expanded_complex.real, expanded_complex.imag], dim=2)

        sens_maps_v2 = self.sme(pseudo_kspace, num_low_freqs_val)

        # ---- 用更新后的灵敏度图，重新合并原始(未精修)的欠采样图像 ----
        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            combined_v2 = coil_combine(coil_images, sens_maps_v2.float())

        refined_v2 = self.recon_unet_2(combined_v2)  # 最终重建结果

        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            refined_v2_fp32 = refined_v2.float()
            final_complex = torch.complex(refined_v2_fp32[:, 0], refined_v2_fp32[:, 1])
            final_image = final_complex.abs().unsqueeze(1)

        return final_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = IterativeSMEUNetModel(sme_chans=4, sme_pool_layers=2, unet_chans=8, unet_pool_layers=2)

    batch_size, num_coils, h, w = 1, 6, 96, 96
    dummy_kspace = torch.randn(batch_size, num_coils, 2, h, w)
    dummy_mask = torch.zeros(batch_size, 1, h, w)
    dummy_mask[:, :, :, 40:56] = 1.0
    num_low_freqs = 16

    output = model(dummy_kspace, dummy_mask, num_low_freqs)
    print(f"输入k-space形状: {dummy_kspace.shape}")
    print(f"输出图像形状: {output.shape}")
    assert output.shape == (batch_size, 1, h, w)
    print("形状检查通过\n")

    num_params = count_parameters(model)
    print(f"总参数量: {num_params:,}")

    target = torch.randn(batch_size, 1, h, w)
    loss = torch.nn.functional.l1_loss(output, target)
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"梯度正常回传: {has_grad}")
    print(f"梯度含NaN: {has_nan}")
    print("迭代式SME+U-Net自检通过")