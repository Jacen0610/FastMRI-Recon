"""
完整 E2E-VarNet（End-to-End Variational Network），Sriram et al. 架构的简化实现。

在"SME+U-Net(单次)"基础上，把单次的"合并+精修"升级为多级联的"合并-精修-展开-
数据一致性"循环，每一轮都用真实测量数据(k-space层面)重新校正，这是完整
E2E-VarNet相比前一阶段"SME+U-Net"最核心的升级。

设计要点：
- 灵敏度图(sens_maps)只在开始时用SME估计一次，所有级联共享同一份，这是官方
  E2E-VarNet的标准做法(灵敏度图代表的是线圈的物理特性，在整个重建过程中不变，
  不需要每个级联重新估计)
- 每个级联内部：
    1. 用sens_maps把当前k-space估计"合并"为单张复数图像(coil_combine)
    2. CNN在图像域精修
    3. 用sens_maps把精修后的图像"展开"回多线圈k-space(coil_combine的伴随操作，
       即SENSE正向模型: image_c = S_c * combined_image)
    4. 数据一致性：已采样位置强制替换回真实测量值(每个线圈独立操作)
"""

import torch
import torch.nn as nn

from .sme import SensitivityModel, coil_combine
from .unet import UNet


def ifft2c_multicoil(kspace_2ch):
    """对(B, num_coils, 2, H, W)格式的k-space做正交归一化IFFT"""
    complex_ks = torch.complex(kspace_2ch[:, :, 0], kspace_2ch[:, :, 1])
    image = torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.ifftshift(complex_ks, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    return torch.stack([image.real, image.imag], dim=2)


def fft2c_multicoil(image_2ch):
    """对(B, num_coils, 2, H, W)格式的图像做正交归一化FFT"""
    complex_img = torch.complex(image_2ch[:, :, 0], image_2ch[:, :, 1])
    kspace = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.fftshift(complex_img, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    return torch.stack([kspace.real, kspace.imag], dim=2)


def expand_to_coils(combined_2ch, sens_maps):
    """
    coil_combine的伴随操作：把单张合并图像，按SENSE正向模型展开回多线圈图像。
        image_c = S_c * combined_image
    这是数据一致性能在多线圈k-space上生效的必要前置步骤。

    参数:
        combined_2ch: (B, 2, H, W) 单张复数图像
        sens_maps: (B, num_coils, 2, H, W) 线圈灵敏度图

    返回:
        coil_images: (B, num_coils, 2, H, W)
    """
    combined_complex = torch.complex(combined_2ch[:, 0], combined_2ch[:, 1])  # (B, H, W)
    sens_complex = torch.complex(sens_maps[:, :, 0], sens_maps[:, :, 1])  # (B, num_coils, H, W)
    coil_images_complex = sens_complex * combined_complex.unsqueeze(1)  # 广播到每个线圈
    return torch.stack([coil_images_complex.real, coil_images_complex.imag], dim=2)


class DataConsistencyMulticoil(nn.Module):
    """多线圈版数据一致性：每个线圈独立做已采样位置的强制替换"""

    def forward(self, refined_kspace, measured_kspace, mask):
        # mask: (B, 1, H, W)，广播到 (B, num_coils, 2, H, W)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, 1, H, W)
        return mask_expanded * measured_kspace + (1 - mask_expanded) * refined_kspace


class E2EVarNetCascadeBlock(nn.Module):
    """单个级联：合并 -> CNN精修 -> 展开 -> 数据一致性"""

    def __init__(self, chans=18, num_pool_layers=3):
        super().__init__()
        self.refine_net = UNet(in_channels=2, out_channels=2, chans=chans, num_pool_layers=num_pool_layers)
        self.dc = DataConsistencyMulticoil()

    def forward(self, current_kspace, measured_kspace, mask, sens_maps):
        with torch.autocast(device_type=current_kspace.device.type, enabled=False):
            current_kspace_fp32 = current_kspace.float()
            coil_images = ifft2c_multicoil(current_kspace_fp32)
            combined = coil_combine(coil_images, sens_maps.float())

        refined_combined = self.refine_net(combined)

        with torch.autocast(device_type=current_kspace.device.type, enabled=False):
            refined_combined_fp32 = refined_combined.float()
            refined_coil_images = expand_to_coils(refined_combined_fp32, sens_maps.float())
            refined_kspace = fft2c_multicoil(refined_coil_images)
            return self.dc(refined_kspace, measured_kspace.float(), mask.float())


class E2EVarNet(nn.Module):
    """
    参数:
        num_cascades: 级联层数
        sme_chans / sme_pool_layers: SME内部网络配置
        cascade_chans / cascade_pool_layers: 每个级联内部CNN配置
    """

    def __init__(self, num_cascades=4, sme_chans=8, sme_pool_layers=2,
                 cascade_chans=18, cascade_pool_layers=3):
        super().__init__()
        self.sme = SensitivityModel(chans=sme_chans, num_pool_layers=sme_pool_layers)
        self.cascades = nn.ModuleList([
            E2EVarNetCascadeBlock(chans=cascade_chans, num_pool_layers=cascade_pool_layers)
            for _ in range(num_cascades)
        ])

    def forward(self, masked_kspace, mask, num_low_freqs):
        """
        参数:
            masked_kspace: (B, num_coils, 2, H, W)
            mask: (B, 1, H, W)
            num_low_freqs: ACS区域宽度

        返回:
            final_image: (B, 1, H, W) 最终重建图像幅度
        """
        if torch.is_tensor(num_low_freqs):
            num_low_freqs_val = int(num_low_freqs[0].item())
        else:
            num_low_freqs_val = int(num_low_freqs)

        # 灵敏度图只估计一次，所有级联共享
        sens_maps = self.sme(masked_kspace, num_low_freqs_val)

        current_kspace = masked_kspace
        for cascade in self.cascades:
            current_kspace = cascade(current_kspace, masked_kspace, mask, sens_maps)

        with torch.autocast(device_type=masked_kspace.device.type, enabled=False):
            final_coil_images = ifft2c_multicoil(current_kspace.float())
            final_combined = coil_combine(final_coil_images, sens_maps.float())
            final_complex = torch.complex(final_combined[:, 0], final_combined[:, 1])
            final_image = final_complex.abs().unsqueeze(1)

        return final_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自检：用小尺寸验证前向/反向传播正确性(真实尺寸训练在GPU上进行)
    torch.manual_seed(0)
    model = E2EVarNet(num_cascades=2, sme_chans=4, sme_pool_layers=2,
                       cascade_chans=8, cascade_pool_layers=2)

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
    sme_params = count_parameters(model.sme)
    cascade_params = sum(count_parameters(c) for c in model.cascades)
    print(f"总参数量: {num_params:,}")
    print(f"  其中SME: {sme_params:,}")
    print(f"  其中{len(model.cascades)}个级联合计: {cascade_params:,}")

    target = torch.randn(batch_size, 1, h, w)
    loss = torch.nn.functional.l1_loss(output, target)
    loss.backward()
    has_grad_sme = all(p.grad is not None for p in model.sme.parameters() if p.requires_grad)
    has_grad_cascades = all(
        p.grad is not None for c in model.cascades for p in c.parameters() if p.requires_grad
    )
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"\nSME梯度正常回传: {has_grad_sme}")
    print(f"级联梯度正常回传: {has_grad_cascades}")
    print(f"梯度含NaN: {has_nan}")
    print("完整E2E-VarNet自检通过")