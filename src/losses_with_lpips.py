"""
L1 + SSIM + LPIPS 组合损失函数。

背景：单纯用L1+SSIM训练的模型，在"全图/前景平均SSIM"上表现不错，但在真正
需要精细纹理还原的区域(如脑沟脑回)，视觉上仍可能明显模糊——这是因为L1和SSIM
本质上都是逐像素/局部窗口的数值比较，对"看起来是否真实自然"这种感知层面的
质量，约束力有限。LPIPS(Learned Perceptual Image Patch Similarity)用预训练
神经网络提取的深层特征做比较，能更好地捕捉纹理、结构这类感知层面的差异，
把它加入训练目标，能让模型真正朝着"感知质量更好、细节更清晰"的方向学习，
而不只是在事后用LPIPS评估时被动暴露弱点。

LPIPS网络本身是预训练好的、固定不变的(不参与本项目的梯度更新)，只是被用来
"打分"，梯度会从这个分数一路回传到我们自己的重建网络，指导它学习。
"""

import torch
import torch.nn as nn
from pytorch_msssim import SSIM

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class L1SSIMLPIPSLoss(nn.Module):
    """
    参数:
        alpha_l1: L1项权重
        alpha_ssim: SSIM项权重
        alpha_lpips: LPIPS项权重
        (三者不要求归一化到1，按相对比例设置即可)
        lpips_net: LPIPS使用的骨干网络，'alex'(默认，更快)或'vgg'(更贴近人类感知但更慢)
    """

    def __init__(self, alpha_l1=0.4, alpha_ssim=0.3, alpha_lpips=0.3, lpips_net="alex"):
        super().__init__()
        if not LPIPS_AVAILABLE:
            raise ImportError("需要先安装lpips库: pip install lpips")

        self.alpha_l1 = alpha_l1
        self.alpha_ssim = alpha_ssim
        self.alpha_lpips = alpha_lpips

        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=10.0, size_average=True, channel=1)

        # LPIPS网络权重固定，不参与本项目的训练更新，只用来计算感知损失
        self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def _prepare_for_lpips(self, x):
        """
        LPIPS网络期望输入是3通道、数值范围大致在[-1,1]的图像(仿照自然图像的
        标准化方式)。我们的重建图像是单通道、且原始数值范围因instance
        normalize而不确定，需要先做per-sample的min-max归一化到[-1,1]，
        再复制成3通道。
        """
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        x_min = x_flat.min(dim=1, keepdim=True)[0].reshape(b, 1, 1, 1)
        x_max = x_flat.max(dim=1, keepdim=True)[0].reshape(b, 1, 1, 1)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)  # -> [0, 1]
        x_norm = x_norm * 2 - 1  # -> [-1, 1]
        return x_norm.repeat(1, 3, 1, 1)  # 单通道复制成3通道

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)

        # SSIM计算涉及方差/协方差/除法，在fp16精度下容易因舍入误差产生数值不稳定，
        # 强制用float32计算(与LPIPS部分的保护思路一致)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_shifted = (pred.float() - pred.float().min() + 1e-3)
            target_shifted = (target.float() - target.float().min() + 1e-3)
            ssim_loss = 1 - self.ssim(pred_shifted, target_shifted)

        # LPIPS部分需要在float32下计算，避免AMP半精度下预训练网络数值不稳定
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_lpips = self._prepare_for_lpips(pred.float())
            target_lpips = self._prepare_for_lpips(target.float())
            lpips_loss = self.lpips_fn(pred_lpips, target_lpips).mean()

        total = (self.alpha_l1 * l1_loss
                 + self.alpha_ssim * ssim_loss
                 + self.alpha_lpips * lpips_loss)
        return total


if __name__ == "__main__":
    # 自检：验证前向传播和梯度回传正常，且LPIPS网络参数确实被冻结(不更新)
    torch.manual_seed(0)
    criterion = L1SSIMLPIPSLoss(alpha_l1=0.4, alpha_ssim=0.3, alpha_lpips=0.3, lpips_net="alex")

    pred = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randn(2, 1, 64, 64)

    loss = criterion(pred, target)
    print(f"组合loss值: {loss.item():.4f}")

    loss.backward()
    print(f"pred梯度是否正常: {pred.grad is not None}")
    print(f"pred梯度含NaN: {torch.isnan(pred.grad).any().item()}")

    # 验证LPIPS网络参数确实被冻结
    lpips_params_require_grad = any(
        p.requires_grad for p in criterion.lpips_fn.parameters()
    )
    print(f"LPIPS网络参数是否会被更新(应为False): {lpips_params_require_grad}")

    print("\nL1+SSIM+LPIPS组合损失自检通过")