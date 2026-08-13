"""
U-Net，用于MRI重建任务中的图像域后处理（伪影去除）。

任务设定：
    输入: Zero-filled重建图像 (带混叠伪影), shape (B, 1, H, W)
    输出: 去除伪影后的重建图像, shape (B, 1, H, W)

设计考量（针对8GB显存的消费级GPU）：
    - 基础通道数(chans)默认32，不用更宽的配置，控制显存占用
    - 4次下采样/上采样，对320x320输入足够提取多尺度特征，不需要更深
    - 每层用两次卷积(double conv)是U-Net的标准设计，可提取更充分的局部特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(卷积 -> BatchNorm -> ReLU) x 2，U-Net中每一层的基本组成单元"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """编码器的一层：先下采样(2x2 maxpool)，再做DoubleConv提取特征"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """
    解码器的一层：先上采样(转置卷积)放大尺寸，
    再和编码器对应层的特征做跳跃连接(拼接)，最后DoubleConv融合特征
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 转置卷积：把通道数减半的同时，把空间尺寸放大2倍
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 拼接后通道数 = in_channels//2 (来自上采样) + in_channels//2 (来自跳跃连接)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)

        # 处理尺寸不完全整除2的边缘情况，用padding对齐后再拼接
        diff_h = skip_connection.shape[2] - x.shape[2]
        diff_w = skip_connection.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])

        # 跳跃连接：在通道维度拼接编码器保留的细节特征
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    标准U-Net结构，用于图像域伪影去除。

    参数:
        in_channels: 输入通道数，默认1（单通道幅度图像）
        out_channels: 输出通道数，默认1
        chans: 第一层基础通道数，后续每下采样一次通道数翻倍
        num_pool_layers: 下采样/上采样的层数(即"U"的深度)
    """

    def __init__(self, in_channels=1, out_channels=1, chans=32, num_pool_layers=4):
        super().__init__()
        self.num_pool_layers = num_pool_layers

        # 输入层：先做一次DoubleConv，不下采样
        self.in_conv = DoubleConv(in_channels, chans)

        # 编码器：每层通道数翻倍 (chans -> 2*chans -> 4*chans -> 8*chans -> 16*chans)
        self.down_layers = nn.ModuleList()
        ch = chans
        for _ in range(num_pool_layers):
            self.down_layers.append(Down(ch, ch * 2))
            ch *= 2

        # 解码器：每层通道数减半，与编码器对称
        self.up_layers = nn.ModuleList()
        for _ in range(num_pool_layers):
            self.up_layers.append(Up(ch, ch // 2))
            ch //= 2

        # 输出层：1x1卷积把通道数映射回目标输出通道数
        self.out_conv = nn.Conv2d(chans, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器前向传播，同时保存每一层的输出用于跳跃连接
        skip_connections = []
        x = self.in_conv(x)
        skip_connections.append(x)

        for down in self.down_layers[:-1]:
            x = down(x)
            skip_connections.append(x)

        # 最后一层下采样不需要保存(它直接送入解码器第一层作为输入)
        x = self.down_layers[-1](x)

        # 解码器前向传播，依次和保存的跳跃连接拼接(逆序取用)
        for up, skip in zip(self.up_layers, reversed(skip_connections)):
            x = up(x, skip)

        return self.out_conv(x)


def count_parameters(model):
    """统计可训练参数量，方便评估模型规模是否适合当前显存"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自检：验证输入输出形状一致，并打印参数量和粗略显存占用估计
    model = UNet(in_channels=1, out_channels=1, chans=32, num_pool_layers=4)

    dummy_input = torch.randn(2, 1, 320, 320)  # (batch=2, channel=1, H=320, W=320)
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == dummy_input.shape, "输入输出形状不一致！"
    print("形状检查通过：输入输出尺寸一致\n")

    num_params = count_parameters(model)
    print(f"可训练参数量: {num_params:,} ({num_params / 1e6:.2f}M)")
    # 粗略估计：fp32下每个参数4字节，模型本身占用(不含中间激活值/梯度/优化器状态)
    print(f"模型权重显存占用(fp32粗估): {num_params * 4 / 1024**2:.1f} MB")
    print("(实际训练时显存占用还包括中间激活值、梯度、优化器状态，通常是模型权重的数倍，"
          "建议开启混合精度训练(AMP)并从小batch_size开始测试)")