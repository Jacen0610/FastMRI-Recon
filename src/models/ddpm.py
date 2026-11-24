"""
DDPM 去噪网络与扩散过程(用于脑部MRI图像域无条件生成先验)。

设计要点:
- 骨干复用项目 unet.py 的 DoubleConv/Down/Up 组件(与重建网络同族,便于对比);
- 时间条件注入用 FiLM:正弦时间嵌入 -> MLP -> 每个尺度输出做 (1+gamma)*x + beta,
  改动最小、对 InstanceNorm 结构无侵入;
- 扩散调度: T=1000, 余弦 alpha_bar (Nichol & Dhariwal 2021), 对小数据更稳;
- 采样: DDIM 确定性步(eta=0), 步数可调(默认50), 为后续 DC 引导采样留接口。
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import DoubleConv, Down, Up


def timestep_embedding(t, dim=128):
    """正弦时间嵌入。t: (B,) 整数/浮点时刻 -> (B, dim)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeUNet(nn.Module):
    """带时间条件(FiLM)的 U-Net, 骨架与项目重建 U-Net 一致。"""

    def __init__(self, in_ch=1, out_ch=1, chans=64, num_pools=4, temb_dim=128):
        super().__init__()
        self.temb = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim),
        )
        self.in_conv = DoubleConv(in_ch, chans)
        self.downs = nn.ModuleList()
        self.films = nn.ModuleList([nn.Linear(temb_dim, 2 * chans)])
        ch = chans
        for _ in range(num_pools):
            self.downs.append(Down(ch, ch * 2))
            self.films.append(nn.Linear(temb_dim, 2 * ch * 2))
            ch *= 2
        self.ups = nn.ModuleList()
        for _ in range(num_pools):
            self.ups.append(Up(ch, ch // 2))
            self.films.append(nn.Linear(temb_dim, 2 * (ch // 2)))
            ch //= 2
        self.out_conv = nn.Conv2d(chans, out_ch, 1)

    def _film(self, x, e, lin):
        g, b = lin(e)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + torch.tanh(g)) + b

    def forward(self, x, t):
        e = self.temb(timestep_embedding(t, 128))
        h = self._film(self.in_conv(x), e, self.films[0])
        skips = [h]
        for i, down in enumerate(self.downs[:-1]):
            h = self._film(down(h), e, self.films[i + 1])
            skips.append(h)
        h = self._film(self.downs[-1](h), e, self.films[len(self.downs)])
        for i, up in enumerate(self.ups):
            h = up(h, skips[-1 - i])
            h = self._film(h, e, self.films[len(self.downs) + 1 + i])
        return self.out_conv(h)


def cosine_alphas_bar(T=1000, s=0.008):
    t = torch.linspace(0, 1, T + 1)
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return f / f[0]


class DDPM(nn.Module):
    """扩散过程: 训练(噪声预测MSE) + DDIM确定性采样。"""

    def __init__(self, chans=64, num_pools=4, T=1000):
        super().__init__()
        self.T = T
        self.net = TimeUNet(chans=chans, num_pools=num_pools)
        ab = cosine_alphas_bar(T)
        betas = 1 - ab[1:] / ab[:-1]
        betas = betas.clamp(1e-8, 0.999)
        alphas = 1 - betas
        ab = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1mab", (1 - ab).sqrt())

    def loss(self, x0):
        """x0: (B,1,H,W) 标准化图像。随机 t 加噪, 预测噪声。"""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.sqrt_ab[t][:, None, None, None] * x0 + self.sqrt_1mab[t][:, None, None, None] * eps
        return torch.nn.functional.mse_loss(self.net(xt, t), eps)

    @torch.no_grad()
    def predict_x0(self, xt, t):
        eps = self.net(xt, torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long))
        return (xt - self.sqrt_1mab[t] * eps) / self.sqrt_ab[t].clamp(min=1e-4), eps

    @torch.no_grad()
    def ddim_sample(self, n, device, size=(1, 384, 384), steps=50, callback=None):
        """DDIM 确定性采样(eta=0)。callback(x0_pred, t) 可插入数据一致性引导。"""
        ts = torch.linspace(self.T - 1, 0, steps).long().tolist()
        x = torch.randn(n, *size, device=device)
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else 0
            x0, eps = self.predict_x0(x, t)
            if callback is not None:
                x0 = callback(x0, t)                      # 引导: 在x0估计上做修正
            x = self.sqrt_ab[t_prev] * x0 + self.sqrt_1mab[t_prev] * eps
        return x.clamp(-3, 3)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = DDPM(chans=8, num_pools=2, T=1000)
    x = torch.randn(2, 1, 96, 96)
    l = m.loss(x)
    l.backward()
    xs = m.ddim_sample(1, "cpu", size=(1, 96, 96), steps=5)
    print("loss", l.item(), "sample", tuple(xs.shape))
    print("参数量:", sum(p.numel() for p in m.parameters()))
