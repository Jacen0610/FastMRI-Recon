"""
fastMRI 风格的笛卡尔欠采样 mask 生成。

设计逻辑：
- mask 沿"相位编码方向"（k-space的列方向）一维生成，再广播到整个二维k-space矩阵
- 中心 ACS (Auto-Calibration Signal) 区域全部保留（低频，决定图像整体轮廓/对比度）
- 外围区域按目标加速倍数随机欠采样（高频，决定图像细节，牺牲空间更大）

标准配置（对齐 fastMRI 官方论文设定）：
    - 4x 加速: center_fraction=0.08, acceleration=4  (约25%的列被保留)
    - 8x 加速: center_fraction=0.04, acceleration=8  (约12.5%的列被保留)
"""

import numpy as np


def create_cartesian_mask(shape, center_fraction, acceleration, seed=None):
    """
    生成一维笛卡尔欠采样 mask，并广播为与k-space匹配的二维mask。

    参数:
        shape: tuple, (height, width)，对应k-space单个切片的形状，如 (640, 368)
        center_fraction: float, 中心ACS区域占总列数的比例，如 0.08 (对应4x) 或 0.04 (对应8x)
        acceleration: int, 目标加速倍数，如 4 或 8
        seed: int or None, 随机种子。若为None则每次调用生成不同的随机mask
              (训练时建议每个样本用不同mask以增强泛化；验证/测试时建议固定seed保证可复现)

    返回:
        mask: np.ndarray, shape=(height, width)，float32类型，值为0或1
        mask_1d: np.ndarray, shape=(width,)，一维mask，便于检查实际保留比例
    """
    height, width = shape
    rng = np.random.RandomState(seed)

    # 1. 计算中心ACS区域的宽度，并标记为保留(1)
    num_low_freqs = int(round(width * center_fraction))
    mask_1d = np.zeros(width, dtype=np.float32)

    center_start = (width - num_low_freqs) // 2
    center_end = center_start + num_low_freqs
    mask_1d[center_start:center_end] = 1.0

    # 2. 在中心区域之外，按目标加速倍数随机采样，补足到目标保留比例
    #    目标总保留比例 = 1 / acceleration
    #    ACS区域已经贡献了 center_fraction 的保留比例，外围区域需要补足剩余部分
    target_overall_fraction = 1.0 / acceleration
    num_target_total = int(round(width * target_overall_fraction))
    num_additional = max(num_target_total - num_low_freqs, 0)

    # 外围候选位置：除ACS区域之外的所有列
    outer_indices = np.array(
        [i for i in range(width) if not (center_start <= i < center_end)]
    )

    if num_additional > 0 and len(outer_indices) > 0:
        num_additional = min(num_additional, len(outer_indices))
        chosen = rng.choice(outer_indices, size=num_additional, replace=False)
        mask_1d[chosen] = 1.0

    # 3. 广播为二维mask，匹配k-space形状 (height, width)
    #    因为欠采样只沿相位编码方向(列)进行，频率编码方向(行)始终完整保留
    mask_2d = np.tile(mask_1d, (height, 1)).astype(np.float32)

    actual_fraction = mask_1d.mean()
    return mask_2d, mask_1d, actual_fraction


def apply_mask(kspace, mask):
    """
    将mask应用到k-space上，模拟欠采样。

    参数:
        kspace: np.ndarray, complex64, shape=(height, width)，单张切片的完整k-space
        mask: np.ndarray, shape=(height, width)，0/1 mask

    返回:
        undersampled_kspace: np.ndarray, complex64, 同shape，未采样位置为0
    """
    return kspace * mask


# 标准加速倍数配置，对齐 fastMRI 官方论文设定
ACCELERATION_CONFIGS = {
    4: {"center_fraction": 0.08, "acceleration": 4},
    8: {"center_fraction": 0.04, "acceleration": 8},
}


def get_mask_for_acceleration(shape, acceleration, seed=None):
    """
    便捷接口：根据加速倍数(4或8)直接生成对应配置的mask。

    参数:
        shape: tuple, (height, width)
        acceleration: int, 4 或 8
        seed: int or None

    返回:
        mask_2d, mask_1d, actual_fraction (同 create_cartesian_mask)
    """
    if acceleration not in ACCELERATION_CONFIGS:
        raise ValueError(
            f"暂不支持 {acceleration}x 加速，目前仅配置了 {list(ACCELERATION_CONFIGS.keys())}"
        )
    config = ACCELERATION_CONFIGS[acceleration]
    return create_cartesian_mask(
        shape,
        center_fraction=config["center_fraction"],
        acceleration=config["acceleration"],
        seed=seed,
    )


if __name__ == "__main__":
    # 简单自检：打印4x和8x两种mask的实际保留比例，确认与预期(25%/12.5%)接近
    test_shape = (640, 368)
    for acc in [4, 8]:
        _, mask_1d, actual_fraction = get_mask_for_acceleration(test_shape, acc, seed=42)
        expected_fraction = 1.0 / acc
        print(
            f"{acc}x 加速 | 目标保留比例: {expected_fraction:.4f} | "
            f"实际保留比例: {actual_fraction:.4f} | 一维mask长度: {len(mask_1d)}"
        )