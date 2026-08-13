"""
ROI(感兴趣区域)遮罩生成：在"背景遮罩"基础上进一步聚焦，只保留脑组织内部
纹理细节丰富的区域(如脑沟脑回)，排除相对均匀、即使模糊也不太影响诊断判断的
区域(如大片白质、脑脊液)。

设计动机：SSIM/PSNR这类全图/全前景平均的指标，容易被大面积"容易重建准确"的
均匀区域主导，稀释掉模型在真正需要精细结构还原的区域(细粒度纹理)的实际表现，
导致指标"看起来不错"但视觉上仍有明显模糊——这与文献中报告的SSIM"均匀池化"
局限性一致。ROI遮罩通过只在纹理密集区域计算指标，让评估结果更贴近"临床上
医生真正关心的重建质量"，而不是被大片背景/均匀组织稀释后的平均分数。

全流程自动化，不需要人工标注：基于Ground Truth图像的局部方差(反映纹理丰富
程度)，用阈值筛选出"纹理最密集的前N%区域"，作为自动生成的ROI。
"""

import numpy as np
from scipy import ndimage


def compute_local_variance(image, window_size=7):
    """
    计算图像每个像素位置的局部方差，用于衡量该位置周围的纹理丰富程度。
    脑沟脑回这类细节密集区域，局部像素值变化剧烈，局部方差高；
    均匀的白质/脑脊液区域，局部方差低。

    参数:
        image: 2D np.ndarray，输入图像(通常是Ground Truth)
        window_size: 局部窗口边长(奇数)，值越大，纹理度量越"平滑"/宏观

    返回:
        local_var: 2D np.ndarray，与输入同尺寸，每个位置是该处的局部方差
    """
    image = image.astype(np.float64)
    local_mean = ndimage.uniform_filter(image, size=window_size)
    local_sq_mean = ndimage.uniform_filter(image ** 2, size=window_size)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.clip(local_var, 0, None)  # 数值误差可能导致极小负数，裁剪保护
    return local_var


def compute_roi_mask(gt, background_mask=None, window_size=7, top_fraction=0.5):
    """
    自动生成ROI遮罩：在(可选的)背景遮罩基础上，进一步筛选出纹理最丰富的
    前 top_fraction 比例的前景像素，作为最终ROI。

    参数:
        gt: 2D np.ndarray，Ground Truth图像
        background_mask: 2D bool np.ndarray，已有的前景/背景遮罩(如compute_background_mask
                          的输出)，None则先自动生成
        window_size: 计算局部方差时的窗口大小
        top_fraction: 在前景区域内，保留局部方差最高的前多少比例作为ROI
                      (0.5表示保留纹理最丰富的前50%前景像素)

    返回:
        roi_mask: 2D bool np.ndarray，True表示该位置属于ROI(细节丰富、需要精细
                  评估的区域)
    """
    if background_mask is None:
        try:
            from skimage.filters import threshold_otsu
            try:
                threshold = threshold_otsu(gt)
            except ValueError:
                threshold = 0.05 * gt.max()
        except ImportError:
            threshold = 0.05 * gt.max()

        background_mask = gt > threshold
        background_mask = ndimage.binary_fill_holes(background_mask)
        background_mask = ndimage.binary_opening(background_mask, structure=np.ones((3, 3)))

        # 只保留面积最大的连通域，排除散落的背景噪声孤岛(与Otsu阈值配合，双重保险)
        labeled, num_features = ndimage.label(background_mask)
        if num_features > 1:
            sizes = ndimage.sum(background_mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            background_mask = (labeled == largest_label)

    local_var = compute_local_variance(gt, window_size=window_size)

    # 只在前景区域内排序，找到"前top_fraction比例"对应的方差阈值
    fg_variances = local_var[background_mask]
    if len(fg_variances) == 0:
        return background_mask  # 极端情况(全背景)，退化为原始前景遮罩

    var_threshold = np.percentile(fg_variances, (1 - top_fraction) * 100)

    roi_mask = background_mask & (local_var >= var_threshold)

    # 用形态学开运算清理孤立的噪点像素，让ROI边界更平滑连贯
    roi_mask = ndimage.binary_opening(roi_mask, structure=np.ones((2, 2)))

    return roi_mask


if __name__ == "__main__":
    # 自检：构造一个模拟"脑部图像"(中心纹理丰富，外围相对均匀)，验证ROI是否
    # 正确聚焦在纹理丰富区域
    np.random.seed(0)
    h, w = 128, 128
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")

    # 背景(颅骨外)：接近0
    brain_region = (xx ** 2 + yy ** 2) < 0.7

    # 模拟纹理：中心区域叠加高频噪声(模拟脑沟脑回)，外围区域平滑(模拟白质)
    base = np.exp(-(xx ** 2 + yy ** 2) * 1.5)
    texture_region = (xx ** 2 + yy ** 2) < 0.3  # 中心纹理丰富区
    texture_noise = np.random.randn(h, w) * 0.15 * texture_region
    gt = (base + texture_noise) * brain_region
    gt = np.clip(gt, 0, None).astype(np.float32)

    roi_mask = compute_roi_mask(gt, top_fraction=0.5)

    print(f"图像尺寸: {gt.shape}")
    print(f"背景遮罩(前景)覆盖像素占比: {brain_region.sum() / gt.size * 100:.1f}%")
    print(f"ROI遮罩覆盖像素占比: {roi_mask.sum() / gt.size * 100:.1f}%")

    # 验证ROI是否主要落在我们构造的"纹理丰富区域"内
    overlap_with_texture = (roi_mask & texture_region).sum() / roi_mask.sum()
    print(f"ROI与预设纹理丰富区域的重合比例: {overlap_with_texture * 100:.1f}% "
          f"(应显著高于50%，说明ROI确实聚焦在纹理密集区)")

    assert overlap_with_texture > 0.6, "ROI未能有效聚焦纹理丰富区域"
    print("\nROI遮罩自检通过")