"""
PUIP: Pixel-wise Unequal Interval Partition
按像素的不等间隔深度采样模块。

PUIP 在级联 MVS 网络的后续 stage 中，根据上一阶段输出的概率体与深度采样值，
针对每个像素自适应地构造非均匀深度采样区间，从而把更多的采样点放在概率高、
邻域梯度大的位置，提升深度估计精度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PUIP(nn.Module):
    """Pixel-wise Unequal Interval Partition.

    用法：
        puip = PUIP()
        depth_range_samples = puip.sample(
            cur_depth, ndepth, depth_inteval_pixel, shape,
            prob_volume_prev, depth_values_prev,
        )
    或直接通过 ``get_depth_range_samples`` 统一调度（兼容初始 stage）。
    """

    def __init__(self):
        super(PUIP, self).__init__()

    @staticmethod
    def sample(cur_depth, ndepth, depth_inteval_pixel, shape,
               prob_volume_prev=None, depth_values_prev=None):
        """严格按照 PUIP 算法计算 Hmin/Hmax，并基于像素梯度做非均匀采样。"""
        # 如果没有提供前一阶段的概率信息，使用均匀采样作为兜底
        if prob_volume_prev is None or depth_values_prev is None:
            cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)
            cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
            new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)

            depth_range_samples = cur_depth_min.unsqueeze(1) + (
                torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype
                             ).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
            return depth_range_samples

        B, H, W = cur_depth.shape

        # 关键修改：将前一阶段的信息上采样到当前阶段的尺寸
        prob_volume_prev_upsampled = F.interpolate(
            prob_volume_prev, size=(H, W), mode='bilinear', align_corners=False)
        depth_values_prev_upsampled = F.interpolate(
            depth_values_prev, size=(H, W), mode='bilinear', align_corners=False)

        # 扩展 cur_depth 的维度 (B, H, W) -> (B, 1, H, W)
        Hx_expanded = cur_depth.unsqueeze(1)

        # 计算差异平方
        diff_sq = (depth_values_prev_upsampled - Hx_expanded) ** 2

        # 创建掩码
        mask_big = depth_values_prev_upsampled >= Hx_expanded
        mask_small = depth_values_prev_upsampled < Hx_expanded

        # 加权平方和
        big = (prob_volume_prev_upsampled * diff_sq * mask_big).sum(dim=1)
        small = (prob_volume_prev_upsampled * diff_sq * mask_small).sum(dim=1)

        # 计算 HBig 和 HSmall
        H_big = torch.sqrt(big.clamp(min=1e-8))
        H_small = torch.sqrt(small.clamp(min=1e-8))

        # 计算 Hmin 和 Hmax
        H_min = cur_depth - H_small
        H_max = cur_depth + H_big

        # 8 邻域深度极值（向量化）
        padded_depth = F.pad(cur_depth.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        neighbor_max = F.max_pool2d(padded_depth, kernel_size=3, stride=1, padding=0).squeeze(1)
        neighbor_min = -F.max_pool2d(-padded_depth, kernel_size=3, stride=1, padding=0).squeeze(1)

        # 计算 Smax 与 Smin
        Smax = neighbor_max - cur_depth
        Smin = cur_depth - neighbor_min

        # 处理 Smax 或 Smin <= 0 的情况
        Smax = torch.clamp_min(Smax, 0)
        Smin = torch.clamp_min(Smin, 0)

        # 向量化计算 Nmax 与 Nmin
        denominator = Smax + Smin
        valid_mask = denominator > 1e-8

        Nmax = torch.zeros_like(Smax)
        Nmin = torch.zeros_like(Smin)
        Nmax[valid_mask] = ndepth * Smax[valid_mask] / denominator[valid_mask]
        Nmin[valid_mask] = ndepth * Smin[valid_mask] / denominator[valid_mask]

        # 边界情况处理
        Nmax = torch.where((Smax <= 0) & (Smin > 0), torch.ones_like(Nmax), Nmax)
        Nmin = torch.where((Smin <= 0) & (Smax > 0), torch.ones_like(Nmin), Nmin)
        both_zero = (Smax <= 0) & (Smin <= 0)
        Nmax[both_zero] = ndepth // 2
        Nmin[both_zero] = ndepth - ndepth // 2

        # 四舍五入并调整
        Nmax_int = torch.round(Nmax).long()
        Nmin_int = torch.round(Nmin).long()

        # 确保总和为 ndepth
        total = Nmax_int + Nmin_int
        diff = ndepth - total
        Nmax_int = Nmax_int + diff

        # 限定到合理范围
        Nmax_int = torch.clamp(Nmax_int, 1, ndepth - 1)
        Nmin_int = ndepth - Nmax_int

        # ================== 关键优化：向量化采样 ==================
        indices = torch.arange(ndepth, device=cur_depth.device).float()

        # 每个像素的采样间隔
        interval_max = (H_max - cur_depth) / Nmax_int.float()
        interval_min = (cur_depth - H_min) / Nmin_int.float()

        # 采样掩码
        mask_max = indices.reshape(1, 1, 1, -1) < Nmax_int.unsqueeze(-1)
        mask_min = indices.reshape(1, 1, 1, -1) >= (ndepth - Nmin_int).unsqueeze(-1)

        # Hmax 段采样
        k_max = indices.reshape(1, 1, 1, -1) + 1  # k 从 1 开始
        samples_max = cur_depth.unsqueeze(-1) + k_max * interval_max.unsqueeze(-1)
        samples_max = torch.where(mask_max, samples_max, torch.nan)

        # Hmin 段采样（包含 H(x)）
        k_min = ndepth - indices.reshape(1, 1, 1, -1)  # 反向计数
        samples_min = cur_depth.unsqueeze(-1) - k_min * interval_min.unsqueeze(-1)
        samples_min = torch.where(mask_min, samples_min, torch.nan)

        # 合并采样点
        depth_range_samples = torch.where(mask_max, samples_max, samples_min)

        # 处理 Nmax 或 Nmin 为 0 的特殊情况
        zero_max_mask = (Nmax_int == 0).unsqueeze(-1)
        zero_min_mask = (Nmin_int == 0).unsqueeze(-1)
        depth_range_samples = torch.where(zero_max_mask, samples_min, depth_range_samples)
        depth_range_samples = torch.where(zero_min_mask, samples_max, depth_range_samples)

        # 排序保证采样深度递增
        depth_range_samples = depth_range_samples.reshape(B, H, W, ndepth)
        sorted_indices = torch.argsort(depth_range_samples, dim=-1)

        batch_indices = torch.arange(B, device=cur_depth.device).reshape(B, 1, 1, 1).expand(-1, H, W, ndepth)
        height_indices = torch.arange(H, device=cur_depth.device).reshape(1, H, 1, 1).expand(B, -1, W, ndepth)
        width_indices = torch.arange(W, device=cur_depth.device).reshape(1, 1, W, 1).expand(B, H, -1, ndepth)

        depth_range_samples_sorted = depth_range_samples[
            batch_indices, height_indices, width_indices, sorted_indices
        ]

        # 转回 (B, ndepth, H, W) 格式
        depth_range_samples = depth_range_samples_sorted.permute(0, 3, 1, 2)
        return depth_range_samples


# ----------------------------------------------------------------------------
# 兼容入口：保留与旧代码一致的函数式调用方式
# ----------------------------------------------------------------------------

# 全局共享一个 PUIP 实例（无可学习参数，纯算法模块）
_PUIP_SINGLETON = PUIP()


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape,
                                prob_volume_prev=None, depth_values_prev=None):
    """与原代码同名的兼容函数，内部转调 PUIP.sample。"""
    return _PUIP_SINGLETON.sample(
        cur_depth, ndepth, depth_inteval_pixel, shape,
        prob_volume_prev=prob_volume_prev, depth_values_prev=depth_values_prev,
    )


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape, stage,
                            prob_volume_prev=None, depth_values_prev=None):
    """统一的深度采样入口：stage1 走均匀采样，stage2/3 走 PUIP。"""
    if cur_depth.dim() == 2:  # stage 1 - 初始深度值
        cur_depth_min = cur_depth[:, 0]   # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)

        depth_range_samples = cur_depth_min.unsqueeze(1) + (
            torch.arange(0, ndepth, device=device, dtype=dtype, requires_grad=False
                         ).reshape(1, -1) * new_interval.unsqueeze(1))  # (B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, shape[1], shape[2])  # (B, D, H, W)
    else:  # stage 2/3 - 使用前一阶段信息
        depth_range_samples = get_cur_depth_range_samples(
            cur_depth, ndepth, depth_inteval_pixel, shape, prob_volume_prev, depth_values_prev)

    return depth_range_samples


def uncertainty_aware_samples(cur_depth, depth_min, depth_max, exp_var, ndepth, device, dtype, shape):
    """UCS-Net 中使用的不确定性感知采样（保留以兼容旧接口）。"""
    eps = 1e-12
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)
        depth_range_samples = cur_depth_min.unsqueeze(1) + (
            torch.arange(0, ndepth, device=device, dtype=dtype, requires_grad=False
                         ).reshape(1, -1) * new_interval.unsqueeze(1))
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, shape[1], shape[2])
    else:
        batch_num, d_num, w_num, h_num = cur_depth.shape
        low_bound = cur_depth - exp_var
        high_bound = cur_depth + exp_var

        tensor_depth_min = depth_min.view(batch_num, 1, 1, 1).repeat(1, 1, w_num, h_num)
        tensor_depth_max = depth_max.view(batch_num, 1, 1, 1).repeat(1, 1, w_num, h_num)

        lower_than_min = (low_bound - tensor_depth_min) < 0
        higher_than_max = (high_bound - tensor_depth_max) > 0
        low_bound[lower_than_min] = tensor_depth_min[lower_than_min]
        high_bound[higher_than_max] = tensor_depth_max[higher_than_max]

        assert ndepth > 1
        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []
        for i in range(int(ndepth)):
            new_samps.append(low_bound + step * i + eps)

        depth_range_samples = torch.cat(new_samps, 1)

    return depth_range_samples
