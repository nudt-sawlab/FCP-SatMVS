"""PUIP: Pixel-wise Unequal Interval Partition.

In later cascade stages, PUIP derives a per-pixel non-uniform depth
sampling interval from (i) the probability volume and depth hypotheses of
the previous stage, and (ii) the local depth gradient of the current
estimate. This concentrates samples where the previous-stage posterior is
sharp and the local geometry is steep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PUIP(nn.Module):
    """Pixel-wise Unequal Interval Partition.

    Usage::

        puip = PUIP()
        depth_range_samples = puip.sample(
            cur_depth, ndepth, depth_inteval_pixel, shape,
            prob_volume_prev, depth_values_prev,
        )

    or use :func:`get_depth_range_samples` as a unified entry point that
    also handles the initial (uniform) stage.
    """

    def __init__(self):
        super(PUIP, self).__init__()

    @staticmethod
    def sample(cur_depth, ndepth, depth_inteval_pixel, shape,
               prob_volume_prev=None, depth_values_prev=None):
        """Compute Hmin / Hmax following the PUIP algorithm and perform a
        non-uniform per-pixel depth sampling."""
        # Fallback to uniform sampling when previous-stage information is
        # unavailable (e.g. the first cascade stage).
        if prob_volume_prev is None or depth_values_prev is None:
            cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)
            cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
            new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)

            depth_range_samples = cur_depth_min.unsqueeze(1) + (
                torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype
                             ).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
            return depth_range_samples

        B, H, W = cur_depth.shape

        # Upsample previous-stage prob volume and hypotheses to the current
        # spatial resolution.
        prob_volume_prev_upsampled = F.interpolate(
            prob_volume_prev, size=(H, W), mode='bilinear', align_corners=False)
        depth_values_prev_upsampled = F.interpolate(
            depth_values_prev, size=(H, W), mode='bilinear', align_corners=False)

        # Expand current depth: (B, H, W) -> (B, 1, H, W)
        Hx_expanded = cur_depth.unsqueeze(1)

        # Squared difference w.r.t. the current estimate
        diff_sq = (depth_values_prev_upsampled - Hx_expanded) ** 2

        # Split into above / below masks
        mask_big = depth_values_prev_upsampled >= Hx_expanded
        mask_small = depth_values_prev_upsampled < Hx_expanded

        # Probability-weighted squared sums
        big = (prob_volume_prev_upsampled * diff_sq * mask_big).sum(dim=1)
        small = (prob_volume_prev_upsampled * diff_sq * mask_small).sum(dim=1)

        # Hbig and Hsmall
        H_big = torch.sqrt(big.clamp(min=1e-8))
        H_small = torch.sqrt(small.clamp(min=1e-8))

        # Hmin and Hmax
        H_min = cur_depth - H_small
        H_max = cur_depth + H_big

        # 8-neighborhood depth extrema (vectorized)
        padded_depth = F.pad(cur_depth.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        neighbor_max = F.max_pool2d(padded_depth, kernel_size=3, stride=1, padding=0).squeeze(1)
        neighbor_min = -F.max_pool2d(-padded_depth, kernel_size=3, stride=1, padding=0).squeeze(1)

        # Local positive / negative slopes
        Smax = neighbor_max - cur_depth
        Smin = cur_depth - neighbor_min

        # Clamp non-positive slopes to zero
        Smax = torch.clamp_min(Smax, 0)
        Smin = torch.clamp_min(Smin, 0)

        # Vectorized Nmax / Nmin allocation
        denominator = Smax + Smin
        valid_mask = denominator > 1e-8

        Nmax = torch.zeros_like(Smax)
        Nmin = torch.zeros_like(Smin)
        Nmax[valid_mask] = ndepth * Smax[valid_mask] / denominator[valid_mask]
        Nmin[valid_mask] = ndepth * Smin[valid_mask] / denominator[valid_mask]

        # Boundary handling
        Nmax = torch.where((Smax <= 0) & (Smin > 0), torch.ones_like(Nmax), Nmax)
        Nmin = torch.where((Smin <= 0) & (Smax > 0), torch.ones_like(Nmin), Nmin)
        both_zero = (Smax <= 0) & (Smin <= 0)
        Nmax[both_zero] = ndepth // 2
        Nmin[both_zero] = ndepth - ndepth // 2

        # Round and rebalance to ensure Nmax + Nmin == ndepth
        Nmax_int = torch.round(Nmax).long()
        Nmin_int = torch.round(Nmin).long()
        total = Nmax_int + Nmin_int
        diff = ndepth - total
        Nmax_int = Nmax_int + diff

        # Keep counts within a valid range
        Nmax_int = torch.clamp(Nmax_int, 1, ndepth - 1)
        Nmin_int = ndepth - Nmax_int

        # ---------------- Vectorized sampling ----------------
        indices = torch.arange(ndepth, device=cur_depth.device).float()

        # Per-pixel sampling step
        interval_max = (H_max - cur_depth) / Nmax_int.float()
        interval_min = (cur_depth - H_min) / Nmin_int.float()

        # Sampling masks
        mask_max = indices.reshape(1, 1, 1, -1) < Nmax_int.unsqueeze(-1)
        mask_min = indices.reshape(1, 1, 1, -1) >= (ndepth - Nmin_int).unsqueeze(-1)

        # Samples in the Hmax segment (k = 1, 2, ...)
        k_max = indices.reshape(1, 1, 1, -1) + 1
        samples_max = cur_depth.unsqueeze(-1) + k_max * interval_max.unsqueeze(-1)
        samples_max = torch.where(mask_max, samples_max, torch.nan)

        # Samples in the Hmin segment, including H(x); k counts backwards
        k_min = ndepth - indices.reshape(1, 1, 1, -1)
        samples_min = cur_depth.unsqueeze(-1) - k_min * interval_min.unsqueeze(-1)
        samples_min = torch.where(mask_min, samples_min, torch.nan)

        # Merge the two segments
        depth_range_samples = torch.where(mask_max, samples_max, samples_min)

        # Handle degenerate cases (Nmax == 0 or Nmin == 0)
        zero_max_mask = (Nmax_int == 0).unsqueeze(-1)
        zero_min_mask = (Nmin_int == 0).unsqueeze(-1)
        depth_range_samples = torch.where(zero_max_mask, samples_min, depth_range_samples)
        depth_range_samples = torch.where(zero_min_mask, samples_max, depth_range_samples)

        # Sort samples so that the depth dimension is monotonically increasing
        depth_range_samples = depth_range_samples.reshape(B, H, W, ndepth)
        sorted_indices = torch.argsort(depth_range_samples, dim=-1)

        batch_indices = torch.arange(B, device=cur_depth.device).reshape(B, 1, 1, 1).expand(-1, H, W, ndepth)
        height_indices = torch.arange(H, device=cur_depth.device).reshape(1, H, 1, 1).expand(B, -1, W, ndepth)
        width_indices = torch.arange(W, device=cur_depth.device).reshape(1, 1, W, 1).expand(B, H, -1, ndepth)

        depth_range_samples_sorted = depth_range_samples[
            batch_indices, height_indices, width_indices, sorted_indices
        ]

        # Reshape back to (B, ndepth, H, W)
        depth_range_samples = depth_range_samples_sorted.permute(0, 3, 1, 2)
        return depth_range_samples


# ----------------------------------------------------------------------------
# Functional wrappers kept for backward compatibility.
# ----------------------------------------------------------------------------

# A single shared PUIP instance (the module has no learnable parameters).
_PUIP_SINGLETON = PUIP()


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape,
                                prob_volume_prev=None, depth_values_prev=None):
    """Backward-compatible wrapper that forwards to :meth:`PUIP.sample`."""
    return _PUIP_SINGLETON.sample(
        cur_depth, ndepth, depth_inteval_pixel, shape,
        prob_volume_prev=prob_volume_prev, depth_values_prev=depth_values_prev,
    )


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape, stage,
                            prob_volume_prev=None, depth_values_prev=None):
    """Unified depth-sampling entry point.

    Stage 1 falls back to uniform sampling; later stages dispatch to PUIP.
    """
    if cur_depth.dim() == 2:  # Stage 1: initial depth range
        cur_depth_min = cur_depth[:, 0]
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)

        depth_range_samples = cur_depth_min.unsqueeze(1) + (
            torch.arange(0, ndepth, device=device, dtype=dtype, requires_grad=False
                         ).reshape(1, -1) * new_interval.unsqueeze(1))  # (B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, shape[1], shape[2])  # (B, D, H, W)
    else:  # Stage 2 / 3: use previous-stage information
        depth_range_samples = get_cur_depth_range_samples(
            cur_depth, ndepth, depth_inteval_pixel, shape, prob_volume_prev, depth_values_prev)

    return depth_range_samples


def uncertainty_aware_samples(cur_depth, depth_min, depth_max, exp_var, ndepth, device, dtype, shape):
    """Uncertainty-aware sampling used by UCS-Net (kept for API compatibility)."""
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
