import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.module import (
    FeatureNet, CostRegNet, RefineNet, depth_regression,
)
from modules.warping import homo_warping, rpc_warping, rpc_warping_enisum
from modules.PUIP import get_depth_range_samples
from modules.FFE import FFE


Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization,
                geo_model, use_qc=False):
        if not use_qc:
            proj_matrices = torch.unbind(proj_matrices, 1)

        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shape[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homography, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        if geo_model == "rpc" and not use_qc:
            b_num, f_num, img_h, img_w = ref_feature.shape
            coef = torch.ones((b_num, img_h * img_w * num_depth, 20), dtype=torch.double).cuda()
        else:
            coef = None

        for src_fea, src_proj in zip(src_features, src_projs):
            if geo_model == "rpc" and not use_qc:
                warped_volume = rpc_warping(src_fea, src_proj, ref_proj, depth_values, coef)
            elif geo_model == "rpc" and use_qc:
                warped_volume = rpc_warping_enisum(src_fea, src_proj, ref_proj, depth_values)
            else:
                warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume

        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(
                F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(
                prob_volume,
                depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float),
            ).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return {
            "depth": depth,
            "photometric_confidence": photometric_confidence,
            "prob_volume": prob_volume,
        }


class CascadeMVSNet(nn.Module):
    def __init__(self, geo_model, refine=False, min_interval=2.5,
                 ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1],
                 share_cr=False, grad_method="detach", arch_mode="fpn",
                 cr_base_chs=[8, 8, 8], use_qc=False,
                 img_h=384, img_w=768):
        super(CascadeMVSNet, self).__init__()
        self.geo_model = geo_model
        assert self.geo_model in ["rpc", "pinhole"]
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.min_interval = min_interval
        self.use_qc = use_qc

        if self.use_qc and geo_model == "rpc":
            print("Quaternary Cubic Form is used for RPC warping")

        print("**********ndepths:{}, depth_intervals_ratio:{}, grad:{}, chs:{}************".format(
            ndepths, depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1": {"scale": 4.0},
            "stage2": {"scale": 2.0},
            "stage3": {"scale": 1.0},
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage,
                                  arch_mode=self.arch_mode)
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([
                CostRegNet(in_channels=self.feature.out_channels[i], base_channels=self.cr_base_chs[i])
                for i in range(self.num_stage)
            ])
        if self.refine:
            self.refine_network = RefineNet()
        self.DepthNet = DepthNet()

        # Fusion Feature Extractor，集成 CDAM 注意力的多尺度特征融合模块
        self.ffe = FFE(img_h=img_h, img_w=img_w)

    def forward(self, imgs, proj_matrices, depth_values):

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        outputs = {}
        depth, cur_depth = None, None

        # 保存前一阶段的概率体与深度采样值（供 PUIP 使用）
        prev_depth_samples = None
        prev_prob_volume = None

        # 把每个视角的多尺度特征拆解出来，便于后续按 stage 融合
        features_stage1, features_stage2, features_stage3 = [], [], []
        for nview_idx in range(imgs.size(1)):
            current_view_stage1 = torch.stack([feat["stage1"] for feat in features])  # [B, C1, H1, W1]
            current_view_stage2 = torch.stack([feat["stage2"] for feat in features])
            current_view_stage3 = torch.stack([feat["stage3"] for feat in features])
            features_stage1.append(current_view_stage1[nview_idx])
            features_stage2.append(current_view_stage2[nview_idx])
            features_stage3.append(current_view_stage3[nview_idx])

        for stage_idx in range(self.num_stage):
            cur_stage = stage_idx + 1

            # 通过 FFE 完成多尺度跨 stage 特征融合
            features_stage = []
            for nview_idx in range(imgs.size(1)):
                feat1 = features_stage1[nview_idx]
                feat2 = features_stage2[nview_idx]
                feat3 = features_stage3[nview_idx]
                fused = self.ffe(feat1, feat2, feat3, cur_stage)
                features_stage.append(fused)

            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(
                    cur_depth.unsqueeze(1),
                    [img.shape[2], img.shape[3]],
                    mode='bilinear',
                    align_corners=Align_Corners_Range,
                ).squeeze(1)
            else:
                cur_depth = depth_values

            # PUIP：基于前一阶段的概率体进行像素级非均匀采样
            depth_range_samples = get_depth_range_samples(
                cur_depth=cur_depth,
                ndepth=self.ndepths[stage_idx],
                depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * self.min_interval,
                dtype=img[0].dtype,
                device=img[0].device,
                shape=[img.shape[0], img.shape[2], img.shape[3]],
                stage=cur_stage,
                prob_volume_prev=prev_prob_volume,
                depth_values_prev=prev_depth_samples,
            )

            outputs_stage = self.DepthNet(
                features_stage,
                proj_matrices_stage,
                depth_values=F.interpolate(
                    depth_range_samples.unsqueeze(1),
                    [self.ndepths[stage_idx],
                     img.shape[2] // int(stage_scale),
                     img.shape[3] // int(stage_scale)],
                    mode='trilinear',
                    align_corners=Align_Corners_Range,
                ).squeeze(1),
                num_depth=self.ndepths[stage_idx],
                cost_regularization=(self.cost_regularization
                                     if self.share_cr else self.cost_regularization[stage_idx]),
                geo_model=self.geo_model,
                use_qc=self.use_qc,
            )

            depth = outputs_stage['depth']
            # 保存当前阶段的结果，供下一阶段 PUIP 使用
            prev_prob_volume = outputs_stage['prob_volume'].detach().clone()
            prev_depth_samples = depth_range_samples.detach().clone()

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs
