"""
FFE: Fusion Feature Extractor
跨 stage 多尺度特征融合提取器。

将 FeatureNet 输出的三个尺度特征（stage1: C=32, stage2: C=16, stage3: C=8）
按当前 stage 进行通道/尺寸对齐，并通过 CDAM 注意力增强后输出当前 stage
所需的融合特征。
"""

import torch
import torch.nn as nn

from modules.CDAM import CDAM_Block, DWConv


class FFE(nn.Module):
    """Fusion Feature Extractor.

    Parameters
    ----------
    img_h, img_w : int
        参考视角原始图像的高宽。CDAM 内部需要确定 H、W 大小，因此在构建时
        必须传入；推理与训练阶段保持一致即可。
    """

    def __init__(self, img_h=384, img_w=768):
        super(FFE, self).__init__()

        # stage 大小：stage1 = (h/4, w/4)，stage2 = (h/2, w/2)，stage3 = (h, w)
        h1, w1 = img_h // 4, img_w // 4
        h2, w2 = img_h // 2, img_w // 2
        h3, w3 = img_h, img_w

        # ---- stage1 融合分支 (输出通道 32) ----
        self.stage1_fusion = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, dilation=1),                  # 0 身份映射
            DWConv(16, 32, 3, stride=2, padding=1),                       # 1 下采样
            DWConv(8, 32, 5, stride=4, padding=2),                        # 2 下采样
            nn.BatchNorm2d(32),                                           # 3
            nn.ReLU(inplace=True),                                        # 4
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),                  # 5 身份映射
            CDAM_Block(channel=32, h=h1, w=w1),                           # 6 注意力
        )

        # ---- stage2 融合分支 (输出通道 16) ----
        self.stage2_fusion = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, dilation=1),
            DWConv(8, 16, 3, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            CDAM_Block(channel=16, h=h2, w=w2),
        )

        # ---- stage3 融合分支 (输出通道 8) ----
        self.stage3_fusion = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, dilation=1),
            nn.ConvTranspose2d(32, 8, 3, stride=4, padding=1, output_padding=3),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=2, dilation=2),
            CDAM_Block(channel=8, h=h3, w=w3),
        )

    def _fuse(self, branch, feat_main, feat_a, feat_b):
        """通用融合：身份映射(主) + 两路尺度对齐 -> BN -> 身份映射(膨胀) -> CDAM.

        注：保持与原始实现一致，BN 之后**未**接 ReLU（branch[4] 虽然定义为 ReLU，
        但原代码也未调用，此处沿用其语义不做激活）。
        """
        fused = branch[0](feat_main) + branch[1](feat_a) + branch[2](feat_b)
        # branch[3]=BN；branch[5]=身份映射(膨胀)；branch[6]=CDAM
        fused = branch[5](branch[3](fused))
        fused = branch[6](fused)
        return fused

    def forward(self, feat1, feat2, feat3, cur_stage):
        """根据当前 stage 选择对应的融合分支。

        feat1 / feat2 / feat3 分别对应 FeatureNet 输出的 stage1 / stage2 / stage3
        特征（通道 32 / 16 / 8）。
        """
        if cur_stage == 1:
            return self._fuse(self.stage1_fusion, feat1, feat2, feat3)
        elif cur_stage == 2:
            return self._fuse(self.stage2_fusion, feat2, feat3, feat1)
        else:
            return self._fuse(self.stage3_fusion, feat3, feat1, feat2)
