"""FFE: Fusion Feature Extractor.

Cross-stage multi-scale feature fusion. The three FeatureNet outputs
(stage1: C=32, stage2: C=16, stage3: C=8) are aligned in channel and
spatial resolution for the current stage, then refined by CDAM attention.
"""

import torch.nn as nn

from modules.CDAM import CDAM_Block, DWConv


class FFE(nn.Module):
    """Fusion Feature Extractor.

    Parameters
    ----------
    img_h, img_w : int
        Height and width of the reference image. The internal CDAM blocks
        require fixed H / W at construction time, so training and inference
        must use the same input size.
    """

    def __init__(self, img_h=384, img_w=768):
        super(FFE, self).__init__()

        # Per-stage feature map sizes: stage1 = (h/4, w/4),
        # stage2 = (h/2, w/2), stage3 = (h, w).
        h1, w1 = img_h // 4, img_w // 4
        h2, w2 = img_h // 2, img_w // 2
        h3, w3 = img_h, img_w

        # ---- stage-1 fusion branch (output channels = 32) ----
        self.stage1_fusion = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, dilation=1),                  # 0 identity
            DWConv(16, 32, 3, stride=2, padding=1),                       # 1 downsample x2
            DWConv(8, 32, 5, stride=4, padding=2),                        # 2 downsample x4
            nn.BatchNorm2d(32),                                           # 3
            nn.ReLU(inplace=True),                                        # 4 (unused; kept for parity)
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),                  # 5 dilated identity
            CDAM_Block(channel=32, h=h1, w=w1),                           # 6 attention
        )

        # ---- stage-2 fusion branch (output channels = 16) ----
        self.stage2_fusion = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, dilation=1),
            DWConv(8, 16, 3, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            CDAM_Block(channel=16, h=h2, w=w2),
        )

        # ---- stage-3 fusion branch (output channels = 8) ----
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
        """Shared fusion logic: identity(primary) + two scale-aligned paths
        -> BN -> dilated identity -> CDAM.

        Note: ``branch[4]`` (ReLU) is defined for parity with the original
        implementation but is intentionally not invoked, matching the
        reference code semantics.
        """
        fused = branch[0](feat_main) + branch[1](feat_a) + branch[2](feat_b)
        # branch[3] = BN, branch[5] = dilated identity, branch[6] = CDAM
        fused = branch[5](branch[3](fused))
        fused = branch[6](fused)
        return fused

    def forward(self, feat1, feat2, feat3, cur_stage):
        """Select the fusion branch for ``cur_stage``.

        ``feat1`` / ``feat2`` / ``feat3`` correspond to FeatureNet outputs
        ``stage1`` / ``stage2`` / ``stage3`` (channels 32 / 16 / 8).
        """
        if cur_stage == 1:
            return self._fuse(self.stage1_fusion, feat1, feat2, feat3)
        elif cur_stage == 2:
            return self._fuse(self.stage2_fusion, feat2, feat3, feat1)
        else:
            return self._fuse(self.stage3_fusion, feat3, feat1, feat2)
