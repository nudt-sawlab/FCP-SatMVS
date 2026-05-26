"""CDAM: Cross-Dimensional Attention Module.

Attention is modeled along the H and W axes independently via :class:`LAM`,
then combined by an outer product. A softmax produces a per-pixel attention
map that is multiplied element-wise with the input feature.
"""

import torch
import torch.nn as nn


class DWConv(nn.Module):
    """Depth-wise separable 2D convolution."""

    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0):
        super().__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=ksize, stride=stride,
                               padding=(ksize - 1) // 2, groups=in_channels)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, groups=1)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class LAM(nn.Module):
    """Linear attention along a single spatial axis."""

    def __init__(self, channel, input_size):
        super(LAM, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1, input_size))
        self.dense = nn.Linear(input_size, 1)
        self.conv = DWConv(channel, channel, 3, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        x1 = self.GAP(x)        # [B, C, 1, input_size]
        x1 = x1.squeeze(2)      # [B, C, input_size]
        x1 = self.dense(x1)     # [B, C, 1]
        x1 = x1.unsqueeze(-1)   # [B, C, 1, 1]
        x2 = self.conv(x)       # [B, C, H, W]
        return x1.expand_as(x2) * x2


class CDAM_Block(nn.Module):
    """Cross-Dimensional Attention Module.

    Builds two LAM branches, one along H and one along W, fuses them via an
    outer product, applies softmax, and multiplies the resulting
    ``[B, C, h, w]`` attention map with the input feature.
    """

    def __init__(self, channel, h, w, reduction=16):
        super(CDAM_Block, self).__init__()
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
        self.ha_h = LAM(channel, h)
        self.ha_w = LAM(channel, w)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_h = self.avg_pool_x(x)
        x_h = self.ha_h(x_h)
        x_w = self.avg_pool_y(x)
        x_w = self.ha_w(x_w)
        x_h_expanded = x_h.expand(-1, -1, -1, self.w)
        x_w_expanded = x_w.expand(-1, -1, self.h, -1)
        attention = x_h_expanded * x_w_expanded
        attention = self.softmax(attention)
        return attention * x


if __name__ == '__main__':
    net = CDAM_Block(channel=32, h=64, w=32)
    x = torch.randn(2, 32, 64, 32)
    print(net(x).shape)
