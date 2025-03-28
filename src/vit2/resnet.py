from collections import OrderedDict
# from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d import StdConv2d


# helper functions
def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    return StdConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        groups=groups,
    )


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return StdConv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
    )


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, in_channels, out_channels=None, mid_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or out_channels // 4

        self.gn1 = nn.GroupNorm(32, mid_channels, eps=1e-6)
        self.conv1 = conv1x1(in_channels, mid_channels, bias=False)

        self.gn2 = nn.GroupNorm(32, mid_channels, eps=1e-6)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride, bias=False)

        self.gn3 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv3 = conv1x1(mid_channels, out_channels, bias=False)

        # 活性化関数
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            """
            stride != 1: 画像のH, Wが変化する
            in_channels != out_channels: チャネル数が変化する
            """
            # チャネル数を合わせる（downsample属性を追加している）
            self.downsample = conv1x1(in_channels, out_channels, stride, bias=False)
            # 正規化して尺度を合わせる
            self.gn_proj = nn.GroupNorm(out_channels, out_channels)

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, "downsample"):  # ここでチャネル数を合わせている
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)

        return y
