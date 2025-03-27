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
