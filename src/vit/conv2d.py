# from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# helper functions
def _standardize_weight(weight, dim, eps):
    # if keepdim=True: keep the dimension of the mean tensor
    mean = torch.mean(weight, dim=dim)
    std = torch.std(weight, dim=dim)
    return (weight - mean) / (std + eps)


# class
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        # dim: 重み全体を集めて「カーネルの幅」の位置ごとに正規化している
        std_w = _standardize_weight(self.weight, dim=[0, 1, 2], eps=1e-5)
        return F.conv2d(
            x, std_w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
