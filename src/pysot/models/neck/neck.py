# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

import torch
import torch.nn as nn


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        self.downsample = nn.ModuleList([
            AdjustLayer(
                in_channels[i],
                out_channels[i],
                center_size
            )
            for i in range(self.num)
        ])

    def forward(self, features: List[torch.Tensor]):
        out = []
        # TODO: Switch for a zip or something if https://github.com/pytorch/pytorch/issues/16123 is resolved.
        index = 0
        for downsample in self.downsample:
            out.append(downsample(features[index]))
            index += 1
        return out
