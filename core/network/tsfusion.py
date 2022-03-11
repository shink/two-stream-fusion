# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: tsfusion.py
@date: 2022/3/10
@description: 
"""

import torch.nn as nn
import torchvision.models as models


class TwoStreamFusion(nn.Module):
    """
    The two-stream network
    """

    __class_num: int
    __spatial_stream: nn.Module
    __temporal_stream: nn.Module

    def __init__(self, class_num):
        super(TwoStreamFusion, self).__init__()

        self.__class_num = class_num

        self.__spatial_stream = SpatialStream(self.__class_num)
        self.__temporal_stream = TemporalStream(self.__class_num)

    def forward(self, x_spatial, x_temporal):
        spatial_out = self.__spatial_stream(x_spatial)
        temporal_out = self.__temporal_stream(x_temporal)
        return spatial_out + temporal_out


class SpatialStream(nn.Module):
    """
    The spatial stream network
    """

    def __init__(self, class_num):
        super(SpatialStream, self).__init__()

        self.spatial_stream = models.resnet34(pretrained=True)
        self.spatial_stream.fc = nn.Linear(in_features=2048, out_features=class_num)

    def forward(self, x):
        spatial_out = self.spatial_stream(x)
        return spatial_out


class TemporalStream(nn.Module):
    """
    The temporal stream network
    """

    def __init__(self, class_num):
        super(TemporalStream, self).__init__()

        self.temporal_stream = models.resnet34(pretrained=False)
        self.temporal_stream.fc = nn.Linear(in_features=2048, out_features=class_num)

    def forward(self, x):
        temporal_out = self.temporal_stream(x)
        return temporal_out
