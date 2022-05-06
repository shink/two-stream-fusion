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

    def __init__(self, class_num):
        super(TwoStreamFusion, self).__init__()

        self.spatial_net = SpatialStream()
        self.temporal_net = TemporalStream()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=class_num)

    def forward(self, x_spatial, x_temporal):
        spatial_out = self.spatial_net(x_spatial)
        temporal_out = self.temporal_net(x_temporal)

        # convolutional fusion
        x = self.conv(spatial_out + temporal_out)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class SpatialStream(nn.Module):
    """
    The spatial stream network
    """

    def __init__(self, pretrained=True):
        super(SpatialStream, self).__init__()

        self.spatial_stream = models.resnet34(pretrained=pretrained)
        # remove the avgpool layer and fc layer
        self.spatial_stream = nn.Sequential(*list(self.spatial_stream.children())[:-2])

    def forward(self, x):
        spatial_out = self.spatial_stream(x)
        return spatial_out


class TemporalStream(nn.Module):
    """
    The temporal stream network
    """

    def __init__(self):
        super(TemporalStream, self).__init__()

        self.temporal_stream = models.resnet34()
        # remove the avgpool layer and fc layer
        self.temporal_stream = nn.Sequential(*list(self.temporal_stream.children())[:-2])

    def forward(self, x):
        temporal_out = self.temporal_stream(x)
        return temporal_out
