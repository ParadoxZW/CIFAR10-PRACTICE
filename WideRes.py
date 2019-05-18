# this code is work, I've 92% accuracy on test set
from modules import ResBlock, SampleResBlock, Conv2d
from torch import tensor
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class ResGroup(nn.Module):
    """
    resblocks with different input and output size.
    """

    def __init__(self, n, in_channels, in_width, dropout_rate):
        super(ResGroup, self).__init__()
        group = [SampleResBlock(in_channels=in_channels, 
                                in_width=in_width, dropout_rate=dropout_rate)]
        group += [ResBlock(channels=in_channels * 2, dropout_rate=dropout_rate) for _ in range(n - 1)]
        self.group = nn.Sequential(*group)

    def forward(self, x):
        return self.group(x)


class ResGroupIn(nn.Module):
    """
    resblocks with same input and output size.
    """

    def __init__(self, n, in_channels):
        super(ResGroupIn, self).__init__()
        group = [ResBlock(channels=in_channels) for _ in range(n)]
        self.group = nn.Sequential(*group)

    def forward(self, x):
        return self.group(x)


class WideRes(nn.Module):
    """
    wideresnet for cifar10.
    """

    def __init__(self, n=6, k=10):
        super(WideRes, self).__init__()
        self.cin = Conv2d(3, 16*k,
                          kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * k, 10)
        self.resnet = nn.Sequential(ResGroupIn(n=n, in_channels=16 * k),
                                    ResGroup(n=n, in_channels=16 * k, in_width=32, dropout_rate=0.1),
                                    ResGroup(n=n, in_channels=32 * k, in_width=16, dropout_rate=0.2))
        self.GlobalAvgPooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.cin(x)
        x = self.resnet(x)
        x = self.GlobalAvgPooling(x)
        x = self.fc(x.view(x.shape[0], -1))
        # return F.softmax(x, dim=1)
        return x
