# just for fun, give channel some meanings about relations
# between positions.
from modules import *
from torch import tensor
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn



class Group(nn.Module):
    """
    resblocks with same input and output size.
    """

    def __init__(self, n, in_channels, in_width):
        super(Group, self).__init__()
        branch1 = [SeResBlock(channels=in_channels) for _ in range(n)]
        self.branch1 = nn.Sequential(*group)
        branch2 = [Channel_Attn(id_dim=in_channels, N=in_width**2) for _ in range(n)]
        self.branch2 = nn.Sequential(*group)

    def forward(self, x):
        return torch.cat((self.branch1(x), self.branch2(x)), 1)


class Chanet(nn.Module):
    """
    wideresnet for cifar10.
    """

    def __init__(self, n=6, k=10):
        super(Chanet, self).__init__()
        self.cin = Conv2d(3, 16 * k,
                          kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * k, 10)
        self.resnet = nn.Sequential(Group(n=n, in_channels=16 * k, in_width=32),
                                    nn.MaxPool2d(2, stride=2, padding=0),
                                    Group(n=n, in_channels=32 * k, in_width=16),
                                    nn.MaxPool2d(2, stride=2, padding=0),
                                    Group(n=n, in_channels=64 * k, in_width=8))
        self.GlobalAvgPooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.cin(x)
        x = self.resnet(x)
        x = self.GlobalAvgPooling(x)
        x = self.fc(x.view(x.shape[0], -1))
        # return F.softmax(x, dim=1)
        return x
