from modules import FMPBlock, LayerNorm
from torch import tensor
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

class DropoutFMP(nn.Module):
    "fractional max pooling with dropout"
    def __init__(self, size, out_channels, dropout=0):
        super(DropoutFMP, self).__init__()
        self.norm = LayerNorm(features=size)
        self.dropout = nn.Dropout(dropout)
        self.fmp = FMPBlock(size[0], out_channels)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fmp(x)
        x = self.dropout(x)
        return x

class FMPNet(nn.Module):
    "implemnet of a cnn network with fractional max pooling"
    def __init__(self):
        super(FMPNet, self).__init__()
        n = 160
        m = 512
        self.input = DropoutFMP((3, 32, 32), n)
        layers = []
        h = 25  # height
        k = 1   # times of 160 channels
        while h >= 2:
            ne = DropoutFMP((n*k, h, h), n*(k+1), 0.045*k)
            k += 1
            h = int(0.8 * h)
            layers.append(ne)
        self.layers = nn.Sequential(*layers)
        self.l1 = nn.Linear(n*11, m)
        self.l2 = nn.Linear(m, 10)
    
    def forward(self, x):
        x = self.input(x)
        x = self.layers(x)
        b = x.size()[0]
        return F.softmax(self.l2(self.l1(x.view(b, -1))))

if __name__ == '__main__':
    net = FMPNet()
    print(net)