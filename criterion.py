import torch
from torch import nn

class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum').cuda()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        x = x.log()
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size-1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist.requires_grad_(False)
        return self.criterion(x, true_dist)