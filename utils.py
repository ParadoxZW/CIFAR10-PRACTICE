'''Some helper functions for PyTorch, including:
    - msr_init: net parameter initialization.
    - time_stamp: generate timstamp
'''
import time
import torch
import torch.nn as nn
import torch.nn.init as init


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
#        elif isinstance(m, nn.BatchNorm2d):
#            init.constant_(m.weight, 1)
#            init.constant_(m.bias, 0)
#        elif isinstance(m, nn.Linear):
#            init.normal_(m.weight, std=1e-5)
#            if m.bias is not None:
#                init.constant_(m.bias, 0)


def time_stamp():
    '''generate timstamp'''
    return time.strftime("%m_%d_%H%M", time.localtime())


def accuracy(predict, target):
    '''compute accuracy'''
    pred_y = torch.max(predict, 1)[1].data.squeeze()
    acc = (pred_y == target).sum().item() / float(target.size(0))
    return acc
