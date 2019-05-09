'''Some helper functions for PyTorch, including:
    - msr_init: net parameter initialization.
    - time_stamp: generate timstamp
'''
import time

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def time_stamp():
    '''generate timstamp'''
    return time.strftime("%m_%d_%H%M", time.localtime()) 

def accuracy(predict, target):
    '''compute accuracy'''
    pred_y = torch.max(predict, 1)[1].data.squeeze()
    acc = (pred_y == target).sum().item() / float(target.size(0))
    return acc