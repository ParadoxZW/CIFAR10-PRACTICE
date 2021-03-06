'''
configurations and schedule for network training
this implementation includes some kind fancy tools,
like prefetch_generator, tqdm and tensorboardx.
I also use logging to print information into log file
rather than print function.
'''
import argparse
import os
import time
import logging

import numpy as np
import torch
import torch.utils.data as Data
import torchvision
from torchvision import models
from torchvision import transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from FMPNet import FMPNet
from ResNet import ResNet
from tqdm import tqdm
from criterion import LabelSmoothing
from utils import accuracy, time_stamp, init_params
from autoaugment import CIFAR10Policy
from cutout import Cutout

logging.basicConfig(
    filename='logs/train.log',
    format='%(levelname)-10s %(asctime)s %(message)s',
    level=logging.INFO
)
log = logging.getLogger('train')

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser(description="Train a network for cifar10")
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) +
#                     ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# parser.add_argument('--print-freq', '-p', default=50, type=int,
#                     metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--data-dir', dest='data_dir',
                    help='The directory of data',
                    default='./data', type=str)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
args = parser.parse_args()

log.info('start training.')
# Data
log.info('==> Preparing data..')
# augment = transforms.RandomChoice(
#     [transforms.RandomAffine(degrees=2),
#      transforms.RandomCrop(32, padding=4)
#      ]
# )
# augment = transforms.RandomChoice(
#     [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 0.9), shear=0.9),
#      transforms.RandomCrop(32, padding=4)
#      transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
#      ]
# )

# transform_train = transforms.Compose([
#     transforms.RandomApply([augment], p=0.5),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
    transforms.RandomHorizontalFlip(), CIFAR10Policy(),
    transforms.ToTensor(),
    # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
    Cutout(n_holes=1, length=16),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# don't forget to change download after download
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = Data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test, drop_last=True)
testloader = Data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)


def train():
    '''
        training network
        TODO: early stopping
    '''
    log.info('==> Building model...')
    net = ResNet().double().cuda()
    # log.info(net)
    net = torch.nn.DataParallel(net)
    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if args.resume:
        log.info("=> loading checkpoint '{}'".format(args.resume))
        # custom method for loading last checkpoint
        ckpt = torch.load(args.resume)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        # optim.load_state_dict(ckpt['state_dict'])
        log.info("last checkpoint restored")
    else:
        init_params(net)

    # criterion = LabelSmoothing(10, 0.02).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=args.momentum)
    # optim.param_groups[0]['initial_lr'] = 0.001
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
    #                                                     milestones=[80, 120, 160], last_epoch=start_epoch - 1)

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter(log_dir='./logs')

    # now we start the main loop
    n_iter = start_n_iter
    loss_ = 1
    test_acc = 0
    flag = True
    for epoch in range(start_epoch, args.epochs):

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(BackgroundGenerator(trainloader)),
                    total=len(trainloader))
        start_time = time.time()
        # if loss_ < 0.1:
        net.train()
        if epoch == 2:
            # flag = False
            for param in optim.param_groups:
                param['lr'] *= 10
        if epoch == 90:  # about 35k iterations
            for param in optim.param_groups:
                param['lr'] *= 0.1
        if epoch == 135: # about 50k iterations
            for param in optim.param_groups:
                param['lr'] *= 0.1
        log.info('start epoch: ' + str(epoch) +
                 ' |current lr {:.5e}'.format(optim.param_groups[0]['lr']))
        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            X, target = data
            X = X.double().cuda()
            target = target.cuda()

            # It's very good practice to keep track of preparation time and
            # computation time using tqdm to find any issues in your dataloader
            prepare_time = time.time() - start_time

            # forward and backward pass
            out = net(X)
            loss = criterion(out, target)
            loss_ = loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()

            # udpate tensorboardX
            if n_iter % 50 == 0:
                acc = accuracy(out, target)
                log.info('iter: %3d | loss: %6.3f | accuracy: %6.3f'
                         % (n_iter, loss_, acc))
                writer.add_scalars('loss', {'train': loss_}, n_iter)
                writer.add_scalars('acc', {'train': acc}, n_iter)
            n_iter += 1
            # compute computation time and *compute_efficiency*
            process_time = time.time() - start_time - prepare_time
            pbar.set_description("%2.1f|%2.f|l:%6.3f|ep%3d" % (
                prepare_time, process_time, loss_, epoch))
            start_time = time.time()
        # change lr if needed
        # lr_scheduler.step()
        x = 5
        # save checkpoint if needed
        if epoch % (2*x) == (2*x) - 1:
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter
            }
            torch.save(state,
                       args.save_dir + '/' + time_stamp() + '_' +
                       str(epoch) + '_' + '%.4f'%(test_acc) + '.pkl')
        # maybe do a test pass every x epochs
        if epoch % x == x - 1:
            with torch.no_grads():
                # bring models to evaluation mode
                net.eval()
                #do some tests
                pbar = tqdm(enumerate(BackgroundGenerator(testloader)),
                            total=len(testloader))
                cnt = 0
                test_acc = 0
                for i, data in pbar:
                    X, target = data
                    X = X.double().cuda()
                    target = target.cuda()
                    out = net(X)
                    test_loss = criterion(out, target)
                    test_loss_ = loss.item()
                    test_acc += accuracy(out, target)
                    cnt += 1
                test_acc /= cnt
                log.info('test accuracy: %6.3f' % (test_acc))
                writer.add_scalars('loss', {'test': test_loss_}, n_iter)
                writer.add_scalars('acc', {'test': test_acc}, n_iter)
    writer.close()


if __name__ == '__main__':
    train()
