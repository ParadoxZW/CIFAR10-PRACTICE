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
from tqdm import tqdm
from criterion import LabelSmoothing
from utils import accuracy, time_stamp, init_params

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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
# help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
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
augment = transforms.RandomChoice(
    [transforms.RandomAffine(degrees=40, translate=(0.1, 0.1), scale=(0.8, 0.9), shear=0.9),
     transforms.RandomCrop(32, padding=4),
     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
     ]
)

transform_train = transforms.Compose([
    transforms.RandomApply([augment], p=0.7),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# don't forget to change download after download
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = Data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = Data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


def train():
    '''
        training network
        TODO: early stopping
    '''
    log.info('==> Building model...')
    net = FMPNet().double().cuda()
    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if args.resume:
        try:
            log.info("=> loading checkpoint '{}'".format(args.resume))
            # custom method for loading last checkpoint
            ckpt = torch.load(args.resume)
            net.load_state_dict(ckpt['net'])
            start_epoch = ckpt['epoch']
            start_n_iter = ckpt['n_iter']
            # optim.load_state_dict(ckpt['state_dict'])
            log.info("last checkpoint restored")
        except:
            log.info("=> failed to load checkpoint '{}'".format(args.resume))
            return
    # net = torch.nn.DataParallel(net)
    net.train()

    criterion = LabelSmoothing(10, 0.02).cuda()
    optim = torch.optim.Adam(net.parameters(), args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                        milestones=[100, 150], last_epoch=start_epoch - 1)

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter(log_dir='./logs')

    # now we start the main loop
    n_iter = start_n_iter
    for epoch in range(start_epoch, args.epochs):

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(BackgroundGenerator(trainloader)),
                    total=len(trainloader))
        start_time = time.time()
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
            log.info('1')
            prepare_time = start_time - time.time()

            # forward and backward pass
            out = net(X)
            log.info('2')
            loss = criterion(out, target)
            log.info('3')
            optim.zero_grad()
            log.info('4')
            loss.backward()
            log.info('5')
            optim.step()
            log.info('6')

            # udpate tensorboardX
            if n_iter % 50 == 0:
                loss_ = loss.item()
                log.info('7')
                acc = accuracy(out, target)
                log.info('iter: %3d | loss: %6.3f | accuracy: %6.3f'
                         % (i, loss_, acc))
                writer.add_scalar('scalar/loss', loss_, n_iter)
                writer.add_scalar('scalar/train_acc', acc, n_iter)
            n_iter += 1
            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), epoch, args.epochs))
            start_time = time.time()
        # change lr if needed
        lr_scheduler.step()
        log.info('8')
        # maybe do a test pass every x epochs
        x = 50
        if epoch % x == x - 1:
            # bring models to evaluation mode
            net.eval()
            #do some tests
            pbar = tqdm(enumerate(BackgroundGenerator(testloader)),
                        total=len(testloader))
            cnt = 0
            acc = 0
            for i, data in pbar:
                X, target = data
                X = X.double().cuda()
                target = target.cuda()
                out = net(X)
                acc += accuracy(out, target)
                cnt += 1
            acc /= cnt
            log.info('test accuracy: %6.3f' % (acc))
            # save checkpoint if needed
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter
            }
            torch.save(state,
                       args.save_dir + '/' + time_stamp() + '_' +
                       str(epoch) + '_' + str(acc) + '.pkl')
    writer.close()


if __name__ == '__main__':
    train()
