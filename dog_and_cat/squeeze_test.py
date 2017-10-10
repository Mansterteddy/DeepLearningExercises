'''Train Kaggle Dog and Cat with PyTorch.'''
from __future__ import print_function

# torch package
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

# torchvision package
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# other package
import sys
import os
import os.path
import random
import collections
import shutil
import time
import glob
import csv
import numpy as np
import argparse
from PIL import Image

# self-defined package
from utils import progress_bar
from models import *

#paths
data_path = './dog_cat_kaggle/catdog/'
train_path = './dog_cat_kaggle/catdog/train/'
val_path = './dog_cat_kaggle/catdog/val/'

#heyper parameters

use_cuda = torch.cuda.is_available() #check cuda

best_acc = 0  # best test accuracy

resume = True
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

pretrained_model = False
pretrained_network = ['squeezenet1_0', 'inception_v3', 'vgg19', 'resnet101']

set_batch_size = 256
set_lr = 0.001 #learning rate

# Load data
print('=> Preparing data..')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


train_loader = data.DataLoader(datasets.ImageFolder(train_path, transforms.Compose([transforms.RandomSizedCrop(299),
                                                                                    transforms.RandomHorizontalFlip(),
                                                                                    transforms.ToTensor(),
                                                                                    normalize,])),
                               batch_size=set_batch_size,
                               shuffle=True,
                               num_workers=2,)

val_loader = data.DataLoader(datasets.ImageFolder(val_path, transforms.Compose([transforms.Scale(256),
                                                                                transforms.CenterCrop(299),
                                                                                transforms.ToTensor(),
                                                                                normalize,])),
                             batch_size=set_batch_size,
                             shuffle=True,
                             num_workers=2,)


# Model
if pretrained_model:
    print("==> using pretrained model '{}'".format(pretrained_network[0]))
    net = models.__dict__[pretrained_network[0]](pretrained=True)
    #Don't update non-classifier learned features in the pretrained networks
    #for param in net.parameters():
    #    param.requires_grad = False
    #Replace the last fully-connected layer
    #Parameters of newly constructed module have requires_grad=True by default
    #Final dense layer needs to replaced with previous out channels, and number of classes
    #in this case -- resnet 101 - it's 2048 with two classes (cats and dogs)
else:
    if resume:
        #Load checkpoint
        print("=> Resuming from checkpoint")
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_squeezenet.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    else:
        print("=> Building model")
        # net = LeNet()
        # net = GoogLeNet()
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        net = squeezenet1_1(pretrained=False, num_classes=3)

# If cuda is available, using cuda net
if use_cuda:
    net.cuda()

#print("net: ", net)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimization method
if pretrained_model:
    optimizer = optim.SGD(net.parameters(), lr=set_lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=set_lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        #print("outputs: ", outputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = nn.functional.nll_loss(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            #'net': net.module if use_cuda else net,
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_squeezenet.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
