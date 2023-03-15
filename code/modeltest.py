"""
modeltest.py

seperate testing procedure of model for CIFAR-10

Reference:
[1] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import sampler
import torch.optim as optim
from tensorboardX import SummaryWriter

import os
import argparse
from tqdm import tqdm

from model import *
from utils import *


# argument parser for running program from command line
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=3407,
                    help='seed for torch')
parser.add_argument('--root_path', type=str, default='./data',
                    help='data_path')
parser.add_argument('--model_path', type=str, default='./best_model.pth',
                    help='model_path')
parser.add_argument('--arc_opt', type=int, default=2,
                    help='2: num_planes=[64,128,256,512], num_blocks=[2,1,1,1];\
                    1: num_planes=[32,64,128,256], num_blocks=[2,2,2,2]')

args = parser.parse_args()

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_dataloaders():
    ## seed torch, np and random
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        np.random.RandomState(worker_seed)
        random.seed(worker_seed)

    ## achieve test data and perform normalization
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=args.root_path,
        train=False,
        download=True,
        transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    return test_loader


def test(model, test_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    for data in tqdm(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs=outputs, labels=labels)
        total_loss += loss.item()
        total_acc += acc

    final_loss = total_loss / len(test_loader)
    final_acc = total_acc / len(test_loader)
    print(f'final loss on test set is {final_loss:.3f}')
    print(f'final acc on test set is {final_acc:.3f}\n')


def main():
    ## prepare dataloaders
    test_loader = prepare_dataloaders()

    ## initialize the model
    model = MyResNet()
    if args.arc_opt == 1:
        model = MyResNet(block=BasicBlock,
                         num_planes=[32, 64, 128, 256],
                         num_blocks=[2, 2, 2, 2])
    else:
        model = MyResNet(block=BasicBlock,
                         num_planes=[64, 128, 256, 512],
                         num_blocks=[2, 1, 1, 1])
    model = model.to(DEVICE)

    ## initialize the loss criterion and optimizer for model
    criterion = nn.CrossEntropyLoss()

    ##  load pretrained best model and test
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(DEVICE)
    test(model, test_loader, criterion)

if __name__ == '__main__':
    main()
