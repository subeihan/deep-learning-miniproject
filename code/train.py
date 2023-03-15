"""
train.py

training procedure of model for CIFAR-10

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
parser.add_argument('--save_path', type=str, default='./model',
                    help='save_path')
parser.add_argument('--save_every_e', type=int, default=10,
                    help='save model per save_every_e epoch')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch_size per gpu')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='maximum epoch number to train')
parser.add_argument('--arc_opt', type=int, default=2,
                    help='2: num_planes=[64,128,256,512], num_blocks=[2,1,1,1];\
                    1: num_planes=[32,64,128,256], num_blocks=[2,2,2,2]')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--l2_reg', default=False, action='store_true')
parser.add_argument('--no-l2_reg', dest='l2_reg', action='store_false')
parser.add_argument('--adjust_lr', default=False, action='store_true')
parser.add_argument('--no-adjust_lr', dest='adjust_lr', action='store_false')


args = parser.parse_args()


# We divide the overall train_set into train and val parts, with 80%:20% ratio
NUM_TRAIN = 40000

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

    ## perform necessary data augmentation on train_set and val_set
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        # transforms.RandomGrayscale([0.1]), 
        # transforms.RandomApply(torch.nn.ModuleList([
        #     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5))
        #     ]), p=0.3), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    ## achieves data from torchvision, divide train into train-valid
    train_set = torchvision.datasets.CIFAR10(
        root=args.root_path,
        train=True,
        download=True,
        transform=transform_train
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
        worker_init_fn=seed_worker
    )

    val_set = torchvision.datasets.CIFAR10(
        root=args.root_path,
        train=True,
        download=True,
        transform=transform_test
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=100,
        shuffle=False,
        num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)),
        worker_init_fn=seed_worker
    )

    ## achieve test data and perform normalization
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

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, optimizer, criterion):
    ## visualize tool
    writer = SummaryWriter(args.save_path + '/log')

    max_iterations = args.max_epoch * len(train_loader)
    iteration = 0

    ## record the best(lowest) loss
    cur_best_loss = 10000

    for ep in tqdm(range(args.max_epoch)):
        ## training session
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        for i, data in tqdm((enumerate(train_loader))):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs=outputs, labels=labels)
            total_train_loss += loss.item()
            total_train_acc += acc

            ## adjust learning rate (cosine scheduler) if asjust_lr set to be True
            lr_ = args.learning_rate
            if args.adjust_lr:
                lr_ = args.learning_rate * (1.0 - iteration / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iteration)
            writer.add_scalar('info/train_loss', loss, iteration)
            writer.add_scalar('info/train_acc', accuracy(outputs=outputs, labels=labels), iteration)

            iteration += 1

        per_train_loss = total_train_loss / len(train_loader)
        per_train_acc = total_train_acc / len(train_loader)

        ## eval session
        with torch.no_grad():
            model.eval()
            total_val_loss = 0.0
            total_val_acc = 0.0
            for data in tqdm(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = accuracy(outputs=outputs, labels=labels)
                total_val_loss += loss.item()
                total_val_acc += acc

            per_val_loss = total_val_loss / len(val_loader)
            per_val_acc = total_val_acc / len(val_loader)

            writer.add_scalar('info/val_loss', per_val_loss, ep)
            writer.add_scalar('info/val_acc', per_val_acc, ep)

            if per_val_loss <= cur_best_loss:
                cur_best_loss = per_val_loss
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))

        if ((ep + 1) % args.save_every_e == 0):
            torch.save(model.state_dict(), os.path.join(args.save_path, f'iter_{ep}.pth'))


        print(f'epoch: {ep + 1:03}')
        print(f'\ttrain Loss: {per_train_loss:.3f} | train Acc: {per_train_acc * 100:.2f}%')
        print(f'\t val. Loss: {per_val_loss:.3f} |  val. Acc: {per_val_acc * 100:.2f}%')

    print('training finished')
    writer.close()


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
    train_loader, val_loader, test_loader = prepare_dataloaders()

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

    num_trainable_params = count_parameters(model)
    print('The num of total trainable parameter in our model is', num_trainable_params)

    ## initialize the loss criterion and optimizer for model
    criterion = nn.CrossEntropyLoss()
    weight_decay = 1e-6 if args.l2_reg else 0
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=weight_decay)

    ## train model
    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          optimizer=optimizer,
          criterion=criterion)

    ##  load pretrained best model and test
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'best_model.pth')))
    model = model.to(DEVICE)
    test(model, test_loader, criterion)


if __name__ == '__main__':
    main()