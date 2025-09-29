# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision.transforms as transforms

from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader

import timm
from timm.scheduler.cosine_lr import CosineLRScheduler

import os
import argparse
import csv
import time

from utils import progress_bar
from pcam_dataset import PCamDataset  # あなたのPCamDatasetクラス

# parsers
parser = argparse.ArgumentParser(description='PyTorch Medical DNN Training (PCam)')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training')
parser.add_argument('--net', default='vit_base_16')
parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--size', type=int, default=224)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--num_ops', type=int, default=2)
parser.add_argument('--magnitude', type=int, default=14)
args = parser.parse_args()

bs = args.bs
size = args.size
if args.net in ["inceptionv3"]:
    size = 299

use_amp = not args.noamp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# =======================
# データセット・ローダー
# =======================
print('==> Preparing PCam dataset..')

train_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

eval_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = PCamDataset(
    '/home/gotou/Medical/b4us/pcamdata/camelyonpatch_level_2_split_train_x.h5',
    '/home/gotou/Medical/b4us/pcamdata/camelyonpatch_level_2_split_train_y.h5',
    transform=train_transform
)
val_dataset = PCamDataset(
    '/home/gotou/Medical/b4us/pcamdata/valid_x_uncompressed.h5',
    '/home/gotou/Medical/b4us/pcamdata/valid_y_uncompressed.h5',
    transform=eval_transform
)

trainloader = DataLoader(train_dataset, batch_size=bs, sampler=ImbalancedDatasetSampler(train_dataset), num_workers=4)
testloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

nb_classes = 2  # PCamは2クラス

# =======================
# モデル構築
# =======================
print('==> Building model..')
net = timm.create_model(args.net, pretrained=True, num_classes=nb_classes)

if 'cuda' in device:
    print("using data parallel")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

scheduler = CosineLRScheduler(
    optimizer,
    t_initial=args.n_epochs,
    lr_min=args.lr * 0.05,
    warmup_t=int(0.05 * args.n_epochs),
    warmup_lr_init=args.lr * 0.05,
    warmup_prefix=True
)

# =======================
# トレーニング関数
# =======================
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                     (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

# =======================
# 評価関数
# =======================
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()
        }
        if not os.path.isdir('checkpoint_medical'):
            os.mkdir('checkpoint_medical')
        torch.save(state, f'./checkpoint_medical/pcam-{args.net}-bs{args.bs}-{args.opt}-lr{args.lr}-randaug{args.num_ops}-{args.magnitude}ckpt.t7')
        best_acc = acc

    if not os.path.isdir('log_medical'):
        os.makedirs('log_medical')
    content = time.ctime() + f' Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {acc:.5f}'
    print(content)
    with open(f'log_medical/log_pcam_{args.net}_bs{args.bs}_{args.opt}_lr{args.lr}_randaug{args.num_ops}_{args.magnitude}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

# =======================
# 実行ループ
# =======================
list_loss = []
list_acc = []

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    scheduler.step(epoch+1)

    list_loss.append(val_loss)
    list_acc.append(acc)

    with open(f'log_medical/log_pcam_{args.net}_bs{args.bs}_{args.opt}_lr{args.lr}_randaug{args.num_ops}_{args.magnitude}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)
    print([round(a, 3) for a in list_acc])
