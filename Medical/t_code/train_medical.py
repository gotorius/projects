# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

from torchsampler import ImbalancedDatasetSampler

import timm
from timm.scheduler.cosine_lr import CosineLRScheduler

import os
import argparse
import csv
import time

from utils import progress_bar

# parsers
parser = argparse.ArgumentParser(description='PyTorch Medical DNN Training')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--dataset', default="melanoma")
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--net', default='vit_base_16')
parser.add_argument('--bs', type=int, default='256')
parser.add_argument('--size', type=int, default="224")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--num_ops', type=int, default='2')
parser.add_argument('--magnitude', type=int, default='14')
args = parser.parse_args()

bs = int(args.bs)
size = int(args.size)
if args.net in ["inceptionv3"]:
    size = 299

use_amp = bool(~args.noamp)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# Prepare dataset
if args.dataset == 'melanoma':
    print('melanoma used')
    data_dir = './MedImages/ISIC2018_ImageFolder/'
elif args.dataset == 'dermmel':
    print('dermmel used')
    data_dir = './MedImages/DermMel/'
elif args.dataset == 'oct':
    print('oct used')
    data_dir = './MedImages/CellData/OCT/'
elif args.dataset == 'chestx':
    print('chestx used')
    data_dir = './MedImages/CellData/chest_xray/'
elif args.dataset == 'pcam':
    print('pcam used')
    data_dir = './MedImages/PCam_ImageFolder/'
else:
    raise ValueError("invalid dataset")

if args.dataset == 'dermmel':
    trainset = torchvision.datasets.ImageFolder(root = data_dir + 'train_sep', transform=transform_train)
else:
    trainset = torchvision.datasets.ImageFolder(root = data_dir + 'train', transform=transform_train)
nb_classes = len(trainset.classes)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=ImbalancedDatasetSampler(trainset), num_workers=8)

testset = torchvision.datasets.ImageFolder(root = data_dir + 'test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

# Model factory..
print('==> Building model..')
if args.net=='res50':
    print('resnet50 used')
    net = timm.create_model('resnet50', pretrained=True, num_classes=nb_classes)
    # 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
elif args.net=='res18':
    print('resnet18 used')
    net = timm.create_model('resnet18', pretrained=True, num_classes=nb_classes)
elif args.net=='res101':
    print('resnet101 used')
    net = timm.create_model('resnet101', pretrained=True, num_classes=nb_classes)
elif args.net=='vgg16':
    print('vgg16 used')
    net = timm.create_model('vgg16', pretrained=True, num_classes=nb_classes)
elif args.net=='vgg19':
    print('vgg19 used')
    net = timm.create_model('vgg19', pretrained=True, num_classes=nb_classes)
elif args.net=='inceptionv3':
    print('inceptionv3 used')
    net = timm.create_model('inception_v3', pretrained=True, num_classes=nb_classes)
elif args.net=='densenet121':
    print('densenet121 used')
    net = timm.create_model('densenet121', pretrained=True, num_classes=nb_classes)
elif args.net=='densenet201':
    print('densenet201 used')
    net = timm.create_model('densenet201', pretrained=True, num_classes=nb_classes)
elif args.net=='efficientnet_b1':
    print('efficientnet_b1 used')
    net = timm.create_model('efficientnet_b1', pretrained=True, num_classes=nb_classes)
elif args.net=='mobilenetv2':
    print('mobilenetv2 used')
    net = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=nb_classes)
elif args.net=='mobilenetv3':
    print('mobilenetv3 used')
    net = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=nb_classes)
elif args.net=='vit_base_8':
    print('vit base 8 used')
    net = timm.create_model("vit_base_patch8_224", pretrained=True, num_classes=nb_classes)
elif args.net=='vit_base_16':
    print('vit base 16 used')
    net = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=nb_classes)
    # 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)
elif args.net=='vit_base_32':
    print('vit base 32 used')
    net = timm.create_model("vit_base_patch32_224", pretrained=True, num_classes=nb_classes)
elif args.net=='vit_small_16':
    print('vit small 16 used')
    net = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=nb_classes)
elif args.net=='vit_large_16':
    print('vit large 16 used')
    net = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=nb_classes)
elif args.net=='swin_base_4_7':
    print('swin base 4_7 used')
    net = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=nb_classes)
    # 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)
elif args.net=='cait_xxs24':
    print('cait xxs24 used')
    net = timm.create_model("cait_xxs24_224", pretrained=True, num_classes=nb_classes)
    # 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
elif args.net=='mixer_b16':
    print('mixer b16 used')
    net = timm.create_model("mixer_b16_224", pretrained=True, num_classes=nb_classes)
    # 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)
elif args.net=='coat_tiny':
    print('coat tiny used')
    net = timm.create_model("coat_tiny", pretrained=True, num_classes=nb_classes)
    # 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
else:
    raise ValueError("invalid network model")

# For Multi-GPU
if 'cuda' in device:
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)  
    
# use cosine scheduling
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
# use cosine scheduling with warmup
scheduler = CosineLRScheduler(optimizer, t_initial=args.n_epochs, lr_min=args.lr*0.05, warmup_t=int(0.05*args.n_epochs), warmup_lr_init=args.lr*0.05, warmup_prefix=True)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint_medical'):
            os.mkdir('checkpoint_medical')
        torch.save(state, './checkpoint_medical/{}-{}-bs{}-{}-lr{}-randaug{}-{}ckpt.t7'.format(args.dataset, args.net, args.bs, args.opt, args.lr, args.num_ops, args.magnitude))
        best_acc = acc

    if not os.path.isdir('log_medical'):
        os.makedirs('log_medical')
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log_medical/log_{args.dataset}_{args.net}_bs{args.bs}_{args.opt}_lr{args.lr}_randaug{args.num_ops}_{args.magnitude}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []


net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch+1) # step scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)

    # Write out csv..
    with open(f'log_medical/log_{args.dataset}_{args.net}_bs{args.bs}_{args.opt}_lr{args.lr}_randaug{args.num_ops}_{args.magnitude}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print([round(acc, 3) for acc in list_acc])

