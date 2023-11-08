'''Train CIFAR10 with PyTorch.'''
import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import * 
from utils import progress_bar
from torchsummaryX import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

## Settings for model
parser.add_argument('-m', '--model', default='augshufflenetv2', help='Model Type.')
parser.add_argument('-s', '--scaler', default=1.5, type=float, help='width scaler of models')
parser.add_argument('-sr','--split_ratio', default=0.375, type=float, help='split ratio')

## Settings for data
parser.add_argument('-d', '--dataset', default='cifar10',choices=['cifar10', 'cifar100'], help='Dataset name.')
parser.add_argument('--data_dir', default='./data', help='data path')

## Settings for fast training
parser.add_argument('-g', '--multi_gpu', default=0, help='Model Type.')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--seed', default=666, type=int, help='number of random seed')

parser.add_argument('--schedule', nargs='+', default=[20, 30, 50], type=int)
parser.add_argument('-opt', '--optmizer', default='cos',choices=['cos', 'step'], help='Dataset name.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate gamma')
parser.add_argument('-wd','--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch', default=300, type=int, help='total training epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
args = parser.parse_args()


SEED= args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1  
best_acc = 0

if args.dataset == 'cifar10':
   num_classes = 10
   #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD=(0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
   datagen = torchvision.datasets.CIFAR10

else:
   num_classes = 100
   #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD = (0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762)
   datagen = torchvision.datasets.CIFAR100


transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     #transforms.Normalize(CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD),
                    ])

transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD),
                    ])

trainset = datagen(root=args.data_dir, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

testset = datagen(root=args.data_dir, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.workers)



# select model

if args.model == 'shufflenetv2' :
   net = ShuffleNetV2(args.scaler, num_classes,args.split_ratio)
   print('ShufflenetV2 is loaded')


else:
    net = AugShuffleNetV2(args.scaler, num_classes,args.split_ratio)
    print('AugShufflenetv2 is loaded')

#net = torch.compile(net)
if device == 'cuda' and args.multi_gpu==1:
   net = nn.DataParallel(net)


model_path = './checkpoint/seed_{}_split_{}_{}_{}_{}x_weight_decay_{}_lr_{}_{}_epoch_ckpt.pth'.format(args.seed, args.split_ratio,args.dataset,args.model,args.scaler,args.weight_decay,args.lr,args.epoch)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


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
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, model_path)
        best_acc = acc



summary(net, torch.zeros((1, 3, 32, 32)))
net = net.to(device)

# Train original ResNets


optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

for epoch in range(start_epoch, start_epoch+(args.epoch)):
    train(epoch,optimizer)
    test(epoch)
    scheduler.step()

torch.save(net.state_dict(),model_path)

