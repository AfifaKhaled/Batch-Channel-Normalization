import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from vgg import *
from densenet import *
import numpy as np
import sys

f = open("BCN.txt", 'w')
sys.stdout = f
# from resnet import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.memory_summary(device=None, abbreviated=False)

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--normalization', default='BCN', type=str)
parser.add_argument('--data', default='/data/imagenet/train', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet201', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: densenet121)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--output_file', default='./Output_Result/1.1.pth', type=str)
parser.add_argument('--save_plots', default=True, help='Give argument if plots to be plotted')

best_prec1 = 0.0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    if args.normalization:
        print("=> using normalization'{}'".format(args.normalization))
    else:
        print("Please enter the normalization")
    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    def Save_Stats(trainacc, testacc, exp_name):
        data = []
        data.append(trainacc)
        data.append(testacc)
        data = np.array(data)
        data.reshape((2, -1))
        if not os.path.exists('./Results'):
            os.mkdir('./Results')
        stats_path = './Results/%s_accs.npy' % exp_name
        np.save(stats_path, data)
        SavePlots(data[0], data[1], 'Accuracy', exp_name)

    def SavePlots(y1, y2, metric, exp_name):
        try:
            plt.clf()
        except Exception as e:
            pass
        # plt.title(exp_name)
        plt.xlabel('Iterations')
        plt.ylabel(metric)
        epochs = np.arange(1, len(y1) + 1, 1)
        plt.plot(epochs, y1, label='Train %s' % metric)
        plt.plot(epochs, y2, label='Validation %s' % metric)
        ep = np.argmax(y2)
        plt.legend()
        plt.savefig('./Results/%s_%s' % (exp_name, metric), dpi=95)

    # use cuda
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train_acc = train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1, prec5, valu_acc = validate(val_loader, model, criterion, args.print_freq)
        # remember the best prec@1 and save checkpoint

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch + '.pth')

    torch.save(model.state_dict(), args.output_file)

    if args.save_plots != False:
        Save_Stats(train_acc, valu_acc, args.normalization)


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    train_acc = []
    total_samples, correct_predictions = 0, 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        Y_predicted = output.argmax(dim=1)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        correct_prediction = Y_predicted.eq(target).sum()
        correct_predictions += correct_prediction.item()
        total_samples += Y_predicted.size(0)
        train_acc.append((correct_predictions / total_samples) * 100.)
        np.savetxt('Training_Accuracy.txt', train_acc)
        print("Iterations ", i, "Training  Accuracy", train_acc[-1])
    return train_acc


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5: AverageMeter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    val_acc = []
    total_samples, correct_predictions = 0, 0
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            Y_predicted = output.argmax(dim=1)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        correct_prediction = Y_predicted.eq(target).sum()
        correct_predictions += correct_prediction.item()
        total_samples += Y_predicted.size(0)
        val_acc.append((correct_predictions / total_samples) * 100.)
        print("Epochs ", i, "Validation Accuracy", val_acc[-1])
        # np.savetxt('Validation_Accuracy.txt',val_acc)
    return top1.avg, top5.avg, val_acc


f.close
if __name__ == '__main__':
    main()
