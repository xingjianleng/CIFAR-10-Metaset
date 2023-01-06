import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from RP.model import ResNetRotation
from RP.rotation import rotate_batch

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=110, type=int,
                    help='total number of layers (default: 110)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--name', default='ResNet-110-ss', type=str,
                    help='name of experiment')
parser.add_argument('--rotation_type', default='rand')
parser.set_defaults(augment=True)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Data loading code
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '/data/lengx/cifar', train=True, download=False, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/data/lengx/cifar', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # create model
    model = ResNetRotation(depth=args.layers).to(device)

    # Load the model dictionary
    pretrained_dict = torch.load("model/resnet110-180-9321.pt")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    # Freeze the backbone, only train the fully connected layers
    for param in model.parameters():
        param.requires_grad = False
    model.fc_classification.weight.requires_grad = True
    model.fc_classification.bias.requires_grad = True
    model.fc_rotation.weight.requires_grad = True
    model.fc_rotation.bias.requires_grad = True

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()]))
    )

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, device, criterion, optimizer, epoch)

        # evaluate on validation set for semantic classification
        prec1 = validate(val_loader, model, device, criterion)

        # evaluate on validation set for rotation prediction
        test(val_loader, model, device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best
        )
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, device, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output, _ = model(input)
        loss_cls = criterion(output, target)

        # self-supervised
        inputs_ssh, labels_ssh = rotate_batch(input, args.rotation_type)
        inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
        _, outputs_ssh = model(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)

        # simply add two tasks' losses
        '''
        The users could also choose to only using semantic classification loss to train the backbone
        '''
        loss = loss_cls + loss_ssh

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss_Cls {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1)
            )


def validate(val_loader, model, device, criterion):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output, _ = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1)
            )

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def test(dataloader, model, device):
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    correct = []
    losses = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = rotate_batch(inputs, 'expand')
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    model.train()
    print('self-supervised.avg:{:.4f}'.format(correct.mean() * 100))


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """Saves checkpoint to disk"""
    directory = "model/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model/%s/' % (args.name) + 'model_best.pt')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 8 and 14 epochs"""
    lr = args.lr * (0.1 ** (epoch // 8)) * (0.1 ** (epoch // 14))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
