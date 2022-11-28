# Implementation from
# https://github.com/Simon4Yan/Meta-set/blob/58e498cc95a879eec369d2ccf8da714baf8480e2/learn/train.py

import argparse
import sys
sys.path.append(".")
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

import copy
import pickle
from model import ResNetCifar


class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# function used to extract 
def unpickle(path):
    with open(path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_data(root):
    # image data is stored as N * W * H * C
    # 50000 training data, 10000 testing data
    train_x, train_y = np.zeros((50000, 32, 32, 3), dtype=np.uint8), []
    test_y = []

    # extract training data
    for i in range(1, 6):
        data_path = f"{root}/data_batch_{i}"
        data = unpickle(path=data_path)
        train_x[(i - 1) * 10000: i * 10000] = data[b'data'].reshape(10000, 3, 32, 32).transpose((0, 2, 3, 1))
        train_y.extend(data[b'labels'])

    # extract testing data
    data_path = f"{root}/test_batch"
    data = unpickle(path=data_path)
    test_x = data[b'data'].reshape(10000, 3, 32, 32).transpose((0, 2, 3, 1))
    test_y = data[b'labels']

    # convert to numpy arrays
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    # split the validation set (45000 train, 5000 val)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
    return train_x, val_x, test_x, train_y, val_y, test_y


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return the validation/test accuracy
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet Train')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=180, metavar='N',
                        help='number of epochs to train (default: 180)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # extract data and load to dataset
    train_x, val_x, test_x, train_y, val_y, test_y = load_data("../data/cifar-10-batches-py")
    train_set = CIFARDataset(train_x, train_y, transform_train)
    val_set = CIFARDataset(val_x, val_y, transform_test)
    test_set = CIFARDataset(test_x, test_y, transform_test)

    # put dataset to dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs,
    )

    model = ResNetCifar(depth=110).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[90, 135], gamma=args.gamma)

    best_val_acc = 0.0
    best_model = None
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch} with learning rate: {optimizer.param_groups[0]['lr']}")
        train(args, model, device, train_loader, optimizer, epoch)
        new_val_acc = test(model, device, val_loader)
        # store the best model on the validation set
        if new_val_acc > best_val_acc:
            best_model = copy.deepcopy(model.state_dict())
            best_val_acc = new_val_acc
        scheduler.step()
    
    # load the best state_dict and test on test setc
    model.load_state_dict(best_model)
    test(model, device, test_loader)

    if args.save_model:
        torch.save(best_model, f"../model/resnet110-{args.epochs}.pt")


if __name__ == '__main__':
    main()
