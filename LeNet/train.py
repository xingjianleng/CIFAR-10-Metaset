from ResNet.train import CIFARDataset, load_data, train, test
from LeNet.model import LeNet5
import argparse
import copy

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch LeNet Train')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    train_x, val_x, test_x, train_y, val_y, test_y = load_data("data/cifar-10-batches-py")
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

    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=args.gamma)

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

    # load the best state_dict and test on test set
    model.load_state_dict(best_model)
    test(model, device, test_loader)

    if args.save_model:
        torch.save(best_model, f"model/lenet5-{args.epochs}.pt")
