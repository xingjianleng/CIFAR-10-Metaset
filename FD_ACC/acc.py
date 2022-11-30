import os
from ResNet.model import ResNetCifar
from FD_ACC.utils import dataset_acc, TRANSFORM, CIFAR10F

from tqdm import trange
import numpy as np
import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # determine the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1000

    # load the model and change to evaluation mode
    model = ResNetCifar(depth=110).to(device)
    model.load_state_dict(torch.load("../model/resnet110-180-9321.pt", map_location=torch.device(device)))
    model.eval()

    base_dir = '../data/cifar10-f'
    test_dirs = sorted(os.listdir(base_dir))

    try:
        # skip the .DS_Store in macOS
        test_dirs.remove(".DS_Store")
    except ValueError:
        pass

    path_acc = "../dataset_ACC/cifar10-f.npy"
    acc_stats = np.zeros(len(test_dirs))

    for i in trange(len(test_dirs)):
        path = test_dirs[i]
        test_loader = DataLoader(
            dataset=CIFAR10F(
                path=base_dir + "/" + path,
                transform=TRANSFORM
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        # store the test accuracy on the dataset
        correct, total = dataset_acc(test_loader, model, device)
        acc_stats[i] = sum(correct.values()) / sum(total.values())
    np.save(path_acc, acc_stats)
