import os
from ResNet.model import ResNetCifar
from LeNet.model import LeNet5
from FD_ACC.utils import dataset_acc, TRANSFORM, CIFAR10F, CustomCIFAR

from tqdm import trange, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

# determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 500

# load the model and change to evaluation mode
# model = ResNetCifar(depth=110)
# model.load_state_dict(torch.load("model/resnet110-180-9321.pt", map_location=torch.device("cpu")))
# model.to(device)
model = LeNet5()
model.load_state_dict(torch.load("model/lenet5-50.pt", map_location=torch.device("cpu")))
model.to(device)
model.eval()


def custom_cifar_main():
    # NOTE: change accordingly
    base_dir = "/home/sunxx/project/Auto_evaluation_cla/CIFAR-setup/dataset/"
    files = sorted(os.listdir(base_dir))
    dataset_name = "cifar10-transformed"
    # base_dir = f"data/{dataset_name}/"
    # candidates = sorted(os.listdir(base_dir))

    # NOTE: code for CIFAR transformed 1000
    candidates = []
    for file in files:
        if file.endswith(".npy"):
            candidates.append(file)

    path_acc = f"dataset_lenet_ACC/{dataset_name}.npy"
    acc_stats = np.zeros(len(candidates))

    for i, candidate in enumerate(tqdm(candidates)):
        # data_path = base_dir + f"{candidate}/data.npy"
        # label_path = base_dir + f"{candidate}/labels.npy"
        data_path = base_dir + candidate
        label_path = "data/cifar10-test-transformed/labels.npy"

        test_loader = DataLoader(
            dataset=CustomCIFAR(
                data_path=data_path,
                label_path=label_path,
                transform=TRANSFORM,
            ),
            batch_size=batch_size,
            shuffle=False
        )
        # store the test accuracy on the dataset
        correct, total = dataset_acc(test_loader, model, device)
        acc_stats[i] = sum(correct.values()) / sum(total.values())
    np.save(path_acc, acc_stats)


def cifar_f_main():
    base_dir = 'data/cifar10-f'
    test_dirs = sorted(os.listdir(base_dir))

    # NOTE: the "11" dataset have wrong labels, skip this dataset
    try:
        test_dirs.remove("11")
    except ValueError:
        pass

    path_acc = "dataset_lenet_ACC/cifar10-f.npy"
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


if __name__ == "__main__":
    # cifar_f_main()
    custom_cifar_main()
