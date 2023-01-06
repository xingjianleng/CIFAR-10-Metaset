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
used_model = "resnet"
# used_model = "lenet"

if used_model == "resnet":
    model = ResNetCifar(depth=110)
    model.load_state_dict(torch.load("model/resnet110-180-9321.pt", map_location=torch.device("cpu")))
elif used_model == "lenet":
    model = LeNet5()
    model.load_state_dict(torch.load("model/lenet5-50.pt", map_location=torch.device("cpu")))
else:
    raise ValueError(f"Unexpected used_model: {used_model}")

model.to(device)
model.eval()


def custom_cifar_main():
    # NOTE: change accordingly
    # base_dir = "/data/lengx/cifar/cifar10-test-transformed"
    # files = sorted(os.listdir(base_dir))
    dataset_name = "custom_cifar_clean"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"
    candidates = sorted(os.listdir(base_dir))

    # NOTE: code for CIFAR transformed 1000
    # candidates = []
    # for file in files:
    #     if file.endswith(".npy") and file.startswith("new_data"):
    #         candidates.append(file)

    path_acc = f"dataset_{used_model}_ACC/{dataset_name}.npy"
    acc_stats = np.zeros(len(candidates))

    for i, candidate in enumerate(tqdm(candidates)):
        data_path = base_dir + f"{candidate}/data.npy"
        label_path = base_dir + f"{candidate}/labels.npy"
        # data_path = base_dir + candidate
        # label_path = f"{base_dir}/labels.npy"

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
    # save all accuracy to a file
    np.save(path_acc, acc_stats)

    # save the correspondence of dataset and its accuracy
    with open(f"generated_files/acc_correspondence_{used_model}.txt", "w") as f:
        for candidate, acc in zip(candidates, acc_stats):
            f.write(f"{candidate}: {acc}\n")


def cifar_f_main():
    base_dir = '/data/lengx/cifar/cifar10-f'
    test_dirs = sorted(os.listdir(base_dir))

    # NOTE: the "11" dataset have wrong labels, skip this dataset
    try:
        test_dirs.remove("11")
    except ValueError:
        pass

    path_acc = f"dataset_{used_model}_ACC/cifar10-f.npy"
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


def cifar101_main():
    dataset_name = "cifar-10.1"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"

    path_acc = f"dataset_{used_model}_ACC/{dataset_name}.npy"

    data_path = base_dir + "cifar10.1_v6_data.npy"
    label_path = base_dir + "cifar10.1_v6_labels.npy"

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
    acc_stats = sum(correct.values()) / sum(total.values())
    # save all accuracy to a file
    np.save(path_acc, acc_stats)


if __name__ == "__main__":
    # cifar_f_main()
    # custom_cifar_main()
    cifar101_main()
