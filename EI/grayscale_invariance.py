import os

import torch
import numpy as np
import torch.utils.data
from tqdm import tqdm, trange

from EI.utils import gray_invariance
from FD_ACC.utils import CustomCIFAR, CIFAR10F, TRANSFORM
from ResNet.model import ResNetCifar
from LeNet.model import LeNet5


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
    # base_dir = "/data/lengx/cifar/cifar10-test-transformed/"
    # files = sorted(os.listdir(base_dir))
    dataset_name = "custom_cifar_clean"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"
    candidates = sorted(os.listdir(base_dir))

    # NOTE: code for CIFAR transformed 1000
    # candidates = []
    # for file in files:
    #     if file.endswith(".npy") and file.startswith("new_data"):
    #         candidates.append(file)

    path_gi = f"dataset_{used_model}_GI/{dataset_name}.npy"
    gi_stats = np.zeros(len(candidates))

    for i, candidate in enumerate(tqdm(candidates)):
        data_path = base_dir + f"{candidate}/data.npy"
        label_path = base_dir + f"{candidate}/labels.npy"
        # data_path = base_dir + candidate
        # label_path = f"{base_dir}/labels.npy"

        test_loader = torch.utils.data.DataLoader(
            dataset=CustomCIFAR(
                data_path=data_path,
                label_path=label_path,
                transform=TRANSFORM,
            ),
            batch_size=batch_size,
            shuffle=False
        )
        # store rotation invaraince on the dataset
        gi_stats[i] = gray_invariance(test_loader, model, device)
    np.save(path_gi, gi_stats)

    # save the correspondence of dataset and its rotation invariance
    # with open(f"generated_files/gi_correspondence_{used_model}.txt", "w") as f:
    #     for candidate, gi in zip(candidates, gi_stats):
    #         f.write(f"{candidate}: {gi}\n")


def cifar_f_main():
    base_dir = '/data/lengx/cifar/cifar10-f-32'
    test_dirs = sorted(os.listdir(base_dir))

    # NOTE: the "11" dataset have wrong labels, skip this dataset
    try:
        test_dirs.remove("11")
    except ValueError:
        pass

    path_gi = f"dataset_{used_model}_GI/cifar10-f.npy"
    gi_stats = np.zeros(len(test_dirs))

    for i in trange(len(test_dirs)):
        path = test_dirs[i]
        test_loader = torch.utils.data.DataLoader(
            dataset=CIFAR10F(
                path=base_dir + "/" + path,
                transform=TRANSFORM
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        # store rotation invariance on the dataset
        gi_stats[i] = gray_invariance(test_loader, model, device)
    np.save(path_gi, gi_stats)


def cifar101_main():
    dataset_name = "cifar-10.1"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"

    path_gi = f"dataset_{used_model}_GI/{dataset_name}.npy"

    data_path = base_dir + "cifar10.1_v6_data.npy"
    label_path = base_dir + "cifar10.1_v6_labels.npy"

    test_loader = torch.utils.data.DataLoader(
        dataset=CustomCIFAR(
            data_path=data_path,
            label_path=label_path,
            transform=TRANSFORM,
        ),
        batch_size=batch_size,
        shuffle=False
    )
    # store rotation invariance on the dataset
    grayscale_inv = gray_invariance(test_loader, model, device)
    np.save(path_gi, grayscale_inv)


if __name__ == "__main__":
    # cifar_f_main()
    custom_cifar_main()
    # cifar101_main()
