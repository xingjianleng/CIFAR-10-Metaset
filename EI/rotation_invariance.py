import os

import torch
import numpy as np
import torch.utils.data
from tqdm import tqdm

from EI.utils import rotat_invariance
from FD_ACC.utils import CustomCIFAR, TRANSFORM

# determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 500

# load the model and change to evaluation mode
used_model = "resnet"
# used_model = "repvgg"

if used_model == "resnet":
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
elif used_model == "repvgg":
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True)
else:
    raise ValueError(f"Unexpected used_model: {used_model}")

model.to(device)
model.eval()


def main():
    # NOTE: change accordingly
    base_dir = "/data/lengx/cifar/cifar10-test-transformed/"
    files = sorted(os.listdir(base_dir))
    dataset_name = "cifar10-transformed"
    # base_dir = f"/data/lengx/cifar/{dataset_name}/"
    # candidates = sorted(os.listdir(base_dir))

    # NOTE: code for CIFAR transformed 1000
    candidates = []
    for file in files:
        if file.endswith(".npy") and file.startswith("new_data"):
            candidates.append(file)

    path_ri = f"dataset_{used_model}_RI/{dataset_name}.npy"
    ri_stats = np.zeros(len(candidates))

    for i, candidate in enumerate(tqdm(candidates)):
        # data_path = base_dir + f"{candidate}/data.npy"
        # label_path = base_dir + f"{candidate}/labels.npy"
        data_path = base_dir + candidate
        label_path = f"{base_dir}/labels.npy"

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
        ri_stats[i] = rotat_invariance(test_loader, model, device)
    np.save(path_ri, ri_stats)

    # save the correspondence of dataset and its rotation invariance
    # with open(f"generated_files/ri_correspondence_{used_model}.txt", "w") as f:
    #     for candidate, ri in zip(candidates, ri_stats):
    #         f.write(f"{candidate}: {ri}\n")


if __name__ == "__main__":
    main()
