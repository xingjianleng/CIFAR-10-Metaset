import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from FD_ACC.utils import TRANSFORM, CustomCIFAR, predict_multiple


# determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 500
threshold = 0.8

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
    # base_dir = "/data/lengx/cifar/cifar10-test-transformed/"
    # files = sorted(os.listdir(base_dir))
    dataset_name = "google_cartoon"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"
    candidates = sorted(os.listdir(base_dir))

    # NOTE: code for CIFAR transformed 1000
    # candidates = []
    # for file in files:
    #     if file.endswith(".npy") and file.startswith("new_data"):
    #         candidates.append(file)

    path_ps = f"dataset_{used_model}_PS_{threshold}/{dataset_name}.npy"
    ps_stats = np.zeros(len(candidates))

    for i, candidate in enumerate(tqdm(candidates)):
        data_path = base_dir + f"{candidate}/data.npy"
        label_path = base_dir + f"{candidate}/labels.npy"
        # data_path = base_dir + candidate
        # label_path = f"{base_dir}/labels.npy"

        test_set = CustomCIFAR(
            data_path=data_path,
            label_path=label_path,
            transform=TRANSFORM,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
        )
        # store the predicted scores on the dataset
        total = len(test_set)
        correct = prediction_score(test_loader, model, device, threshold=threshold)
        ps_stats[i] = correct / total
    np.save(path_ps, ps_stats)

    # save the correspondence of dataset and its predicted scores
    # with open(f"generated_files/ps_correspondence_{used_model}.txt", "w") as f:
    #     for candidate, ps in zip(candidates, ps_stats):
    #         f.write(f"{candidate}: {ps}\n")


def prediction_score(dataloader, model, device, threshold):
    # prediction score on images in the folder
    correct = 0

    with torch.no_grad():
        for imgs, _ in iter(dataloader):
            imgs = imgs.to(device)
            _, probs = predict_multiple(model=model, imgs=imgs)
            correct += np.sum(np.max(probs, axis=1) > threshold)

    return correct


if __name__ == "__main__":
    main()
