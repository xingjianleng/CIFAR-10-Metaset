import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange

from ResNet.model import ResNetCifar
from LeNet.model import LeNet5
from FD_ACC.utils import TRANSFORM, CustomCIFAR, CIFAR10F, predict_multiple


# determine the device to use
device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    dataset_name = "diffusion_processed"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"
    candidates = sorted(os.listdir(base_dir))

    # NOTE: code for CIFAR transformed 1000
    # candidates = []
    # for file in files:
    #     if file.endswith(".npy") and file.startswith("new_data"):
    #         candidates.append(file)

    path_ps = f"dataset_{used_model}_PS/{dataset_name}.npy"
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
        correct = prediction_score(test_loader, model, device, threshold=0.9)
        ps_stats[i] = correct / total
    np.save(path_ps, ps_stats)

    # save the correspondence of dataset and its predicted scores
    # with open(f"generated_files/ps_correspondence_{used_model}.txt", "w") as f:
    #     for candidate, ps in zip(candidates, ps_stats):
    #         f.write(f"{candidate}: {ps}\n")


def cifar_f_main():
    base_dir = '/data/lengx/cifar/cifar10-f-32'
    test_dirs = sorted(os.listdir(base_dir))

    # NOTE: the "11" dataset have wrong labels, skip this dataset
    try:
        test_dirs.remove("11")
    except ValueError:
        pass

    path_ps = f"dataset_{used_model}_PS/cifar10-f.npy"
    ps_stats = np.zeros(len(test_dirs))

    for i in trange(len(test_dirs)):
        path = test_dirs[i]
        dataset = CIFAR10F(
            path=base_dir + "/" + path,
            transform=TRANSFORM
        )
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # store the predicted scores on the dataset
        total = len(dataset)
        correct = prediction_score(test_loader, model, device, threshold=0.9)
        ps_stats[i] = correct / total
    np.save(path_ps, ps_stats)


def cifar101_main():
    dataset_name = "cifar-10.1"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"

    path_ps = f"dataset_{used_model}_PS/{dataset_name}.npy"

    data_path = base_dir + "cifar10.1_v6_data.npy"
    label_path = base_dir + "cifar10.1_v6_labels.npy"

    dataset = CustomCIFAR(
        data_path=data_path,
        label_path=label_path,
        transform=TRANSFORM,
    )
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )
    # store the predicted scores on the dataset
    total = len(dataset)
    correct = prediction_score(test_loader, model, device, threshold=0.9)
    np.save(path_ps, correct / total)


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
    custom_cifar_main()
    # cifar101_main()
    # cifar_f_main()
