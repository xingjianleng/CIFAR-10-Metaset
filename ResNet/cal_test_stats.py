from model import ResNetCifar
from prediction import dataset_acc, get_dataloader, TRANSFORM
from fd import get_activations, calculate_frechet_distance
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision


# determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256

# load the model and change to evaluation mode
model = ResNetCifar(depth=110).to(device)
model.load_state_dict(torch.load("../model/resnet110-120.pt", map_location=torch.device(device)))
model.eval()

# submodel used for comparing FD between data
feature_extractor_model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten()).eval()
path_acc = Path("../model/acc_stats")
path_fd = Path("../model/fd_stats.npy")
acc_stats = None
fd_stats = None

# load/calculate the statistics on collected test set
if path_acc.exists() and path_acc.is_file():
    with open(path_acc, "rb") as f:
        acc_stats = pickle.load(f)

if path_fd.exists() and path_fd.is_file():
    fd_stats = np.load(path_fd)
    
if not acc_stats or fd_stats is None:
    # load datasets
    dataset_paths = [f"~/Downloads/dataset/F-{i}" for i in range(8, 16)]
    dataloaders = [get_dataloader(dataset_path, batch_size) for dataset_path in dataset_paths]

    if not acc_stats:
        acc_stats = [dataset_acc(dataloader, model, device) for dataloader in dataloaders]
        with open(path_acc, "wb") as f:
            pickle.dump(acc_stats, f)

    if fd_stats is None:
        # NOTE: also include original test file
        acts = [get_activations(dataloader, feature_extractor_model, cuda=(device == "cuda")) for dataloader in dataloaders]
        testloader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10(
                    "../data",
                    train=False,
                    transform=TRANSFORM
                ),
                batch_size=batch_size,
                shuffle=False,
            )
        acts.append(get_activations(testloader, feature_extractor_model, cuda=(device == "cuda")))

        fd_stats = np.zeros((len(dataset_paths) + 1, len(dataset_paths) + 1))
        for i in range(0, len(dataset_paths)):
            for j in range(i + 1, len(dataset_paths) + 1):
                mu1 = np.mean(acts[i], axis=0)
                sigma1 = np.cov(acts[i], rowvar=False)
                mu2 = np.mean(acts[j], axis=0)
                sigma2 = np.cov(acts[j], rowvar=False)
                fd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
                fd_stats[i, j] = fd
                fd_stats[j, i] = fd
        np.save(path_fd, fd_stats)
