from ResNet.model import ResNetCifar
from FD_ACC.utils import CIFAR10F, CustomCIFAR, predict_multiple, TRANSFORM
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

device = "cuda:1" if torch.cuda.is_available() else "cpu"
batch_size = 1000
model = ResNetCifar(depth=110)
model.load_state_dict(torch.load("model/resnet110-180-9321.pt", map_location=torch.device("cpu")))
model.to(device)
model.eval()
# NOTE: Change this to adjust the sample accuracy
target_acc = 0.40
num_dataset = 4


def sample_with_accuracy(
    correct_imgs: np.ndarray,
    correct_labels: np.ndarray,
    wrong_imgs: np.ndarray,
    wrong_labels: np.ndarray,
    target_acc: float,
    num_dataset: int=None,
    dataset_size: int=1000,
    replace=False,
):
    # sample the data to a given accuracy
    # NOTE: `replace` parameter only indicate whether duplicate will happen in a singe dataset
    assert len(correct_imgs) == len(correct_labels) and len(wrong_imgs) == len(wrong_labels)
    rtn_imgs, rtn_labels = [], []

    if target_acc == 1.0:
        # case where we want only correct data
        if num_dataset is None:
            num_dataset = len(correct_labels) // dataset_size
        else:
            num_dataset = min(num_dataset, len(correct_labels) // dataset_size)
        for _ in range(num_dataset):
            selected_indices = np.random.choice(
                range(len(correct_labels)),
                size=dataset_size,
                replace=replace,
            )
            rtn_imgs.append(correct_imgs[selected_indices])
            rtn_labels.append(correct_labels[selected_indices])

    elif target_acc == 0.0:
        # case where we want only wrong data
        if num_dataset is None:
            num_dataset = len(wrong_labels) // dataset_size
        else:
            num_dataset = min(num_dataset, len(wrong_labels) // dataset_size)
        for _ in range(num_dataset):
            selected_indices = np.random.choice(
                range(len(wrong_labels)),
                size=dataset_size,
                replace=replace,
            )
            rtn_imgs.append(wrong_imgs[selected_indices])
            rtn_labels.append(wrong_labels[selected_indices])

    else:
        max_correct_dataset_count = len(correct_labels) // int(dataset_size * target_acc)
        max_wrong_dataset_count = len(wrong_labels) // int(dataset_size * (1 - target_acc))
        if num_dataset is None:
            num_dataset = min(max_correct_dataset_count, max_wrong_dataset_count)
        else:
            num_dataset = min(num_dataset, max_correct_dataset_count, max_wrong_dataset_count)
        num_correct_samples = int(dataset_size * target_acc)
        num_wrong_samples = int(dataset_size * (1 - target_acc))
        for _ in range(num_dataset):
            selected_correct_indices = np.random.choice(
                range(len(correct_labels)),
                size=num_correct_samples,
                replace=replace,
            )
            selected_wrong_indices = np.random.choice(
                range(len(wrong_labels)),
                size=num_wrong_samples,
                replace=replace,
            )
            rtn_imgs.append(
                np.concatenate(
                    (correct_imgs[selected_correct_indices], wrong_imgs[selected_wrong_indices]),
                    axis=0,
                )
            )
            rtn_labels.append(
                np.concatenate(
                    (correct_labels[selected_correct_indices], wrong_labels[selected_wrong_indices]),
                    axis=0,
                )
            )

    return rtn_imgs, rtn_labels


def custom_cifar_main():
    base_dir = "data/custom_processed/"
    datasets = sorted(os.listdir(base_dir))
    dst = "data/correct_wrong/custom_accuracy_sampled"

    correct = {}
    wrong = {}

    for dataset in tqdm(datasets):
        correct[dataset] = []
        wrong[dataset] = []
        dataset_path = base_dir + dataset
        dataloader = DataLoader(
            CustomCIFAR(dataset_path + "/data.npy", dataset_path + "/labels.npy", TRANSFORM),
            batch_size=batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for j, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                pred, _ = predict_multiple(model, imgs)
                pred = pred.squeeze(1).cpu()
                correct[dataset].extend(np.where(pred == labels)[0] + j * batch_size)
                wrong[dataset].extend(np.where(pred != labels)[0] + j * batch_size)


    wrong_imgs = []
    wrong_labels = []
    correct_imgs = []
    correct_labels = []

    for dataset in datasets:
        dataset_path = base_dir + dataset
        ds = CustomCIFAR(dataset_path + "/data.npy", dataset_path + "/labels.npy")
        wrong_data = ds[wrong[dataset]]
        correct_data = ds[correct[dataset]]
        wrong_imgs.extend(wrong_data[0])
        wrong_labels.extend(wrong_data[1])
        correct_imgs.extend(correct_data[0])
        correct_labels.extend(correct_data[1])

    wrong_imgs = np.array(wrong_imgs)
    wrong_labels = np.array(wrong_labels)
    correct_imgs = np.array(correct_imgs)
    correct_labels = np.array(correct_labels)

    # sample according to the accuracy
    rtn_imgs, rtn_labels = sample_with_accuracy(
        correct_imgs=correct_imgs,
        correct_labels=correct_labels,
        wrong_imgs=wrong_imgs,
        wrong_labels=wrong_labels,
        target_acc=target_acc,
        num_dataset=num_dataset,
    )
    for i, (imgs, labels) in enumerate(zip(rtn_imgs, rtn_labels)):
        sampled_data_folder = Path(dst + f"_{i}")
        sampled_data_folder.mkdir(exist_ok=True)
        np.save(sampled_data_folder / "data.npy", imgs)
        np.save(sampled_data_folder / "labels.npy", labels)


def cifar_f_main():
    base_dir = "data/cifar10-f/"
    datasets = sorted(os.listdir(base_dir))
    dst = "data/correct_wrong/cifar_f_accuracy_sampled"

    try:
        # this dataset has wrong labels
        datasets.remove("11")
    except ValueError:
        pass

    correct = {}
    wrong = {}

    for dataset in tqdm(datasets):
        correct[dataset] = []
        wrong[dataset] = []
        dataset_path = base_dir + dataset
        dataloader = DataLoader(
            CIFAR10F(dataset_path, TRANSFORM),
            batch_size=batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for j, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                pred, _ = predict_multiple(model, imgs)
                pred = pred.squeeze(1).cpu()
                correct[dataset].extend(np.where(pred == labels)[0] + j * batch_size)
                wrong[dataset].extend(np.where(pred != labels)[0] + j * batch_size)

    wrong_imgs = []
    wrong_labels = []
    correct_imgs = []
    correct_labels = []

    for dataset in datasets:
        ds = CIFAR10F(base_dir + dataset)
        wrong_data = ds[wrong[dataset]]
        correct_data = ds[correct[dataset]]
        wrong_imgs.extend(wrong_data[0])
        wrong_labels.extend(wrong_data[1])
        correct_imgs.extend(correct_data[0])
        correct_labels.extend(correct_data[1])

    wrong_imgs = np.array(wrong_imgs)
    wrong_labels = np.array(wrong_labels)
    correct_imgs = np.array(correct_imgs)
    correct_labels = np.array(correct_labels)

    # sample according to the accuracy
    rtn_imgs, rtn_labels = sample_with_accuracy(
        correct_imgs=correct_imgs,
        correct_labels=correct_labels,
        wrong_imgs=wrong_imgs,
        wrong_labels=wrong_labels,
        target_acc=target_acc,
        num_dataset=num_dataset
    )
    for i, (imgs, labels) in enumerate(zip(rtn_imgs, rtn_labels)):
        sampled_data_folder = Path(dst + f"_{i}")
        sampled_data_folder.mkdir(exist_ok=True)
        np.save(sampled_data_folder / "data.npy", imgs)
        np.save(sampled_data_folder / "labels.npy", labels)


if __name__ == "__main__":
    custom_cifar_main()
    # cifar_f_main()
