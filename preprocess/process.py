from FD_ACC.utils import process_multiple, CLASSES, FOLDER_ALIAS
from pathlib import Path
from collections import Counter

import numpy as np


def save_as_array(src: Path, dst: Path):
    # this function turns the source directory to a numpy array of images for each class
    # is stored as a numpy array
    # NOTE: assume input source directory are in a specific form
    imgs = []
    labels = []

    for sub_folder in src.iterdir():
        if sub_folder.name != ".DS_Store":
            folder_imgs = process_multiple(imgs_path=sub_folder)
            imgs.extend(folder_imgs)
            if sub_folder.name not in FOLDER_ALIAS:
                truth = CLASSES.index(sub_folder.name)
            else:
                truth = CLASSES.index(FOLDER_ALIAS[sub_folder.name])
            labels.extend([truth] * len(folder_imgs))

    np.save(str(dst / "data.npy"), np.array(imgs))
    np.save(str(dst / "labels.npy"), np.array(labels))


def sample(data: np.ndarray, labels: np.ndarray, dst: str):
    # randomly sample the provided data.npy and labels.npy
    assert len(data) == len(labels)
    num_dataset = min(Counter(labels).values()) // 100
    indices_each_class = []

    for i in range(len(CLASSES)):
        # NOTE: currently, we don't allow replacement
        chosen_idx = np.random.choice(np.where(labels == i)[0], num_dataset * 100, replace=False)
        indices_each_class.append(np.split(chosen_idx, num_dataset))

    for i in range(num_dataset):
        # each dataset has 100 * 10 = 1000 data
        data_rtn = np.zeros((1000, *data.shape[1:]), dtype=data.dtype)
        labels_rtn = np.zeros(1000, dtype=labels.dtype)
        for j in range(len(CLASSES)):
            sampled_data_folder = Path(dst + f"_{i}")
            sampled_data_folder.mkdir(exist_ok=True)
            data_rtn[100 * j: 100 * (j + 1)] = data[indices_each_class[j][i]]
            labels_rtn[100 * j: 100 * (j + 1)] = labels[indices_each_class[j][i]]
            np.save(str(sampled_data_folder / "data.npy"), data_rtn)
            np.save(str(sampled_data_folder / "labels.npy"), labels_rtn)


def process_main():
    dataset_name = "F-9"
    src = Path(f"~/Downloads/labelled/{dataset_name}").expanduser().absolute()
    dst = Path(f"data/custom_processed/{dataset_name}").expanduser().absolute()
    save_as_array(src=src, dst=dst)


def sample_main():
    dataset_name = "F-11"
    data = np.load(f"data/custom_processed/{dataset_name}/data.npy")
    labels = np.load(f"data/custom_processed/{dataset_name}/labels.npy")
    dst = f"data/custom_sampled/{dataset_name}"
    sample(data, labels, dst)


if __name__ == "__main__":
    process_main()
    # sample_main()
