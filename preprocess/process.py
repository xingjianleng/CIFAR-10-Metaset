from FD_ACC.utils import process_multiple, CLASSES, FOLDER_ALIAS
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def save_as_array(src: Path, dst: Path):
    # this function turns the source directory to a numpy array of images for each class
    # is stored as a numpy array
    # NOTE: assume input source directory are in a specific form
    imgs = []
    labels = []

    # make directory for the destination
    dst.mkdir(exist_ok=True)

    for sub_folder in src.iterdir():
        folder_imgs = process_multiple(imgs_path=sub_folder)
        imgs.extend(folder_imgs)
        if sub_folder.name not in FOLDER_ALIAS:
            truth = CLASSES.index(sub_folder.name)
        else:
            truth = CLASSES.index(FOLDER_ALIAS[sub_folder.name])
        labels.extend([truth] * len(folder_imgs))

    np.save(str(dst / "data.npy"), np.array(imgs))
    np.save(str(dst / "labels.npy"), np.array(labels))


def sample(
    data: np.ndarray,
    labels: np.ndarray,
    dst: str,
    num_dataset: int=None,
    dataset_size: int=1000,
    replace=False
):
    # `randomly` sample the provided data.npy and labels.npy into the desired
    # dataset_size and number of datasets
    # NOTE: `replace` parameter only indicate whether duplicate will happen in a singe dataset
    assert len(data) == len(labels)
    if num_dataset is not None:
        num_dataset = min(num_dataset, len(labels) // dataset_size)
    else:
        num_dataset = len(labels) // dataset_size

    for i in range(num_dataset):
        sampled_data_folder = Path(dst + f"_{i}")
        sampled_data_folder.mkdir(exist_ok=True)
        # Chosen `size` data from input, without replacement
        chosen_indices = np.random.choice(range(len(labels)), size=dataset_size, replace=replace)
        data_rtn = data[chosen_indices]
        labels_rtn = labels[chosen_indices]
            
        np.save(str(sampled_data_folder / "data.npy"), data_rtn)
        np.save(str(sampled_data_folder / "labels.npy"), labels_rtn)


def process_main():
    base_dir = "/data/lengx/cifar/custom_cifar_clean_original"
    for dataset_name in tqdm(os.listdir(base_dir)):
        src = Path(f"{base_dir}/{dataset_name}")
        dst = Path(f"/data/lengx/cifar/custom_cifar_clean/{dataset_name}")
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
