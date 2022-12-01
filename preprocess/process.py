from FD_ACC.utils import process_multiple, CLASSES, FOLDER_ALIAS
from pathlib import Path

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


def sample(data, labels):
    # TODO: randomly sample the provided data.npy and labels.npy
    pass


def main():
    dataset_name = "F-11"
    src = Path(f"~/Downloads/clean_original/{dataset_name}").expanduser().absolute()
    dst = Path(f"data/custom_processed/{dataset_name}").expanduser().absolute()
    save_as_array(src=src, dst=dst)


if __name__ == "__main__":
    main()
