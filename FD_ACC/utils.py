import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

FOLDER_ALIAS = {
    "sedan": "automobile",
    "suv": "automobile",
}

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def process_single(img_path):
    # reshape image to 32x32, convert to RGB color space
    # the resize process could be very slow
    # As some images are encoded in RGBA, we enforce each image to be in RGBA
    # and drop the last channel
    img = Image.open(img_path)
    if img.mode == "P":
        # if the image is in mode P, convert to RGBA image
        img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img_arr = np.asarray(img)
    if len(img_arr.shape) < 3:
        print(img_path)
    if img.mode == "RGBA":
        img_arr = img_arr[:, :, :3]
    return cv2.resize(img_arr, (32, 32))


def process_multiple(imgs_path):
    # the imgs_path is a Path object
    assert isinstance(imgs_path, Path)
    # convert multiple images (List of ndarray)
    rtn = []
    img_suffix = (".jpg", ".png")
    for file in imgs_path.iterdir():
        if file.suffix in img_suffix:
            try:
                res = process_single(str(file))
                rtn.append(res)
            except:
                print(f"File: {file} skipped due to processing failure")
    return rtn


class CustomCIFAR(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        # data_path and label_path are assumed to be ndarray objects
        self.imgs = np.load(data_path)
        # cast to int64 for model prediction
        self.labels = np.load(label_path).astype(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR10F(Dataset):
    # Custom Dataset object for loading the CIFAR-10-F dataset
    def __init__(self, path, transform=None):
        self.imgs = []
        self.labels = []
        self.transform = transform
        path_obj = Path(path)
        for file in path_obj.iterdir():
            if file.suffix in (".jpg", ".png"):
                self.labels.append(int(file.name.split("_")[0]))
                self.imgs.append(cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB))
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def predict_single(model, img):
    # assume single image input with shape (3, 32, 32)
    assert isinstance(img, torch.Tensor) and img.shape == (3, 32, 32)
    # NOTE: make sure model is in validation mode
    model.eval()
    with torch.no_grad():
        prob = model(img.unsqueeze(0))[0]
        pred = torch.argmax(prob)
    return pred, torch.nn.functional.softmax(prob, dim=0).cpu().numpy()


def predict_multiple(model, imgs):
    # assume multiple image inputs with shape (N, 3, 32, 32) where N is the batch size
    assert isinstance(imgs, torch.Tensor) and imgs.shape[1:] == (3, 32, 32)
    # NOTE: make sure model is in validation mode
    model.eval()
    with torch.no_grad():
        prob = model(imgs)
        pred = prob.argmax(dim=1, keepdim=True)
    return pred, torch.nn.functional.softmax(prob, dim=1).cpu().numpy()


def dataset_acc(dataloader, model, device):
    correct = []
    # prediction on images in the folder
    with torch.no_grad():
        for imgs, labels in iter(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            pred_multi, _ = predict_multiple(model=model, imgs=imgs)
            correct.append(pred_multi.squeeze(1).eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    return np.mean(correct)
