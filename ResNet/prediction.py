# Helper library for making predictions on a given CIFAR-10 variant
import cv2
import torch
import numpy as np
import torchvision.transforms as T
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
    "sedans": "automobile",
    "suvs": "automobile",
    "trucks": "truck",
}

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def process_single(img_path):
    # reshape image to 32x32, convert to RGB color space
    return cv2.cvtColor(cv2.resize(cv2.imread(img_path), (32, 32)), cv2.COLOR_BGR2RGB)


def process_multiple(imgs_path):
    # the imgs_path is a Path object
    assert isinstance(imgs_path, Path)
    # convert multiple images (List of ndarray)
    rtn = []
    img_suffix = (".jpg", ".png")
    for file in imgs_path.iterdir():
        if file.suffix in img_suffix:
            rtn.append(process_single(str(file)))
    return rtn


class CustomCIFAR(torch.utils.data.Dataset):
    # Custom Dataset class for new CIFAR test set
    # NOTE: This Dataset class is subjected to change, current implementation require
    #       the following file structure.
    #       dataset_folder
    #           |______class1
    #           |______class2
    #           .
    #           .
    #           |______class11
    #       TODO: Later, all images will be put into one folder, with a separate label file.
    def __init__(self, dataset_path, transform=None):
        self.imgs = []
        self.labels = []
        self.transform = transform
        path = Path(dataset_path).expanduser().absolute()
        # the dataset_path should be a directory
        assert path.is_dir()
        # add all imgs to the imgs attribute
        for class_path in path.iterdir():
            if class_path.is_dir():
                # map the class name to its index
                if class_path.name not in FOLDER_ALIAS:
                    truth = CLASSES.index(class_path.name)
                else:
                    truth = CLASSES.index(FOLDER_ALIAS[class_path.name])
                subdirectory_imgs = process_multiple(class_path)
                self.imgs.extend(subdirectory_imgs)
                self.labels.extend([truth] * len(subdirectory_imgs))
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
    prob = model(img.unsqueeze(0))[0]
    pred = torch.argmax(prob)
    return pred, torch.nn.functional.softmax(prob, dim=0).cpu().detach().numpy()


def predict_multiple(model, imgs):
    # assume multiple image inputs with shape (N, 3, 32, 32) where N is the batch size
    assert isinstance(imgs, torch.Tensor) and imgs.shape[1:] == (3, 32, 32)
    # NOTE: make sure model is in validation mode
    model.eval()
    prob = model(imgs)
    pred = prob.argmax(dim=1, keepdim=True)
    return pred, torch.nn.functional.softmax(prob, dim=1).cpu().detach().numpy()


def get_dataloader(dataset_path, batch_size):
    dataset = CustomCIFAR(dataset_path=dataset_path, transform=TRANSFORM)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def dataset_acc(dataloader, model, device):
    total_classes = len(CLASSES)
    total = dict(zip(range(total_classes), [0] * total_classes))
    correct = dict(zip(range(total_classes), [0] * total_classes))

    # prediction on images in the folder
    for imgs, labels in iter(dataloader):
        imgs = imgs.to(device)
        pred_multi, _ = predict_multiple(model=model, imgs=imgs)
        for i in range(len(labels)):
            total[int(labels[i])] += 1
            if pred_multi[i] == labels[i]:
                correct[int(labels[i])] += 1
        
    return correct, total


def get_accuracy(correct, total, label):
    # get the accuracy of classification for one label
    idx = CLASSES.index(label)
    return correct[idx] / total[idx]
