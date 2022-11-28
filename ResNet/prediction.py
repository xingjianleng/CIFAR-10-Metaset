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

CLASSNAME_MAPPING = {
    "sedans": "automobile",
    "suvs": "automobile",
    "trucks": "truck",
}

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def process_single(img_path):
    # reshape image to 32x32, convert to RGB color space
    return cv2.cvtColor(cv2.resize(cv2.imread(img_path), (32, 32)), cv2.COLOR_BGR2RGB)


def process_multiple(imgs_path):
    # the imgs_path is a Path object
    assert isinstance(imgs_path, Path)
    # convert multiple images
    rtn = []
    img_suffix = (".jpg", ".png")
    for file in imgs_path.iterdir():
        if file.suffix in img_suffix:
            rtn.append(process_single(str(file)))
    return np.array(rtn)


class CustomCIFAR(torch.utils.data.Dataset):
    # Custom Dataset class for new CIFAR test set
    def __init__(self, imgs_path, transform=None):
        self.imgs = process_multiple(imgs_path=imgs_path)
        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        return img


def predict_single(model, img, device):
    # assume single image input with shape (3, 32, 32)
    assert isinstance(img, torch.Tensor) and img.shape == (3, 32, 32)
    # NOTE: make sure model is in validation mode
    model.eval()
    prob = model(img.unsqueeze(0).to(device))[0]
    pred_labels = CLASSES[torch.argmax(prob)]
    # print(f"Predicted label is: {classes[torch.argmax(prob)]}")
    return pred_labels, torch.nn.functional.softmax(prob, dim=0).cpu().detach().numpy()


def predict_multiple(model, imgs, device):
    # assume multiple image inputs with shape (N, 3, 32, 32) where N is the batch size
    assert isinstance(imgs, torch.Tensor) and imgs.shape[1:] == (3, 32, 32)
    # NOTE: make sure model is in validation mode
    model.eval()
    prob = model(imgs.to(device))
    pred = prob.argmax(dim=1, keepdim=True)
    pred_labels = [CLASSES[idx] for idx in pred]
    # print(f"Predicted labels are: {[classes[idx] for idx in pred]}")
    return pred_labels, torch.nn.functional.softmax(prob, dim=1).cpu().detach().numpy()


def dataset_acc(dataset_path, model, device, verbose):
    total = 0
    correct = 0
    dataset_path = Path(dataset_path).expanduser().absolute()
    for class_path in dataset_path.iterdir():
        if class_path.is_dir():
            if class_path.name not in CLASSNAME_MAPPING:
                truth = class_path.name
            else:
                truth = CLASSNAME_MAPPING[class_path.name]
            custom_set = CustomCIFAR(imgs_path=class_path, transform=transform)
            custom_loader = torch.utils.data.DataLoader(custom_set, batch_size=len(custom_set), shuffle=False)

            # prediction on images in the folder
            pred_multi, _ = predict_multiple(model=model, imgs=next(iter(custom_loader)), device=device)
            res = np.array(list(map(lambda x: x == truth, pred_multi)))
            if verbose:
                print(f"Acc for {class_path.name} is: {res.sum() / len(res)}")

            total += len(res)
            correct += np.sum(res)
        
    return correct / total
