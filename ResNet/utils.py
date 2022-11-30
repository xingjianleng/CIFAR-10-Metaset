import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class CIFAR101(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.imgs = np.load(f"{dataset_path}/cifar10.1_v6_data.npy")
        # default labels are int32, convert to int64
        self.labels = np.load(f"{dataset_path}/cifar10.1_v6_labels.npy").astype(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
