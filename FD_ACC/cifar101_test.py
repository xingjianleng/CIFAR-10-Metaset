from FD_ACC.utils import CustomCIFAR, TRANSFORM, dataset_acc

import torch
from torch.utils.data import DataLoader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
    
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    model.to(device)
    model.eval()

    dataset = CustomCIFAR(
        "/data/lengx/cifar/cifar-10.1/01/data.npy",
        "/data/lengx/cifar/cifar-10.1/01/labels.npy",
        TRANSFORM,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        **kwargs,
    )

    test_acc = dataset_acc(test_loader, model, device)
    print(f"Test accuracy on CIFAR-10.1 is: {test_acc}")


if __name__ == "__main__":
    main()
