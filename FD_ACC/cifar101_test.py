from ResNet.train import test
# from ResNet.model import ResNetCifar
from LeNet.model import LeNet5
from FD_ACC.utils import CustomCIFAR, TRANSFORM

import torch
from torch.utils.data import DataLoader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

    # model = ResNetCifar(depth=110)
    # model.load_state_dict(torch.load("model/resnet110-180-9321.pt", map_location=torch.device("cpu")))
    # model.to(device)
    model = LeNet5()
    model.load_state_dict(torch.load("model/lenet5-50.pt", map_location=torch.device("cpu")))
    model.to(device)

    dataset = CustomCIFAR(
        "data/cifar-10.1/cifar10.1_v6_data.npy",
        "data/cifar-10.1/cifar10.1_v6_labels.npy",
        TRANSFORM,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        **kwargs,
    )

    test_acc = test(model, device, test_loader)
    print(f"Test accuracy on CIFAR-10.1 is: {test_acc}")


if __name__ == "__main__":
    main()
