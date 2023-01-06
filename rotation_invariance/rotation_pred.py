import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm, trange

from rotation_invariance.model import ResNetRotation
from rotation_invariance.rotation import rotate_batch
from FD_ACC.utils import TRANSFORM, CIFAR10F, CustomCIFAR

parser = argparse.ArgumentParser(description='Rotation prediction accuracy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=110, type=int,
                    help='total number of layers (default: 110)')
args = parser.parse_args()

# the model used
used_model = "resnet"

# create model and decide which GPU to use
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = ResNetRotation(depth=args.layers)
# only load the state_dict from the checkpoint, drop the `best_prec`` and `epoch`` info
model.load_state_dict(torch.load("model/resnet110-ss-9315.pt")["state_dict"])
model = model.to(device)
model.eval()


def custom_cifar_main():
    # test model on the given dataset
    ssh_acc = []

    # NOTE: change accordingly, may use os.listdir() method
    # base_dir = "/data/lengx/cifar/cifar10-test-transformed/"
    # files = sorted(os.listdir(base_dir))
    dataset_name = "custom_cifar_clean"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"
    candidates = sorted(os.listdir(base_dir))
    # candidates = []
    # for file in files:
    #     if file.endswith(".npy") and file.startswith("new_data"):
    #         candidates.append(file)
    path_RP = f"dataset_{used_model}_RP/{dataset_name}.npy"

    with torch.no_grad():
        for candidate in tqdm(candidates):
            # NOTE: The semantic classification is the same as the accuracy
            #       so we skip the classification accuracy as it was covered in
            #       FD_ACC module. So, only rotation prediction accuracy is recorded

            data_path = base_dir + f"{candidate}/data.npy"
            label_path = base_dir + f"{candidate}/labels.npy"
            # CIFAR-10-Transformed
            # data_path = base_dir + candidate
            # label_path = "/data/lengx/cifar/cifar10-test-transformed/labels.npy"
            test_loader = torch.utils.data.DataLoader(
                dataset=CustomCIFAR(
                    data_path=data_path,
                    label_path=label_path,
                    transform=TRANSFORM,
                ),
                batch_size=64,
                shuffle=False
            )
            # evaluate on rotation prediction
            ss = test(test_loader, model, device)
            ssh_acc.append(ss)
    ssh_acc = np.array(ssh_acc)
    np.save(path_RP, ssh_acc)

    # save the correspondence of dataset and its rotation prediction score
    with open(f"generated_files/rp_correspondence_{used_model}.txt", "w") as f:
        for candidate, rot_pred in zip(candidates, ssh_acc):
            f.write(f"{candidate}: {rot_pred}\n")


def cifar_f_main():
    # test model on the given dataset
    ssh_acc = []

    base_dir = '/data/lengx/cifar/cifar10-f-32'
    test_dirs = sorted(os.listdir(base_dir))

    # NOTE: the "11" dataset have wrong labels, skip this dataset
    try:
        test_dirs.remove("11")
    except ValueError:
        pass

    for i in trange(len(test_dirs)):
        # NOTE: The semantic classification is the same as the accuracy
        #       so we skip the classification accuracy as it was covered in
        #       FD_ACC module. So, only rotation prediction accuracy is recorded
        path = test_dirs[i]
        test_loader = torch.utils.data.DataLoader(
            dataset=CIFAR10F(
                path=base_dir + "/" + path,
                transform=TRANSFORM
            ),
            batch_size=64,
            shuffle=False,
        )
        # evaluate on rotation prediction
        ss = test(test_loader, model, device)
        ssh_acc.append(ss)
    ssh_acc = np.array(ssh_acc)
    np.save(f"dataset_{used_model}_RP/cifar10-f.npy", ssh_acc)


def cifar101_main():
    dataset_name = "cifar-10.1"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"

    path_acc = f"dataset_{used_model}_RP/{dataset_name}.npy"

    data_path = base_dir + "cifar10.1_v6_data.npy"
    label_path = base_dir + "cifar10.1_v6_labels.npy"

    test_loader = torch.utils.data.DataLoader(
        dataset=CustomCIFAR(
            data_path=data_path,
            label_path=label_path,
            transform=TRANSFORM,
        ),
        batch_size=64,
        shuffle=False
    )
    # store the test accuracy on the dataset
    ss = test(test_loader, model, device)
    # save all accuracy to a file
    np.save(path_acc, ss)


def test(dataloader, model, device):
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    correct = []
    losses = []
    for _, (inputs, labels) in enumerate(dataloader):
        inputs, labels = rotate_batch(inputs, 'expand')
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    # print('self-supervised.avg:{:.4f}'.format(correct.mean() * 100))
    return correct.mean() * 100


if __name__ == '__main__':
    # cifar_f_main()
    custom_cifar_main()
    # cifar101_main()
