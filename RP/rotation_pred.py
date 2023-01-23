import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from RP.model import ResNetRotation, RepVGGRotation
from RP.rotation import rotate_batch
from FD_ACC.utils import TRANSFORM, CustomCIFAR

parser = argparse.ArgumentParser(description='Rotation prediction accuracy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
args = parser.parse_args()

# the model used
used_model = "resnet"
# used_model = "repvgg"

# create model and decide which GPU to use
device = "cuda" if torch.cuda.is_available() else "cpu"
if used_model == "resnet":
    model = ResNetRotation()
    model_state = model.state_dict()
    fc_rot_weights = torch.load(
        "model/resnet-rotation-fc.pt", map_location=torch.device("cpu")
    )
elif used_model == "repvgg":
    model = RepVGGRotation()
    model_state = model.state_dict()
    fc_rot_weights = torch.load(
        "model/repvgg-rotation-fc.pt", map_location=torch.device("cpu")
    )
else:
    raise ValueError(f"Unexpected used_model: {used_model}")

for key, value in fc_rot_weights.items():
    model_state[key] = value
model.load_state_dict(model_state)
model.to(device)
model.eval()


def main():
    # test model on the given dataset
    ssh_acc = []

    # NOTE: change accordingly, may use os.listdir() method
    # base_dir = "/data/lengx/cifar/cifar10-test-transformed/"
    # files = sorted(os.listdir(base_dir))
    dataset_name = "google_cartoon"
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
    # with open(f"generated_files/rp_correspondence_{used_model}.txt", "w") as f:
    #     for candidate, rot_pred in zip(candidates, ssh_acc):
    #         f.write(f"{candidate}: {rot_pred}\n")


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
    main()
