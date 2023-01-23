# Calculate FD between datasets. Original code is from
# https://github.com/Simon4Yan/Meta-set/blob/58e498cc95a879eec369d2ccf8da714baf8480e2/FD/many_fd.py
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from FD_ACC.utils import TRANSFORM, CustomCIFAR

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from tqdm import tqdm
from scipy import linalg

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description='PyTorch CIFAR-10 FD-Metaset')
parser.add_argument('-c', '--gpu', default='1', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('-s', '--save', default=True, type=bool,
                    help='whether save the calculated features')
args = parser.parse_args()

batch_size = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model and change to evaluation mode
# used_model = "resnet"
used_model = "repvgg"

if used_model == "resnet":
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
    dims = 64
elif used_model == "repvgg":
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
    dims = 1280
else:
    raise ValueError(f"Unexpected used_model: {used_model}")

model.to(device)
model.eval()


def get_activations(dataloader, model, dims, device, verbose=False):
    batch_size = dataloader.batch_size
    n_used_imgs = len(dataloader.dataset)
    n_batches = n_used_imgs // batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if verbose:
                print(
                    '\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True
                )
            start = i * batch_size
            end = start + batch_size

            batch, _ = data
            batch = batch.to(device)

            pred = model(batch)
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch.shape[0], -1)

    if verbose:
        print('Done')

    return pred_arr


def calculate_activation_statistics(dataloader, model, dims, device, verbose=False):
    act = get_activations(dataloader, model, dims, device, verbose=verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; ''adding %s to diagonal of cov estimates' % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_cifar_test_feat():
    cifar_feat_path = f"dataset_{used_model}_feature/cifar10-test/"
    if args.save:
        try:
            os.makedirs(cifar_feat_path)
        except FileExistsError:
            pass
    # if features do not exist, calculate them
    if (not os.path.exists(cifar_feat_path) or 
            set(os.listdir(cifar_feat_path)) != {'mean.npy', 'variance.npy', 'feature.npy'}):
        cifar_test_loader = DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root="/data/lengx/cifar",
                train=False,
                transform=TRANSFORM,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        m1, s1, act1 = calculate_activation_statistics(
            cifar_test_loader,
            model,
            dims,
            device,
            verbose=False,
        )
        # saving features of training set
        if args.save:
            np.save(cifar_feat_path + 'mean.npy', m1)
            np.save(cifar_feat_path + 'variance.npy', s1)
            np.save(cifar_feat_path + 'feature.npy', act1)
    else:
        m1 = np.load(cifar_feat_path + 'mean.npy')
        s1 = np.load(cifar_feat_path + 'variance.npy')
        act1 = np.load(cifar_feat_path + 'feature.npy')
    return m1, s1, act1


def main():
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

    path_fd = f"dataset_{used_model}_FD/{dataset_name}.npy"
    feat_path = f"dataset_{used_model}_feature/{dataset_name}/"
    fd_values = np.zeros(len(candidates))
    m1, s1, act1 = get_cifar_test_feat()

    if args.save:
        try:
            os.makedirs(feat_path)
        except FileExistsError:
            pass

    for i, candidate in enumerate(tqdm(candidates)):
        data_path = base_dir + f"{candidate}/data.npy"
        label_path = base_dir + f"{candidate}/labels.npy"
        # CIFAR-10-Transformed
        # data_path = base_dir + candidate
        # label_path = "/data/lengx/cifar/cifar10-test-transformed/labels.npy"

        test_loader = DataLoader(
            dataset=CustomCIFAR(
                data_path=data_path,
                label_path=label_path,
                transform=TRANSFORM,
            ),
            batch_size=batch_size,
            shuffle=False
        )
        m2, s2, act2 = calculate_activation_statistics(
            test_loader,
            model,
            dims,
            device,
            verbose=False
        )
        fd_value = calculate_frechet_distance(m1, s1, m2, s2)
        fd_values[i] = fd_value

        # saving features for nn regression
        if args.save:
            np.save(f"{feat_path}{i}_mean", m2)
            np.save(f"{feat_path}{i}_variance", s2)
            np.save(f"{feat_path}{i}_feature", act2)
    # save all frechet-inception-distance to a file
    np.save(path_fd, fd_values)

    # save the correspondence of dataset and its FID
    # with open(f"generated_files/fid_correspondence_{used_model}.txt", "w") as f:
    #     for candidate, fd_value in zip(candidates, fd_values):
    #         f.write(f"{candidate}: {fd_value}\n")


if __name__ == '__main__':
    main()
