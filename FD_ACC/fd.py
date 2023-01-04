# Calculate FD between datasets. Original code is from
# https://github.com/Simon4Yan/Meta-set/blob/58e498cc95a879eec369d2ccf8da714baf8480e2/FD/many_fd.py
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ResNet.model import ResNetCifar
from LeNet.model import LeNet5Feature
from FD_ACC.utils import TRANSFORM, CIFAR10F, CustomCIFAR

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from tqdm import trange, tqdm
from scipy import linalg

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description='PyTorch CIFAR-10 FD-Metaset')
parser.add_argument('-c', '--gpu', default='1', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('-s', '--save', default=False, type=bool,
                    help='whether save the calculated features')
args = parser.parse_args()

batch_size = 500
use_cuda = args.gpu and torch.cuda.is_available()

# load the model and change to evaluation mode
used_model = "resnet"
# used_model = "lenet"

if used_model == "resnet":
    model = ResNetCifar(depth=110)
    model.load_state_dict(torch.load("model/resnet110-180-9321.pt", map_location=torch.device("cpu")))
    model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
    # dimension of the feature
    dims = 64  # ResNet
elif used_model == "lenet":
    model = LeNet5Feature()
    model.load_state_dict(torch.load("model/lenet5-50.pt", map_location=torch.device("cpu")))
    # dimension of the feature
    dims = 84  # LeNet
else:
    raise ValueError(f"Unexpected used_model: {used_model}")

if use_cuda:
    model.cuda()
model.eval()


def get_activations(dataloader, model, dims,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FD score
                        implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                        of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
    """
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

            if cuda:
                batch = batch.cuda()

            pred = model(batch)
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch.shape[0], -1)

    if verbose:
        print('Done')

    return pred_arr


def calculate_activation_statistics(dataloader, model,
                                    dims=64, cuda=False, verbose=False):
    """Calculation of the statistics used by the FD.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(dataloader, model, dims, cuda, verbose=verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussian X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
    -- mu2   : The sample mean over activations, precalculated on a
                representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on a
                representative data set.
    Returns:
    --   : The Frechet Distance.
    """

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
            use_cuda,
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


def custom_cifar_main():
    # NOTE: change accordingly, may use os.listdir() method
    # base_dir = "/data/lengx/cifar/cifar10-test-transformed"
    # files = sorted(os.listdir(base_dir))
    dataset_name = "custom_cifar_clean"
    base_dir = f"/data/lengx/cifar/{dataset_name}/"
    candidates = sorted(os.listdir(base_dir))

    # candidates = []
    # for file in files:
    #     if file.endswith(".npy"):
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
            use_cuda,
            verbose=False
        )
        fd_value = calculate_frechet_distance(m1, s1, m2, s2)
        fd_values[i] = fd_value

        # saving features for nn regression
        if args.save:
            np.save(feat_path + '%s_mean' % candidate, m2)
            np.save(feat_path + '%s_variance' % candidate, s2)
            np.save(feat_path + '%s_feature' % candidate, act2)
    # save all frechet-inception-distance to a file
    np.save(path_fd, fd_values)

    # save the correspondence of dataset and its FID
    with open(f"generated_files/fid_correspondence_{used_model}.txt", "w") as f:
        for candidate, fd_value in zip(candidates, fd_values):
            f.write(f"{candidate}: {fd_value}\n")


def cifar_f_main():
    base_dir = '/data/lengx/cifar/cifar10-f'
    test_dirs = sorted(os.listdir(base_dir))
    feat_path = f"dataset_{used_model}_feature/cifar10-f/"

    if args.save:
        try:
            os.makedirs(feat_path)
        except FileExistsError:
            pass

    # NOTE: the "11" dataset have wrong labels, skip this dataset
    try:
        test_dirs.remove("11")
    except ValueError:
        pass

    fd_values = np.zeros(len(test_dirs))
    m1, s1, act1 = get_cifar_test_feat()

    with torch.no_grad():
        for i in trange(len(test_dirs)):
            path = test_dirs[i]
            test_loader = DataLoader(
                dataset=CIFAR10F(
                    path=base_dir + "/" + path,
                    transform=TRANSFORM
                ),
                batch_size=batch_size,
                shuffle=False,
            )
            m2, s2, act2 = calculate_activation_statistics(
                test_loader,
                model,
                dims,
                use_cuda,
                verbose=False,
            )

            fd_value = calculate_frechet_distance(m1, s1, m2, s2)
            fd_values[i] = fd_value

            if args.save:
                # saving features for nn regression
                np.save(feat_path + '%s_mean' % path, m2)
                np.save(feat_path + '%s_variance' % path, s2)
                np.save(feat_path + '%s_feature' % path, act2)
        np.save(f"dataset_{used_model}_FD/cifar10-f.npy", fd_values)


if __name__ == '__main__':
    # cifar_f_main()
    custom_cifar_main()
