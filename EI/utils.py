import torch
import torchvision.transforms as T
import numpy as np

from RP.rotation import rotate_batch
from FD_ACC.utils import predict_multiple


def effective_invariance(original: np.ndarray, transformed: np.ndarray):
    # assertion: same batch size
    assert original.shape[0] == transformed.shape[0]

    out = np.zeros(original.shape[0])

    # assume original and transformed are softmax results
    original_pred = np.argmax(original, axis=1)
    transformed_pred = np.argmax(transformed, axis=1)

    for i in range(original.shape[0]):
        if original_pred[i] == transformed_pred[i]:
            out[i] = np.sqrt(
                original[i, original_pred[i]] * transformed[i, transformed_pred[i]]
            )
    return out


def rotat_invariance(dataloader, model, device):
    rotation_invs = []
    with torch.no_grad():
        for imgs, _ in iter(dataloader):
            original = predict_multiple(model, imgs.to(device))[1]
            # col -> three rotations; row -> one rotation for all datasets
            eis = np.zeros((3, imgs.shape[0]))
            for rot in range(1, 4):
                imgs_rotated, _ = rotate_batch(imgs, rot)
                transformed = predict_multiple(model, imgs_rotated.to(device))[1]
                eis[rot - 1] = effective_invariance(original, transformed)
            # the rotation invariance for each image
            rotation_invs.extend(np.mean(eis, axis=0).tolist())
    # the rotation invariance of dataset is the mean of all rotation invariances
    return np.mean(rotation_invs)


def gray_invariance(dataloader, model, device):
    grayscale_transform = T.Grayscale(num_output_channels=3)
    grayscale_invs = []
    with torch.no_grad():
        for imgs, _ in iter(dataloader):
            original = predict_multiple(model, imgs.to(device))[1]
            imgs_gray = grayscale_transform(imgs)
            transformed = predict_multiple(model, imgs_gray.to(device))[1]
            grayscale_invs.extend(effective_invariance(original, transformed))
    return np.mean(grayscale_invs)
