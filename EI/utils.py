import torch
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
    with torch.no_grad():
        for imgs, _ in iter(dataloader):
            original = predict_multiple(model, imgs.to(device))[1]
            # col -> three rotations; row -> one rotation for all datasets
            eis = np.zeros((3, imgs.shape[0]))
            for rot in range(1, 4):
                imgs_rotated, _ = rotate_batch(imgs, rot)
                transformed = predict_multiple(model, imgs_rotated.to(device))[1]
                eis[rot - 1] = effective_invariance(original, transformed)
    # the rotation invariance of dataset is the mean of all rotation invariances
    return np.mean(eis)


def rgb2gray(img):
    # Assumes that tensor is (nchannels, height, width)
    # x = 0.299r + 0.587g + 0.114b
    # FIXME: Should it have 3 channels?
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]


def rgb2gray_batch(imgs):
    # convert multiple RGB images to grayscale image
    images = []
    for img in imgs:
        images.append(rgb2gray(img).unsqueeze(0))
    return torch.cat(images)


def gray_invariance():
    pass
