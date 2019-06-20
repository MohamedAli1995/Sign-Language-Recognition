import numpy as np
import scipy
import os
import cv2
from sklearn.preprocessing import MinMaxScaler


# def read_gray_img(path):
#     img = scipy.misc.imread(path, mode='L').astype(np.float)
#     img = img.reshape(img.shape[0], img.shape[1], 1)
#     return img
import cv2
def preprocess_image(path, target_dims):
    img = scipy.misc.imread(path, mode='L').astype(np.float)
    h, w = img.shape
    img_range = img.max() - img.min()
    img = (img - img.min()) / (img_range + 1e-5)

    if img.min() < -1 or img.max() > 1:
        print(" error in dataset normalization")
        exit()

    if h != target_dims[0] or w != target_dims[1]:
        img = scipy.misc.imresize(img, (target_dims[0], target_dims[1]))
    img = img.reshape(img.shape[0], img.shape[1], 1)

    return img

def paths_to_images(paths, target_dims):
    imgs = []
    for path in paths:
        img = preprocess_image(path, target_dims)
        imgs.append(img)

    return imgs


def one_hot_encoding(label, num_classes):
    one_hot_encoded = [0] * num_classes
    one_hot_encoded[int(label)] = 1
    return one_hot_encoded


def rgb_to_bgr(img):
    """Converts RGB to BGR

    Args:
        img: input image of color bytes arrangement R->G->B.

    Returns:
        Same image with color bytes arrangement B->G->R.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
