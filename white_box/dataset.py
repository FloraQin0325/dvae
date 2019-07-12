from __future__ import  absolute_import
from __future__ import  division
import torch as t
from lisa_dataset import BboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import sys
import sklearn
from sklearn.preprocessing import StandardScaler

def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    #scaler = StandardScaler()
    #img = scaler.fit_transform(img)
    # img = zero_mean_unit_var(img)
    im_R=img.astype(np.float32)[0,:,:]
    im_G=img.astype(np.float32)[1,:,:]
    im_B=img.astype(np.float32)[2,:,:]

    im_R_mean=np.mean(im_R)
    im_G_mean=np.mean(im_G)
    im_B_mean=np.mean(im_B)

    im_R_std=np.std(im_R)
    im_G_std=np.std(im_G)
    im_B_std=np.std(im_B)

    # std_dev = np.std(img.astype(np.float32),axis=0)
    # mean = np.mean(img.astype(np.float32),axis=0)
    # print(im_R_mean)
    normalize = tvtsf.Normalize(mean=[im_R_mean, im_G_mean, im_B_mean],
                               std=[im_R_std, im_G_std, im_B_std])
    img = normalize(t.from_numpy(img).float())
    return img.numpy()
    # return img

def preprocess(img, min_size=300, max_size=1200):
    C, H, W = img.shape
    # scale1 = min_size / min(H, W)
    # scale2 = max_size / max(H, W)

    # scale = min(scale1, scale2)
    img = img.astype(np.float32) / 255.
    # img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size

    # normalize = pytorch_normalze
    # return normalize(img)
    return img




def read_images(path, dtype=np.float32, color = True):
    img = cv2.imread(path)
    if img.ndim == 2:
        ### transfer from [H,W] to [1,H,W]
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def zero_mean_unit_var(image):
    """Normalizes an image to zero mean and unit variance."""

    img_array = (image)
    img_array = img_array.astype(np.float32)

    std_dev = np.std(img_array)
    mean = np.mean(img_array)
    normalised_input = (img_array - mean) / std_dev

    image_normalised = normalised_input

    return image_normalised
class Transform(object):
    def __init__(self, min_size=600, max_size=1200):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        ori_img, adv_img = in_data
        ori_img = preprocess(ori_img)
        adv_img = preprocess(adv_img)
        # horizontally flip
        return ori_img, adv_img

class Dataset:
    def __init__(self, data_dir, adv_data_dir):
        self.db = BboxDataset(data_dir, adv_data_dir)
        self.tsf = Transform()
    def __getitem__(self, idx):
        ori_img, adv_img = self.db.get_example(idx)
        # ori_img = ori_img.astype(np.float32)
        # adv_img = adv_img.astype(np.float32)
        ori_img, adv_img = self.tsf((ori_img, adv_img))
        # ori_img = np.clip(ori_img, -1, 1)
        # adv_img = np.clip(adv_img, -1, 1)

        ori_img = np.clip(ori_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)


        # print(ori_img)
        # print("np.argwhere(ori_img>=1): ",np.argwhere(ori_img>=1))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return ori_img.copy(), adv_img.copy()

    def __len__(self):
        return len(self.db)

class TestDataset:
    def __init__(self, data_dir, adv_data_dir, split='test'):
        self.db = BboxDataset(data_dir, adv_data_dir, split=split)

    def __getitem__(self, idx):
        ori_img, adv_img = self.db.get_example(idx)
        # ori_img = ori_img.astype(np.float32)
        # adv_img = adv_img.astype(np.float32)
        img = preprocess(ori_img)

        adv_img = preprocess(adv_img)
        return img, adv_img

    def __len__(self):
        return len(self.db)
