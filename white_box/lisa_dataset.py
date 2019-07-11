import os
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import glob
import sys
import numpy as np


def read_images(path, dtype=np.float32, color = True):
    img = cv2.imread(path)
    if img.ndim == 2:
        ### transfer from [H,W] to [1,H,W]
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))
class BboxDataset:
    def __init__(self, data_dir, adv_data_dir, split='train'
                 ):
        # data_dir = '../training_data/'
        # id_list_file = os.path.join(data_dir, '{0}.txt'.format(split))
        id_list_file = glob.glob(os.path.join(data_dir, '*.png'))
        self.ids = id_list_file
        self.data_dir = data_dir
        self.adv_data_dir = adv_data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        id_ = id_[-10:-4]
        # Load a image
        img_file = os.path.join(self.data_dir, id_ + '.png')
        img = read_images(img_file, color=True)
        adv_img_file = os.path.join(self.adv_data_dir, id_+'.png')
        adv_img = read_images(adv_img_file, color=True)
        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, adv_img

    __getitem__ = get_example
