# from cleverhans.utils_tf import model_eval
# from cleverhans.attacks import FastGradientMethod
# from cleverhans_tutorials.tutorial_models import *
# from cleverhans.attacks import CarliniWagnerL2
from cnn_models import model_a, model_b, model_c, model_d, model_e, model_f
import os
import math
import cv2
import argparse
import glob
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1)###1:mnist; 2:fmnist
parser.add_argument('--cnn_model', type=int, default=1)
parser.add_argument('--attack_method', type=int, default=1)### 1:fgsm; 2:rand fgsm 3:CW
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--cnn-epochs', type=int, default=20)
parser.add_argument('--imagepath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/cropped_testing/')### 32*32 original image path
parser.add_argument('--advpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/attacked_cropped_testing/')### 32*32 adversarial image path
parser.add_argument('--labelpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/labels/')### the ground truth labels of 32*32 original image path
parser.add_argument('--savedir',  type=str, default='/home/floraqin/Documents/defense-vae/testing_data/')### x,y,adv_x save diretory
args = parser.parse_args()

def parseTestingDatawithAdv(imagePath='/home/floraqin/Documents/lisa-cnn-attack/cropped_testing/',
                            labels = '/home/floraqin/Documents/lisa-cnn-attack/labels/',
                            AdvPath='/home/floraqin/Documents/lisa-cnn-attack/attacked_cropped_testing/'):
    images = glob.glob(imagePath+'*.png')
    X_test = []
    Y_test = []
    X_Advtest = []
    for image in images:
        idx = image[-10:-4]
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_test.append(img)

        advimg = cv2.imread(AdvPath+idx+'.png')
        advimg = cv2.cvtColor(advimg, cv2.COLOR_BGR2RGB)
        X_Advtest.append(advimg)

        label = labels + idx+'.txt'
        file = open(label,'r')
        ylabel = int(file.readline().split(' ')[0])
        Y_test.append(ylabel)
    Y_test = np.asarray(Y_test)
    Y_test = to_categorical(Y_test,17)
    X_test = np.asarray(X_test)
    X_Advtest = np.asarray(X_Advtest)

    print(X_test.shape, Y_test.shape, X_Advtest.shape)
    return X_test, Y_test, X_Advtest  ## (5237, 32, 32, 3), (5237, 17)


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

X_test, Y_test, X_Advtest = parseTestingDatawithAdv(imagePath=args.imagepath, labels=args.labelpath, AdvPath=args.advpath)
savedir = args.savedir
np.save(savedir + 'adv_x.npy', X_Advtest)
np.save(savedir + 'xt.npy', X_test)
np.save(savedir + 'yt.npy', Y_test)
