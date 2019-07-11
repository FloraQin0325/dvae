# from cleverhans.utils_tf import model_eval
# from cleverhans.attacks import FastGradientMethod
# from cleverhans_tutorials.tutorial_models import *
# from cleverhans.attacks import CarliniWagnerL2
from cnn_models import model_a, model_b, model_c, model_d, model_e, model_f
import os
import math
import argparse
import glob
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1)###1:mnist; 2:fmnist
parser.add_argument('--cnn_model', type=int, default=1)
parser.add_argument('--attack_method', type=int, default=3)### 1:fgsm; 2:rand fgsm 3:CW
parser.add_argument('--epsilon', type=float, default=0.25)###for cw: 0.2 without feed; 0.3 with feed
parser.add_argument('--cnn-epochs', type=int, default=20)
parser.add_argument('--imagepath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/cropped_training/')### 32*32 original image path
parser.add_argument('--advpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/attacked_cropped_training/')### 32*32 adversarial image path
parser.add_argument('--labelpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/labels/')### the ground truth labels of 32*32 original image path
parser.add_argument('--savedir',  type=str, default='/home/floraqin/Documents/dvae/training_data/')### x,y,adv_x save diretory
args = parser.parse_args()

def parseTrainingDatawithAdv(imagePath='/home/floraqin/Documents/lisa-cnn-attack/cropped_training/',
                            labels = '/home/floraqin/Documents/lisa-cnn-attack/labels/',
                            AdvPath='/home/floraqin/Documents/lisa-cnn-attack/attacked_cropped_training/'):
    images = glob.glob(imagePath+'*.png')
    X_train = []
    Y_train = []
    X_Advtrain = []
    for image in images:
        idx = image[-10:-4]
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("img.shape: ",img.shape)
        X_train.append(img)

        advimg = cv2.imread(AdvPath+idx+'.png')
        advimg = cv2.cvtColor(advimg, cv2.COLOR_BGR2RGB)
        X_Advtrain.append(advimg)
        label = labels + idx+'.txt'
        file = open(label,'r')
        ylabel = int(file.readline().split(' ')[0])
        Y_train.append(ylabel)
    Y_train = np.asarray(Y_train)
    Y_train = to_categorical(Y_train,17)
    X_train = np.asarray(X_train)
    X_Advtrain = np.asarray(X_Advtrain)

    print(X_train.shape, Y_train.shape, X_Advtrain.shape)
    return X_train, Y_train, X_Advtrain  ## (5237, 32, 32, 3), (5237, 17)


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
X_train, Y_train, X_Advtrain = parseTrainingDatawithAdv(imagePath=args.imagepath, labels=args.labelpath, AdvPath=args.advpath)
savedir = args.savedir
np.save(savedir + 'xt.npy', X_train)
np.save(savedir + 'adv_x.npy', X_Advtrain)
np.save(savedir + 'yt.npy', Y_train)
