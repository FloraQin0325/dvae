# from cleverhans.utils_tf import model_eval
# from cleverhans_tutorials.tutorial_models import *
import argparse
import numpy as np
from cnn_models import model_a, model_b, model_c, model_d

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=int, default=2)###1:mnist; 2:fmnist

parser.add_argument('--r_begin', type=float, default=0.0)
parser.add_argument('--r_end', type=float, default=0.0)
parser.add_argument('--sigmoid', type=bool, default=0)
parser.add_argument('--sigmoid_lambda', type=float, default=10)

parser.add_argument('--vae_model', type=int, default=18)
parser.add_argument('--train_epsilon', type=float, default=0.3)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--train_epochs', type=int, default=5)
parser.add_argument('--model_epochs', type=int, default=5)

parser.add_argument('--optim', type=int, default=1) #

parser.add_argument('--multi', type=int, default=100)
parser.add_argument('--attack_method', type=int, default=1)
parser.add_argument('--cnn_model', type=int, default=1)
parser.add_argument('--test_epsilon', type=float, default=0.3)
parser.add_argument('--cnn-epochs', type=int, default=20)
parser.add_argument('--vae-epochs', type=int, default=20)
parser.add_argument('--defense_method', type=int, default=1)#1,reconstruct 2,optim
args = parser.parse_args()


base_savedir = '../testing_data/'
### read recon im

rec_dir = base_savedir + 'rec_lambda/'
adv_x_recon = np.load(rec_dir + 'rec.npy')
adv_x_recon = adv_x_recon.transpose(0, 2, 3, 1)

### read lable
savedir = base_savedir + 'AdvSave_test/'
y_true = np.load(savedir + 'yt.npy')
####### PS: adv_x_recon.shape, y_true.shape    (1729, 32, 32, 3) (1729, 17)
