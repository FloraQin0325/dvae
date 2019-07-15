from __future__ import print_function
import argparse
import numpy as np
import os
from dataset import *
from torch.utils import data as data_
from vae_models import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1)###1:mnist; 2:fmnist
parser.add_argument('--conv_mu', type=float, default=0.0)
parser.add_argument('--conv_sigma', type=float, default=0.02)
parser.add_argument('--bn_mu', type=float, default=1.0)
parser.add_argument('--bn_sigma', type=float, default=0.02)
parser.add_argument('--r_begin', type=float, default=0.0)
parser.add_argument('--r_end', type=float, default=0.0)
parser.add_argument('--sigmoid', type=bool, default=0)
parser.add_argument('--sigmoid_lambda', type=float, default=10)
parser.add_argument('--vae_model', type=int, default=18)
parser.add_argument('--multi', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--train_epochs', type=int, default=5)
parser.add_argument('--model_name', type=str, default="bestmodel")
parser.add_argument('--optim', type=int, default=1)
parser.add_argument('--test_epsilon', type=float, default=0.3)
parser.add_argument('--attack_method', type=int, default=1)
parser.add_argument('--cnn_model', type=int, default=1)
parser.add_argument('--cnn_epochs', type=int, default=20)
parser.add_argument('--imagepath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/cropped_testing/')### 32*32 original image path
parser.add_argument('--advpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/attacked_cropped_testing/')### 32*32 adversarial image path

args = parser.parse_args()

base_savedir = '../training_data/'

device = torch.device("cpu")
model_dict = {18: VAE18()}
vae_num = args.vae_model
model_vae = model_dict[vae_num].to(device)

load_dir = base_savedir + 'vae_model_lambda/'

model_vae.load_state_dict(torch.load(load_dir + '{}.pth'.format(args.model_name)))
model_vae.to(device)

##load test data
savedir = '../testing_data/'
dataset = TestDataset(data_dir = args.imagepath, adv_data_dir = args.advpath)
test_data_loader = data_.DataLoader(dataset,
                              batch_size=1,
                              shuffle=False)
adv_l = []
### reconstruct
model_vae.training = False
for batch_idx, (clean_batch, adv_batch) in enumerate(test_data_loader):
    rec_adv_x, mu, logvar = model_vae(adv_batch)

    ####save
    rec_adv_x_numpy = rec_adv_x.cpu().detach().numpy()
    adv_l.append(rec_adv_x_numpy)
    rec_dir = savedir + 'rec_lambda/'
if not os.path.exists(rec_dir):
    os.makedirs(rec_dir)
adv_examples = np.vstack(adv_l)
print("adv_examples.shape: ",adv_examples.shape)
np.save(rec_dir + 'rec', adv_examples)
