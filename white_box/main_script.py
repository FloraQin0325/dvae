import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=int, default=1)
parser.add_argument('--cnn_model', type=int, default=1)
parser.add_argument('--trainimagepath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/cropped_training/')### 32*32 original image path
parser.add_argument('--trainadvpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/attacked_cropped_training/')### 32*32 adversarial image path
parser.add_argument('--trainlabelpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/labels/')### the ground truth labels of 32*32 original image path
parser.add_argument('--trainsavedir',  type=str, default='/home/floraqin/Documents/defense-vae/training_data/')### x,y,adv_x save diretory

parser.add_argument('--testimagepath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/cropped_training/')### 32*32 original image path
parser.add_argument('--testadvpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/attacked_cropped_training/')### 32*32 adversarial image path
parser.add_argument('--testlabelpath', type=str, default='/home/floraqin/Documents/lisa-cnn-attack/labels/')### the ground truth labels of 32*32 original image path
parser.add_argument('--testsavedir',  type=str, default='/home/floraqin/Documents/defense-vae/training_data/')### x,y,adv_x save diretory
args = parser.parse_args()

GPU = args.gpu
dataset = args.dataset
cnn_model = args.cnn_model

# Generate training data

# print('Start generating adversarial images for training')
# os.system("CUDA_VISIBLE_DEVICES={} python make_training_data_RP2.py "
#           "--imagepath {} --advpath {} --labelpath {} --savedir {}"
#           .format(GPU, args.trainimagepath, args.trainadvpath, args.trainlabelpath, args.trainsavedir))
#
#
# # Generate testing data
# print('Start generating adversarial images for testing')
# os.system("CUDA_VISIBLE_DEVICES={} python make_test_data_RP2.py "
#           "--imagepath {} --advpath {} --labelpath {} --savedir {}"
#           .format(GPU, args.testimagepath, args.testadvpath, args.testlabelpath, args.testsavedir))


# # Concatenate training data
# print('Concatenate training data')
# os.system("CUDA_VISIBLE_DEVICES={} python make_training_data_multiattack.py "
#           "--dataset {} --cnn_model {}"
#           .format(GPU, dataset, cnn_model))

# Training VAE
print('Start Training VAE')
os.system("CUDA_VISIBLE_DEVICES={} python vae_conv_train_multi.py "
          "--dataset {} --cnn_model {}"
          .format(GPU, dataset, cnn_model))

# Reconstruct the testing data
# print('Start reconstructing adversarial testing images')
# for attack_method in [1,2,3]:
#     for eps in [0.3]:
#         print('Attack method: {}; epsilon: {}'.format(attack_method, eps))
#         os.system("CUDA_VISIBLE_DEVICES={} python rec_init_multi.py "
#                   "--dataset {} --attack_method {} --cnn_model {} --test_epsilon {}"
#                   .format(GPU, dataset, attack_method, cnn_model, eps))
#
# # Defense results
# print('Defense Results:')
# for attack_method in [1,2,3]:
#     for eps in [0.3]:
#         os.system("CUDA_VISIBLE_DEVICES={} python defense_init_multi.py "
#                   "--dataset {} --attack_method {} --cnn_model {} --test_epsilon {}"
#                   .format(GPU, dataset, attack_method, cnn_model, eps))
