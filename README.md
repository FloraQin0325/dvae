# Defense-VAE
This repository contains the implementation of [defense-vae](https://arxiv.org/abs/1812.06570).

## Dependences
Python 2.7, Pytorch, Tensorflow 1.7, Cleverhans 2.1.0

## Repo Structure

* white_box: scripts to generate adversarial images from training and test data, train VAE, reconstruct from adversarial images, and test white-box defense accuracy.
For this project, we specifically choose the latest white_box attack method [Robust Physical-World Attack](https://arxiv.org/abs/1707.08945) on LISA dataset(http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html). Both the original dataset and adversarial dataset has been put in the folders.

## Quick Start

* Clone the Repo

* For the white-box defense, please run:
    * cd white_box
    * python main_script.py --gpu 0
        * --gpu: choose GPU index, use 0 if you only have one.


* The final results will be written into results.txt in the white_box and black_box folders. The attack index 1,2,3 corresponds to the FGSM, RANDFGSM and CW attack.


## Citations
Most of the code comes from the original paper [defense-vae](https://arxiv.org/abs/1812.06570).
If you found this paper useful, please cite it.

    @article{defense-vae18,
      title={Defense-VAE: A Fast and Accurate Defense against Adversarial Attacks},
      author={Xiang Li and Shihao Ji},
      journal={arXiv preprint arXiv:1812.06570},
      year={2018}
    }

## Contacts
Xiang Li, xli62@student.gsu.edu
