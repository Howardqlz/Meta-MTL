# Meta-MTL
Code for AAAI20 oral paper "Constructing Multiple Tasks for Augmentation: Improving Neural Image Classification With K -means Features" by Tao Gui, Lizhi Qing, Qi Zhang, Jiacheng Ye, HangYan, Zichu Fei and Xuanjing Huang.

Cause the codes for different datasets are almost the same, but the auxiliary tasks data are quite large. So this code was used to produce part of the results(*CIFAR10、CIFAR100 and miniImageNet*) in the paper.

## Dependencies
The code was tested with the following setup:
* CentOS Linux release 7.7.1908 (Core)

* Python 3.5.2

* Torch 1.0.0.dev20190328


## Data
Auxiliary tasks data can be found in [this folder](https://github.com/Howardqlz/Meta-MTL/tree/master/aux_tasks), we provide one set of labels based on [ACAI embedding](https://github.com/brain-research/acai) for each dataset. For Cifar10 and Cifar100 the download of the datasets are automatic. For miniImageMet you need to  download the dataset ,then split it into 100 classes. We use 500/100 split for each class.

## Usage
Example:

    CUDA_VISIBLE_DEVICES=0 python main.py --train --cifar10 --extra 15 --num_tasks 4

    CUDA_VISIBLE_DEVICES=0 python main.py --train --cifar100 --extra 100 --num_tasks 8
    
    CUDA_VISIBLE_DEVICES=0 python main.py --train --miniimagenet --extra 100 --num_tasks 4
    

## Credits
The image representations were computed using two open-source codebases from prior works.

* [Deep Clustering for Unsupervised Learning of Visual Features](https://github.com/facebookresearch/deepcluster)

* [Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](https://github.com/brain-research/acai)







