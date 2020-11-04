import torch
import numpy as np

from torchvision import transforms

from core.data_utils import *
from utility.utils import *

dataset_info_list = [
    ['CIFAR-10', get_CIFAR10],
    ['CIFAR-100', get_CIFAR100],
    ['STL-10', get_STL10],
    ['MNIST', get_MNIST],
    ['KMNIST', get_KMNIST],
    ['FashionMNIST', get_FashionMNIST],
    ['SVHN', get_SVHN],
]

customized_transforms = transforms.Compose([
    transforms.ToTensor(),
])

log_path = './data/info_mean_and_std.txt'
if os.path.isfile(log_path): 
    os.remove(log_path)

log_func = lambda string='': log_print(string, log_path)
# log_func = lambda string: print(string)

for name, function in dataset_info_list:
    # 1. load dataset
    train_dataset, _, _, _, _ = function('./data/', train_transforms=customized_transforms)
    
    means = None
    stds = None

    for image, label in train_dataset:
        mean = torch.mean(image, dim=(1, 2)).numpy()
        std = torch.std(image, dim=(1, 2)).numpy()

        if means is None: 
            means = mean
        else:
            means += mean
        
        if stds is None:
            stds = std
        else:
            stds += std
    
    mean = tuple(means / len(train_dataset))
    std = tuple(stds / len(train_dataset))

    log_func(f'# name={name}, mean={mean}, std={std}')
