# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

from core.data_utils import *
from core.utils import *

if __name__ == '__main__':
    data_dir = './data/'
    image_size = 32

    train_transform = transforms.Compose([])

    # for dataset_name in ['CIFAR-10', 'CIFAR-100', 'STL-10', 'MNIST', 'KMNIST', 'FashionMNIST', 'SVHN']:
    for dataset_name in ['CIFAR-10']:
        
        if dataset_name == 'CIFAR-10':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR_10(data_dir, image_size, train_transform=train_transform)

        elif dataset_name == 'CIFAR-100':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR_100(data_dir, image_size, train_transform=train_transform)

        elif dataset_name == 'STL-10':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_STL_10(data_dir, image_size, train_transform=train_transform)

        elif dataset_name == 'MNIST':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_MNIST(data_dir, image_size, train_transform=train_transform)

        elif dataset_name == 'KMNIST':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_KMNIST(data_dir, image_size, train_transform=train_transform)

        elif dataset_name == 'FashionMNIST':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_FashionMNIST(data_dir, image_size, train_transform=train_transform)
        
        elif dataset_name == 'SVHN':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_SVHN(data_dir, image_size, train_transform=train_transform)
        
        print(dataset_name, len(train_dataset), len(validation_dataset), len(test_dataset))

'''
CIFAR-10 50000
CIFAR-100 50000
STL-10 5000
MNIST 60000
KMNIST 60000
FashionMNIST 60000
SVHN 73257
'''

