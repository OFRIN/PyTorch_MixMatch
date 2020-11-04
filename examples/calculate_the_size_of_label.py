# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

from core.data_utils import *
from core.utils import *

if __name__ == '__main__':
    data_dir = ''
    train_transforms = transforms.Compose([])

    for dataset_name in ['CIFAR-10', 'CIFAR-100', 'STL-10', 'MNIST', 'KMNIST', 'FashionMNIST', 'SVHN']:

        if dataset_name == 'CIFAR-10':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR_10(args.data_dir, args.image_size, train_transforms=train_transforms)

        elif dataset_name == 'CIFAR-100':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR_100(args.data_dir, args.image_size, train_transforms=train_transforms)

        elif dataset_name == 'STL-10':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_STL_10(args.data_dir, args.image_size, train_transforms=train_transforms)

        elif dataset_name == 'MNIST':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_MNIST(args.data_dir, args.image_size, train_transforms=train_transforms)

        elif dataset_name == 'KMNIST':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_KMNIST(args.data_dir, args.image_size, train_transforms=train_transforms)

        elif dataset_name == 'FashionMNIST':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_FashionMNIST(args.data_dir, args.image_size, train_transforms=train_transforms)
        
        elif dataset_name == 'SVHN':
            train_dataset, validation_dataset, test_dataset, in_channels, classes = get_SVHN(args.data_dir, args.image_size, train_transforms=train_transforms)
        
        labels = get_labels_from_dataset(train_dataset)
        print(dataset_name, len(labels))

'''
CIFAR-10 50000
CIFAR-100 50000
STL-10 5000
MNIST 60000
KMNIST 60000
FashionMNIST 60000
SVHN 73257
'''