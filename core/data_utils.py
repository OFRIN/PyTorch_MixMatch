
from torchvision import datasets
from torchvision import transforms

from core.utils import *

class CIFAR10(datasets.CIFAR10):
    def __init__(self, data_dir, indices=None, train=True, download=False, transform=None):
        super().__init__(data_dir, train=train, download=download, transform=transform)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

def get_CIFAR10(
    data_dir, 
    image_size=32, 
    mean=(0.4914009, 0.48215914, 0.44653103), std=(0.20230275, 0.1994131, 0.2009607),
    train_transform=None, test_transform=None, 
    download=False
):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    in_channels = 3
    classes = 10

    dataset = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_transform)
    train_indices, validation_indices = split_train_and_validation_datasets(dataset, classes)

    train_dataset = CIFAR10(data_dir, train_indices, train=True, download=download, transform=train_transform)
    validation_dataset = CIFAR10(data_dir, validation_indices, train=True, download=download, transform=test_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=download, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

class CIFAR100(datasets.CIFAR100):
    def __init__(self, data_dir, indices=None, train=True, download=False, transform=None):
        super().__init__(data_dir, train=train, download=download, transform=transform)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

def get_CIFAR_100(
    data_dir, 
    image_size=32, 
    mean=(0.5070757, 0.4865504, 0.44091937), std=(0.20089693, 0.19844234, 0.20229684), 
    train_transform=None, test_transform=None, 
    download=False
):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    in_channels = 3
    classes = 10

    dataset = datasets.CIFAR100(data_dir, train=True, download=download, transform=train_transform)
    train_indices, validation_indices = split_train_and_validation_datasets(dataset, classes)

    train_dataset = CIFAR100(data_dir, train_indices, train=True, download=download, transform=train_transform)
    validation_dataset = CIFAR100(data_dir, validation_indices, train=True, download=download, transform=test_transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=download, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

class STL10(datasets.STL10):
    def __init__(self, data_dir, indices=None, split='train', download=False, transform=None):
        super().__init__(data_dir, split=split, download=download, transform=transform)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

def get_STL10(
    data_dir, 
    image_size=96, 
    mean=(0.44671088, 0.43981022, 0.406646), std=(0.22415751, 0.22150059, 0.22391169), 
    train_transform=None, test_transform=None, 
    download=False
):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    in_channels = 3
    classes = 10

    dataset = datasets.STL10(data_dir, split='train', download=download, transform=train_transform)
    train_indices, validation_indices = split_train_and_validation_datasets(dataset, classes)

    train_dataset = STL10(data_dir, train_indices, split='train', download=download, transform=train_transform)
    validation_dataset = STL10(data_dir, validation_indices, split='train', download=download, transform=test_transform)
    test_dataset = datasets.STL10(data_dir, split='test', download=download, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

class MNIST(datasets.MNIST):
    def __init__(self, data_dir, indices=None, train=True, download=False, transform=None):
        super().__init__(data_dir, train=train, download=download, transform=transform)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

def get_MNIST(
    data_dir, 
    image_size=28, 
    mean=(0.13066047,), std=(0.30150425,), 
    train_transform=None, test_transform=None,
    download=False
):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    in_channels = 1
    classes = 10
    
    dataset = datasets.MNIST(data_dir, train=True, download=download, transform=train_transform)
    train_indices, validation_indices = split_train_and_validation_datasets(dataset, classes)

    train_dataset = MNIST(data_dir, train_indices, train=True, download=download, transform=train_transform)
    validation_dataset = MNIST(data_dir, validation_indices, train=True, download=download, transform=test_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=download, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

class KMNIST(datasets.KMNIST):
    def __init__(self, data_dir, indices=None, train=True, download=False, transform=None):
        super().__init__(data_dir, train=train, download=download, transform=transform)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

def get_KMNIST(
    data_dir, 
    image_size=28, 
    mean=(0.19176215,), std=(0.33852664,), 
    train_transform=None, test_transform=None,
    download=False
):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    in_channels = 1
    classes = 10
    
    dataset = datasets.KMNIST(data_dir, train=True, download=download, transform=train_transform)
    train_indices, validation_indices = split_train_and_validation_datasets(dataset, classes)

    train_dataset = KMNIST(data_dir, train_indices, train=True, download=download, transform=train_transform)
    validation_dataset = KMNIST(data_dir, validation_indices, train=True, download=download, transform=test_transform)
    test_dataset = datasets.KMNIST(data_dir, train=False, download=download, transform=test_transform)
    
    return train_dataset, validation_dataset, test_dataset, in_channels, classes

class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, data_dir, indices=None, train=True, download=False, transform=None):
        super().__init__(data_dir, train=train, download=download, transform=transform)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

def get_FashionMNIST(
    data_dir, 
    image_size=28, 
    mean=(0.28604063,), std=(0.32045338,), 
    train_transform=None, test_transform=None,
    download=False
):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    in_channels = 1
    classes = 10
    
    dataset = datasets.FashionMNIST(data_dir, train=True, download=download, transform=train_transform)
    train_indices, validation_indices = split_train_and_validation_datasets(dataset, classes)

    train_dataset = FashionMNIST(data_dir, train_indices, train=True, download=download, transform=train_transform)
    validation_dataset = FashionMNIST(data_dir, validation_indices, train=True, download=download, transform=test_transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=download, transform=test_transform)
    
    return train_dataset, validation_dataset, test_dataset, in_channels, classes

class SVHN(datasets.SVHN):
    def __init__(self, data_dir, indices=None, split='train', download=False, transform=None):
        super().__init__(data_dir, split=split, download=download, transform=transform)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

def get_SVHN(
    data_dir, 
    image_size=32, 
    mean=(0.43768454, 0.44376868, 0.472804), std=(0.12008653, 0.123137444, 0.10520427), 
    train_transform=None, test_transform=None, 
    download=False
):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    in_channels = 3
    classes = 10

    dataset = datasets.SVHN(data_dir, split='train', download=download, transform=train_transform)
    train_indices, validation_indices = split_train_and_validation_datasets(dataset, classes)

    train_dataset = SVHN(data_dir, train_indices, split='train', download=download, transform=train_transform)
    validation_dataset = SVHN(data_dir, validation_indices, split='train', download=download, transform=test_transform)
    test_dataset = datasets.SVHN(data_dir, split='test', download=download, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, in_channels, classes
