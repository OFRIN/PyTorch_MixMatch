
# Installation
```
python3 -m pip install scipy
```

# Experiments
```
python3 train.py --use_gpu 0 --seed 0 --use_cores 4 --experiment_name MNIST_seed@0 --dataset_name MNIST --image_size 28
python3 train.py --use_gpu 1 --seed 0 --use_cores 4 --experiment_name KMNIST_seed@0 --dataset_name KMNIST --image_size 28
python3 train.py --use_gpu 2 --seed 0 --use_cores 4 --experiment_name FashionMNIST_seed@0 --dataset_name FashionMNIST --image_size 28
python3 train.py --use_gpu 3 --seed 0 --use_cores 4 --experiment_name SVHN_seed@0 --dataset_name SVHN --image_size 32
python3 train.py --use_gpu 0 --seed 0 --use_cores 4 --experiment_name CIFAR-10_seed@0 --dataset_name CIFAR-10 --image_size 32
python3 train.py --use_gpu 1 --seed 0 --use_cores 4 --experiment_name CIFAR-100_seed@0 --dataset_name CIFAR-100 --image_size 32
python3 train.py --use_gpu 2 --seed 0 --use_cores 4 --experiment_name STL-10_seed@0 --dataset_name STL-10 --image_size 96
python3 train.py --use_gpu 3 --seed 0 --use_cores 4 --experiment_name STL-10_seed@0_is@32 --dataset_name STL-10 --image_size 32
```
