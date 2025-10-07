import torch
import torchvision

def download_cifar10(data_path):
    trainset = torchvision.datasets.CIFAR10(
        root=data_path, 
        train=True,
        download=True, 
    )

    return trainset


