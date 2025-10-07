import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CIFAR10_dataset(Dataset):
    def __init__(self, trainset, img_transform = True):
        self.trainset = trainset
        self.img_transform = img_transform

        if self.img_transform:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # Random cropping with size 32×32 and padding 4
                transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with probability 0.5
                transforms.ToTensor(),  # Convert PIL Image to tensor
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),  # Mean for each RGB channel
                    std=(0.2023, 0.1994, 0.2010)    # Standard deviation (variance) for each RGB channel
                )
            ])
        else: 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                )
            ])

    def __len__(self):
        return len(self.trainset)
    
    def __getitem__(self, idx):
        img, label = self.trainset[idx]

        if self.transform is not None:
            img_tensor = self.transform(img) 

        return img_tensor, label
        

        
