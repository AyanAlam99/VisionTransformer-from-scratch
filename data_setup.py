# data_setup.py

import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

def create_dataloaders(train_transform, test_transform, batch_size):
    """Creates training and testing DataLoaders for CIFAR-100."""
    
    train_dataset = CIFAR100(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = CIFAR100(root='./data', train=False, transform=test_transform, download=True)
  
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        
        pin_memory=True
    )

    return train_dataloader, test_dataloader, train_dataset.classes