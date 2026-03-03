import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2

def create_dataloaders(train_transform, test_transform, batch_size, num_classes=100):
    """Creates training and testing DataLoaders for CIFAR-100 with Mixup/Cutmix."""
    
    train_dataset = CIFAR100(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = CIFAR100(root='./data', train=False, transform=test_transform, download=True)
  
    
    cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
    mixup = v2.MixUp(alpha=0.2, num_classes=num_classes)
    
    # Randomly choose between Mixup and Cutmix for each batch
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
       
        return cutmix_or_mixup(*default_collate(batch))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn 
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
       
    )

    return train_dataloader, test_dataloader, train_dataset.classes