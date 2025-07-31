
import torch
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize



def create_dataloaders(transform, batch_size):
    """Creates training and testing DataLoaders."""
    # Download and load the dataset
    dataset = OxfordIIITPet(root='./data', transform=transform, download=True)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create the DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, test_dataloader, dataset.classes