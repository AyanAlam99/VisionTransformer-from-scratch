# train.py

import torch
from torch import nn
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize, 
                                    RandomCrop, RandomHorizontalFlip)
import torchmetrics

from config import Config as config
import model
import data_setup
import engine
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


train_transform = Compose([
    Resize((config["image_size"], config["image_size"])),
    RandomCrop(config["image_size"], padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]) # CIFAR-100 stats
])


test_transform = Compose([
    Resize((config["image_size"], config["image_size"])),
    ToTensor(),
    Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=config["batch_size"]
)

# Initialize model
vit_model = model.VitForClassification(config).to(device)

# Setup loss, optimizer, and accuracy function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vit_model.parameters(), 
                                lr=config["learning_rate"], 
                                weight_decay=config["weight_decay"])
accuracy_func = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device)

# Start the training
results = engine.train(model=vit_model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        accuracy_func=accuracy_func,
                        epochs=config["num_epochs"],
                        device=device)


utils.plot_loss_curves(results)