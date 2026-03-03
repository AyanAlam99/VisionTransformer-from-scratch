import torch
from torch import nn

from torchvision.transforms import v2 
import torchmetrics

from config import Config as config
import model
import data_setup
import engine
import utils
from torch.optim.lr_scheduler import CosineAnnealingLR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Updated to use v2 transforms
train_transform = v2.Compose([
    v2.ToImage(), # Standardize internal format
    v2.RandomResizedCrop(size=(config["image_size"], config["image_size"]), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True), # Converts to float and scales to [0, 1]
    v2.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((config["image_size"], config["image_size"]), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=config["batch_size"],
    num_classes=config["num_classes"] # Pass num_classes for Mixup/Cutmix
)

vit_model = model.VitForClassification(config).to(device)

# CrossEntropyLoss natively supports soft labels (probabilities) from Mixup
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) # Added label smoothing as an extra layer of defense

# Removed the duplicate optimizer definition
optimizer = torch.optim.AdamW(vit_model.parameters(), 
                                lr=config["learning_rate"], 
                                weight_decay=config["weight_decay"])

accuracy_func = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device)

scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])


results = engine.train(model=vit_model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       accuracy_func=accuracy_func,
                       epochs=config["num_epochs"],
                       device=device, 
                       scheduler=scheduler)

utils.plot_loss_curves(results)