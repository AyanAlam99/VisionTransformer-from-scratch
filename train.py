# train.py

import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchmetrics


from config import config
import model
import data_setup
import engine
import utils

#  Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup transforms
transform = Compose([
    Resize((config["image_size"], config["image_size"])),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3)
])

#  Create DataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    transform=transform,
    batch_size=config["batch_size"]
)

#  Initialize model
vit_model = model.VitForClassification(config).to(device)

#  Setup loss, optimizer, and accuracy function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vit_model.parameters(), 
                              lr=config["learning_rate"], 
                              weight_decay=config["weight_decay"])
accuracy_func = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device)

#  Start the training
results = engine.train(model=vit_model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       accuracy_func=accuracy_func,
                       epochs=config["num_epochs"],
                       device=device)

#  Plot the results
utils.plot_loss_curves(results)