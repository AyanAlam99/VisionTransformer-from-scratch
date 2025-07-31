import torch
from torch import nn
from tqdm.auto import tqdm

# Function for a single training step
def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_func,
               device: torch.device):
    """Performs a single training step."""
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        # Send data to the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred_logits = model(X)

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()

        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        train_acc += accuracy_func(y_pred_class, y).item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

# Function for a single testing step
def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              accuracy_func,
              device: torch.device):
    """Performs a single evaluation step."""
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred_logits = model(X)

            # 2. Calculate loss and accuracy
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(y_pred_logits, dim=1)
            test_acc += accuracy_func(y_pred_class, y).item()

    # Calculate average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc




