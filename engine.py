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


# Add this function to your engine.py file

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          accuracy_func,
          epochs: int,
          device: torch.device):
    """The main training loop that calls train_step and test_step."""
    
    # Create a dictionary to store results
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_func=accuracy_func,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_func=accuracy_func,
                                        device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Test Acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results




