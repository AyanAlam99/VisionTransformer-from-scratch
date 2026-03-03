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

            # Handle accuracy calculation for both hard and soft labels
            y_pred_class = torch.argmax(y_pred_logits, dim=1)
            
            # If y is 2D (soft labels from Mixup/Cutmix), convert to 1D class indices for accuracy calculation
            y_true_class = torch.argmax(y, dim=1) if y.ndim == 2 else y
            
            train_acc += accuracy_func(y_pred_class, y_true_class).item()

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




def train(model, train_dataloader , test_dataloader,optimizer,loss_fn,accuracy_func ,epochs,device , scheduler=None ) :
          


    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    print("Starting training...")
    print("-" * 60)

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
        
        if scheduler is not None:
            scheduler.step() # Step the scheduler after each epoch
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

    
        # --- DETAILED LOGGING ---
        print(f"Epoch: {epoch+1:02d}")
        print(f"\tTrain Loss: {train_loss:.5f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tTest Loss:  {test_loss:.5f} | Test Acc:  {test_acc*100:.2f}%")
        print("-" * 60)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    print(" Training finished.")
    return results