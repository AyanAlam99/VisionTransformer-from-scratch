# Vision Transformer (ViT) from Scratch on CIFAR-100

This repository contains a complete, from-scratch PyTorch implementation of the Vision Transformer (ViT) architecture. The model is trained entirely from scratch (no pre-trained weights) on the CIFAR-100 dataset.

Training a ViT from scratch on a small dataset like CIFAR-100 is notoriously difficult because Transformers lack the spatial inductive biases (like translation invariance and locality) inherent to Convolutional Neural Networks (CNNs). This project demonstrates how to overcome the subsequent severe overfitting using aggressive regularization and modern data augmentation techniques.

## 🧠 Model Architecture
The architecture strictly follows the original ViT design principles, implemented completely from scratch:
* **Patch Embeddings:** Images are broken down into patches (8x8) and linearly projected into a 192-dimensional hidden space.
* **CLS Token & Positional Embeddings:** Learnable parameters added to sequence representations.
* **Transformer Encoder:** 6 hidden layers with Multi-Head Self-Attention (6 heads).
* **Classification Head:** MLP head mapping the output of the CLS token to 100 classes.

## 🛠️ Training Strategy & Regularization
To combat the ViT's tendency to memorize small datasets, the training pipeline utilizes a heavy regularization scheme:
* **MixUp & CutMix Augmentations:** Implemented at the batch level via `torchvision.transforms.v2`. This forces the model to learn soft probability distributions rather than hard labels, effectively destroying the capacity to memorize pixel patterns.
* **Label Smoothing:** Applied to the Cross-Entropy Loss (`label_smoothing=0.1`) to prevent overconfidence.
* **Cosine Annealing LR Scheduler:** Smoothly decays the learning rate from `3e-4` to 0 over the course of the training run, allowing for stable convergence.
* **High Weight Decay:** Set to `0.2` using the `AdamW` optimizer.
* **High Dropout:** `hidden_dropout_prob` and `attention_probs_dropout_prob` set to `0.1`.

## 📂 Project Structure
* `config.py`: Centralized configuration dictionary controlling hyperparameters (patch size, hidden layers, batch size, etc.).
* `data_setup.py`: DataLoader configuration, integrating `v2` transforms and a custom `collate_fn` for dynamic MixUp/CutMix batch generation.
* `model.py`: The core ViT architecture built using `torch.nn` modules.
* `engine.py`: Training and testing step functions, equipped to handle soft-labels dynamically.
* `train_test.py` / `train.py`: Main execution script to initialize the model, dataloaders, and trigger the training loop.
* `utils.py`: Helper functions for visualizing loss and accuracy curves.

## 📊 Results

The model was trained for 200 epochs. Due to the aggressive application of MixUp and CutMix on the training set, the training accuracy naturally stays lower than the test accuracy. This is a strong indicator of excellent generalization and successful mitigation of overfitting.

![Training Results](./training_results.png)

* **Final Test Accuracy:** ~53.5%
* **Final Test Loss:** ~2.30

Achieving >50% test accuracy on CIFAR-100 with a ViT completely from scratch (without distillation or massive pre-training) is a solid baseline that proves the model is learning generalized contextual features rather than memorizing data.

## 🚀 Usage

Ensure you have the required dependencies installed:
```bash
pip install torch torchvision torchmetrics matplotlib tqdm einops
