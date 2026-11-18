
from pathlib import Path

import json
import torch
from torch import nn
from torch.optim import Adam

from src.cnn import AudioCNN
from src.data import build_dataloaders

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    NUM_CLASSES,
    SEED,
)
from src.utils import set_seed


# ==========================================
# Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_cnn.pth"
REPORT_DIR = ROOT / "outputs" / "reports"
HISTORY_PATH = REPORT_DIR / "training_history.json"


# ==========================================
# Training one epoch
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate metrics
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# Validation
# ==========================================

def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation
    with torch.no_grad():

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# Main training process
# ==========================================

def main():

    # Set random seed for reproducibility
    set_seed(SEED)

    print(f"Random seed: {SEED}")
    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load datasets
    loaders = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Initialize model
    model = AudioCNN(num_classes=NUM_CLASSES).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Prepare checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Store metrics for every epoch
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        # Record training metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        # Save the best model based on validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,

                    # Save training configuration
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "num_epochs": NUM_EPOCHS,
                        "patience": PATIENCE,
                        "num_classes": NUM_CLASSES,
                        "seed": SEED,
                    },
                },
                BEST_MODEL_PATH,
            )

            print("  -> Best model saved!")

        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("=" * 60)
    print("Training completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {BEST_MODEL_PATH}")

    # Save training history
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=4,
        )

    print(f"Training history saved at: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
