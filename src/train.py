
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.cnn import AudioCNN
from src.data import build_dataloaders


# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
PATIENCE = 7

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_cnn.pth"


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
    model = AudioCNN(num_classes=10).to(device)

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


if __name__ == "__main__":
    main()
