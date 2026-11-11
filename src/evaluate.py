
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

from src.cnn import AudioCNN
from src.data import build_dataloaders


# ==========================================
# Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "best_cnn.pth"

FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

BATCH_SIZE = 32
NUM_CLASSES = 10


# ==========================================
# Load model
# ==========================================

def load_model(device):

    model = AudioCNN(num_classes=NUM_CLASSES).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["val_loss"])
    print("Validation accuracy:", checkpoint["val_accuracy"])

    return model


# ==========================================
# Evaluate on test set
# ==========================================

def evaluate(model, loader, device):

    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)

            outputs = model(features)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    return y_true, y_pred


# ==========================================
# Main
# ==========================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load test DataLoader
    loaders = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    test_loader = loaders["test"]

    print("Test samples:", len(test_loader.dataset))

    # Load best checkpoint
    model = load_model(device)

    # Predict
    y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
    )

    # Class names ordered by classID
    metadata = pd.read_csv(ROOT / "archive" / "UrbanSound8K.csv")

    class_names = (
        metadata[["classID", "class"]]
        .drop_duplicates()
        .sort_values("classID")["class"]
        .tolist()
    )

    labels = list(range(NUM_CLASSES))

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("CNN TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1:{weighted_f1: .4f}")

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("CNN Baseline - UrbanSound8K Test Confusion Matrix")
    fig.tight_layout()

    figure_path = FIGURE_DIR / "cnn_confusion_matrix.png"

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Save text report
    report_path = REPORT_DIR / "cnn_test_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("CNN BASELINE TEST RESULTS\n")
        file.write("=" * 50 + "\n")

        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1: {macro_f1:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")

        file.write(report)

    print("\nSaved confusion matrix:", figure_path)
    print("Saved classification report:", report_path)


if __name__ == "__main__":
    main()
