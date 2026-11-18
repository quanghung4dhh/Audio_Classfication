import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

HISTORY_PATH = (
    ROOT / "outputs" / "reports" / "training_history.json"
)

FIGURE_DIR = ROOT / "outputs" / "figures"


def main():

    # Load training history
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = history["epoch"]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ==========================================
    # Loss Curve
    # ==========================================

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # ==========================================
    # Accuracy Curve
    # ==========================================

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
    )

    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Save figure
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure_path = FIGURE_DIR / "cnn_learning_curves.png"

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Learning curves saved at:", figure_path)


if __name__ == "__main__":
    main()
