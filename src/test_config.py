
import torch

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    NUM_CLASSES,
    SEED,
)

from src.utils import set_seed


def main():

    print("===== CONFIGURATION TEST =====")

    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Seed: {SEED}")

    print("\n===== RANDOM SEED TEST =====")

    set_seed(SEED)
    first = torch.rand(5)

    set_seed(SEED)
    second = torch.rand(5)

    print("First:", first)
    print("Second:", second)

    assert torch.equal(first, second), \
        "Random Seed test failed!"

    print("\nRandom Seed test passed!")


if __name__ == "__main__":
    main()
