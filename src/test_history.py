import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "outputs" / "reports" / "training_history.json"


def main():

    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    print("=" * 60)
    print("TRAINING HISTORY TEST")
    print("=" * 60)

    print("Total epochs:", len(history["epoch"]))

    print("\nAvailable metrics:")

    for key, values in history.items():
        print(f"{key}: {len(values)} values")

    print("\nLast epoch results:")

    for key, values in history.items():
        print(f"{key}: {values[-1]}")

    # Validate history consistency
    lengths = [len(values) for values in history.values()]

    assert len(set(lengths)) == 1, (
        "History arrays have inconsistent lengths!"
    )

    assert len(history["epoch"]) > 0, (
        "Training history is empty!"
    )

    print("\nTraining history test passed!")


if __name__ == "__main__":
    main()
