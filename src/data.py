
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import UrbanSoundMelDataset


# Project root
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "archive"

# Preliminary split for CNN baseline development.
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLDS = [9]
TEST_FOLDS = [10]


def build_dataloaders(
    data_root: str | Path = DATA_ROOT,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Build train, validation and test DataLoaders
    using the predefined UrbanSound8K folds.
    """

    data_root = Path(data_root)
    csv_path = data_root / "UrbanSound8K.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Cannot find metadata CSV: {csv_path}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    # Read metadata
    metadata = pd.read_csv(csv_path)

    required_columns = {
        "slice_file_name",
        "fold",
        "classID",
        "class",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Validate the predefined folds
    available_folds = set(metadata["fold"].unique())

    if available_folds != set(range(1, 11)):
        raise ValueError(
            f"Unexpected fold configuration: {available_folds}"
        )

    # Validate class IDs
    class_ids = sorted(metadata["classID"].unique())

    if class_ids != list(range(10)):
        raise ValueError(
            f"Expected class IDs 0-9, got: {class_ids}"
        )

    mapping_errors = (
        metadata.groupby("classID")["class"].nunique() > 1
    )

    if mapping_errors.any():
        raise ValueError("Inconsistent class ID mapping")

    # Split metadata by predefined folds
    train_df = metadata[
        metadata["fold"].isin(TRAIN_FOLDS)
    ].copy()

    val_df = metadata[
        metadata["fold"].isin(VAL_FOLDS)
    ].copy()

    test_df = metadata[
        metadata["fold"].isin(TEST_FOLDS)
    ].copy()

    # Create Dataset objects
    train_dataset = UrbanSoundMelDataset(
        metadata=train_df,
        data_root=data_root,
    )

    val_dataset = UrbanSoundMelDataset(
        metadata=val_df,
        data_root=data_root,
    )

    test_dataset = UrbanSoundMelDataset(
        metadata=test_df,
        data_root=data_root,
    )

    # Use pinned memory when CUDA is available
    pin_memory = torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


if __name__ == "__main__":

    loaders = build_dataloaders()

    print("=" * 50)
    print("DATALOADER CHECK")
    print("=" * 50)

    for split_name, loader in loaders.items():

        features, labels = next(iter(loader))

        print(f"\n[{split_name.upper()}]")
        print("Dataset size:", len(loader.dataset))
        print("Batch features:", features.shape)
        print("Batch labels:", labels.shape)
        print("Feature dtype:", features.dtype)
        print("Label dtype:", labels.dtype)
        print("Feature finite:", features.isfinite().all().item())
        print("Label range:", labels.min().item(), "-", labels.max().item())
