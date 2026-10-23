from pathlib import Path

import pandas as pd


# Project root: Audio_Classfication/
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "archive"
CSV_PATH = DATA_ROOT / "UrbanSound8K.csv"

REQUIRED_COLUMNS = {
    "slice_file_name",
    "fold",
    "classID",
    "class",
}


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find metadata CSV: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Metadata path: {CSV_PATH}")
    print(f"Metadata rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nSamples per class:")
    print(df["class"].value_counts().sort_index())

    print("\nSamples per classID:")
    print(df["classID"].value_counts().sort_index())

    print("\nSamples per fold:")
    print(
        df["fold"].value_counts()
        .sort_index()
    )

    print("\nNumber of classes:", df["class"].nunique())
    print("Number of class IDs:", df["classID"].nunique())
    print("Number of folds:", df["fold"].nunique())

    # Check that a class ID consistently maps to one class name.
    label_mapping_errors = (
        df.groupby("classID")["class"].nunique() > 1
    ).sum()

    print("Inconsistent class ID mappings:", label_mapping_errors)

    # Check duplicated metadata entries.
    duplicates = df.duplicated(
        subset=["fold", "slice_file_name"]
    ).sum()

    print("Duplicated fold/filename entries:", duplicates)

    # Match each CSV row with its WAV path.
    expected_paths = []

    for row in df.itertuples(index=False):
        path = (
            DATA_ROOT
            / f"fold{int(row.fold)}"
            / row.slice_file_name
        )
        expected_paths.append(path.resolve())

    missing_files = [
        str(path)
        for path in expected_paths
        if not path.is_file()
    ]

    # WAV files on disk that are not referenced by the CSV.
    actual_paths = {
        path.resolve()
        for path in DATA_ROOT.glob("fold*/*.wav")
    }

    metadata_paths = set(expected_paths)
    unreferenced_files = actual_paths - metadata_paths

    fold_dirs = sorted(
        path.name
        for path in DATA_ROOT.glob("fold*")
        if path.is_dir()
    )

    print("\n" + "=" * 60)
    print("FILE INTEGRITY")
    print("=" * 60)

    print("Fold directories:", fold_dirs)
    print("WAV files found:", len(actual_paths))
    print("Missing WAV files:", len(missing_files))
    print("WAV files not referenced by CSV:", len(unreferenced_files))

    if missing_files:
        print("\nFirst missing files:")
        for path in missing_files[:10]:
            print(" -", path)

    if unreferenced_files:
        print("\nFirst unreferenced WAV files:")
        for path in list(sorted(unreferenced_files))[:10]:
            print(" -", path)

    if (
        len(df) == 8732
        and len(actual_paths) == 8732
        and not missing_files
        and not unreferenced_files
        and df["class"].nunique() == 10
        and df["fold"].nunique() == 10
        and label_mapping_errors == 0
        and duplicates == 0
    ):
        print("\nRESULT: BASIC DATASET CHECKS PASSED")
    else:
        print(
            "\nRESULT: CHECK THE WARNINGS ABOVE "
            "BEFORE CONTINUING"
        )


if __name__ == "__main__":
    main()
