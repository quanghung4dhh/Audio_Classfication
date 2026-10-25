from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import UrbanSoundMelDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "archive"

metadata = pd.read_csv(
    DATA_ROOT / "UrbanSound8K.csv"
)

# Small test subset: enough to validate the pipeline.
test_metadata = metadata[
    metadata["fold"] == 1
].head(8)

dataset = UrbanSoundMelDataset(
    metadata=test_metadata,
    data_root=DATA_ROOT,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

features, labels = next(iter(loader))

# In toàn bộ tensor của mẫu đầu tiên
print("\nTensor của mẫu đầu tiên:")
print(features[0])

# In một phần nhỏ để dễ đọc
print("\n5 dải Mel đầu tiên, 10 khung thời gian đầu tiên:")
print(features[0, 0, :5, :10])

# In nhãn của mẫu đầu tiên
print("\nLabel:", labels[0].item())

print("Batch features:", features.shape)
print("Batch labels:", labels.shape)
print("Labels:", labels.tolist())
print("Features dtype:", features.dtype)
print("Labels dtype:", labels.dtype)
print("Feature values finite:", features.isfinite().all().item())
