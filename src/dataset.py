from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import (
    load_fixed_audio,
    waveform_to_log_mel,
)


class UrbanSoundMelDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        data_root: str | Path,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        audio_path = (
            self.data_root
            / f"fold{int(row['fold'])}"
            / row["slice_file_name"]
        )

        waveform = load_fixed_audio(audio_path)

        log_mel = waveform_to_log_mel(waveform)

        # CNN input shape: [channel, mel_bins, time_frames]
        features = torch.from_numpy(log_mel).unsqueeze(0)

        # CrossEntropyLoss expects integer class indices.
        label = torch.tensor(
            int(row["classID"]),
            dtype=torch.long,
        )

        return features, label
