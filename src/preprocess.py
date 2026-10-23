from pathlib import Path

import librosa
import numpy as np


def load_fixed_audio(
    path: str | Path,
    sample_rate: int = 16000,
    duration: float = 4.0,
) -> np.ndarray:
    """
    Load audio as mono, resample it, and return
    a fixed-length float32 waveform.
    """
    waveform, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=True,
    )

    target_length = round(sample_rate * duration)

    if len(waveform) > target_length:
        waveform = waveform[:target_length]

    elif len(waveform) < target_length:
        padding = target_length - len(waveform)
        waveform = np.pad(
            waveform,
            (0, padding),
        )

    return waveform.astype(np.float32)


def waveform_to_log_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 64,
) -> np.ndarray:
    """
    Convert a waveform into a log-Mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    return log_mel.astype(np.float32)
