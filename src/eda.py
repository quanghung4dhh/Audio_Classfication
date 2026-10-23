from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "archive"
CSV_PATH = DATA_ROOT / "UrbanSound8K.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # 1. Class distribution
    counts = df["class"].value_counts().sort_index()

    ax = counts.plot(kind="bar", figsize=(11, 5))
    ax.set_title("UrbanSound8K: Samples per Class")
    ax.set_xlabel("Sound class")
    ax.set_ylabel("Number of samples")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    distribution_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(distribution_path, dpi=150)
    plt.close()

    # 2. Select one sample using its metadata.
    row = df.iloc[0]

    audio_path = (
        DATA_ROOT
        / f"fold{int(row['fold'])}"
        / row["slice_file_name"]
    )

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    info = sf.info(str(audio_path))

    y, sr = librosa.load(
        str(audio_path),
        sr=None,       # Preserve the original sample rate
        mono=True,     # Convert to mono for analysis
    )

    duration = len(y) / sr

    print("Example audio:", audio_path.name)
    print("Class:", row["class"])
    print("Sample rate:", sr)
    print("Original channels:", info.channels)
    print("Duration:", round(duration, 3), "seconds")

    # 3. FFT of a short segment.
    n_fft = 2048

    segment = y[:n_fft]

    if len(segment) < n_fft:
        segment = np.pad(
            segment,
            (0, n_fft - len(segment)),
        )

    windowed = segment * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1 / sr,
    )

    # 4. STFT
    hop_length = 512

    stft = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    )

    stft_db = librosa.amplitude_to_db(
        stft,
        ref=np.max,
    )

    # 5. Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
    )

    # Waveform
    librosa.display.waveshow(
        y,
        sr=sr,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Waveform")
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")

    # FFT
    axes[0, 1].plot(frequencies, spectrum)
    axes[0, 1].set_title("FFT Magnitude Spectrum")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")

    # STFT
    image1 = librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("STFT Spectrogram")
    fig.colorbar(image1, ax=axes[1, 0], format="%+2.0f dB")

    # Mel
    image2 = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Log-Mel Spectrogram")
    fig.colorbar(image2, ax=axes[1, 1], format="%+2.0f dB")

    fig.suptitle(
        f"Audio analysis: {row['class']}",
        fontsize=14,
    )
    fig.tight_layout()

    analysis_path = FIGURES_DIR / "sample_audio_analysis.png"
    fig.savefig(analysis_path, dpi=150)
    plt.close(fig)

    print("\nSaved figures:")
    print(distribution_path)
    print(analysis_path)


if __name__ == "__main__":
    main()
