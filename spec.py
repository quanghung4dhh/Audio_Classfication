import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("QtAgg")


audioPath = "./archive/fold1/7383-3-0-0.wav"

y, sr = librosa.load(audioPath, sr=None)


# Raw signal


def raw():
    plt.plot(y)
    plt.title('Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()

# Using FFT


def FFT():
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft+1))
    plt.plot(ft)
    plt.title('Spectrum')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.show()

# Using STFT


def STFT():
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


# Mel Spectrogram
def Mel():
    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# All 4


def plot_all():

    # 1. Raw signal

    n_fft = 2048

    x = y[:n_fft]

    # 2. FFT

    ft = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(n_fft, d=1/sr)

    # 3. STFT

    spec = np.abs(
        librosa.stft(
            y,
            n_fft=2048,
            hop_length=512
        )
    )

    spec_db = librosa.amplitude_to_db(
        spec,
        ref=np.max
    )

    # 4. Mel Spectrogram

    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=1024
    )

    mel_spect_db = librosa.power_to_db(
        mel_spect,
        ref=np.max
    )

    # Create 2x2 plots

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 10)
    )

    # Raw signal

    librosa.display.waveshow(
        y,
        sr=sr,
        ax=axes[0, 0]
    )

    axes[0, 0].set_title("Raw Signal")
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")

    # FFT

    axes[0, 1].plot(freq, ft)

    axes[0, 1].set_title("Frequency Spectrum")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Amplitude")

    # STFT

    img1 = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="log",
        ax=axes[1, 0]
    )

    axes[1, 0].set_title("Spectrogram")

    fig.colorbar(
        img1,
        ax=axes[1, 0],
        format="%+2.0f dB"
    )

    # Mel Spectrogram

    img2 = librosa.display.specshow(
        mel_spect_db,
        sr=sr,
        hop_length=1024,
        y_axis="mel",
        x_axis="time",
        fmax=8000,
        ax=axes[1, 1]
    )

    axes[1, 1].set_title("Mel Spectrogram")

    fig.colorbar(
        img2,
        ax=axes[1, 1],
        format="%+2.0f dB"
    )

    # =========================
    # Layout
    # =========================

    fig.suptitle(
        "Audio Signal Analysis",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# raw()
# FFT()
# STFT()
Mel()
# plot_all()