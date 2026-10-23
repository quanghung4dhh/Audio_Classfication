# Week 1 — Data and Pipeline Preparation

## 1. Dataset

- Dataset: UrbanSound8K
- WAV files found: 8732
- Metadata: archive/UrbanSound8K.csv
- Number of classes: 10
- Number of folds: 10

## 2. Exploratory Data Analysis

- Class distribution:
  outputs/figures/class_distribution.png
- Audio visualization:
  outputs/figures/sample_audio_analysis.png

## 3. Preprocessing

- Channels: mono
- Target sample rate: 16000 Hz
- Target duration: 4 seconds
- CNN representation: Log-Mel Spectrogram
- Number of Mel bins: 64

## 4. PyTorch Data Pipeline

- Dataset implementation: src/dataset.py
- DataLoader test: src/test_dataloader.py
- Expected input shape: [batch, 1, 64, 201]

## 5. Issues and Next Steps

- Record any remaining data or pipeline errors.
- Next: build and train the CNN baseline.
