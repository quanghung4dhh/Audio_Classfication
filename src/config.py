
"""
Centralized configuration for Audio Classification project.

This file contains the hyperparameters and settings
used throughout the training and evaluation pipeline.
"""

# ==================================================
# 1. Dataset Configuration
# ==================================================

NUM_CLASSES = 10

SAMPLE_RATE = 16000
DURATION = 4.0
N_MELS = 64


# ==================================================
# 2. Model Configuration
# ==================================================

DROPOUT = 0.3


# ==================================================
# 3. Training Configuration
# ==================================================

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

NUM_EPOCHS = 30

PATIENCE = 7


# ==================================================
# 4. Reproducibility Configuration
# ==================================================

SEED = 42
