
"""
Utility functions for the Audio Classification project.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds to improve reproducibility.

    Args:
        seed: Random seed value.
    """

    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch CPU random seed
    torch.manual_seed(seed)

    # PyTorch CUDA random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
