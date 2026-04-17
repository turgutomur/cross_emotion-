"""
Seed and determinism utilities.

Full determinism is critical for the paper's statistical claims
(5 seed × multiple methods × 3 targets = ~60+ runs). This module
centralizes seed control so every experiment is reproducible.
"""
import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds for python, numpy, and torch (CPU + CUDA).

    Args:
        seed: integer seed.
        deterministic: if True, disable cuDNN nondeterminism. Slower
            but required for fully reproducible training runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch >= 1.8 extra knob; may raise if some op lacks a
            # deterministic implementation — we warn rather than fail.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    except ImportError:
        pass


# Canonical seeds used in the paper. Fixed so results are comparable across
# methods; changing the list invalidates cross-method comparisons.
PAPER_SEEDS = [42, 123, 456, 789, 2024]


def get_paper_seeds(n: int = 3) -> list:
    """Return the first `n` canonical seeds."""
    if n > len(PAPER_SEEDS):
        raise ValueError(
            f"Requested {n} seeds but only {len(PAPER_SEEDS)} are canonical. "
            f"Extend PAPER_SEEDS if more are needed (and rerun all methods)."
        )
    return PAPER_SEEDS[:n]
