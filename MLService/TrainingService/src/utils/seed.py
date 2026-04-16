from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Seed Python and NumPy for reproducibility in classical ML workflows.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)