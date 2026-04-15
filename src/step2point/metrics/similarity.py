from __future__ import annotations

import numpy as np


def l1_distance(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(a - b)))
