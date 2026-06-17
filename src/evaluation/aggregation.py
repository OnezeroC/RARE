from __future__ import annotations

import numpy as np


def mean_metric(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values()))) if values else 0.0
