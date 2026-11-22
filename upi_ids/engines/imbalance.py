"""Class-imbalance mitigation applied strictly inside the training partition (Section V-B)."""
from __future__ import annotations

import numpy as np


def resample_training(X: np.ndarray, y: np.ndarray, use_smote: bool = True, sampling_strategy: float = 0.1,
                      k_neighbors: int = 5, seed: int = 42):
    """SMOTE over-sampling of the minority class. Never call on validation/test data."""
    if not use_smote:
        return X, y
    from imblearn.over_sampling import SMOTE

    n_min = int(y.sum())
    k = min(k_neighbors, max(1, n_min - 1))
    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=seed)
    return sm.fit_resample(X, y)


def positive_class_weight(y: np.ndarray) -> float:
    """scale_pos_weight = N_neg / N_pos (class-weighted loss)."""
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)
