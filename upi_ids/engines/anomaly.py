"""Unsupervised anomaly detection engine: Isolation Forest (Section III-E-3).

The normalised anomaly score s(x, n) = 2^(-E[h(x)] / c(n)) is mapped to [0, 1]
(scikit-learn's `score_samples` returns -s, so we negate and rescale against the
training distribution so that the risk engine receives a calibrated severity)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyEngine:
    def __init__(self, n_estimators: int = 200, contamination: float = 0.01, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                     random_state=seed, n_jobs=4)
        self._lo = 0.0
        self._hi = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        """Fit on (predominantly) legitimate traffic. Labels are only used to drop known frauds
        from the normal profile when they are available."""
        Xn = X if y is None else X[np.asarray(y) == 0]
        Z = self.scaler.fit_transform(Xn)
        self.model.fit(Z)
        s = -self.model.score_samples(Z)  # higher = more anomalous
        self._lo, self._hi = float(np.percentile(s, 50)), float(np.percentile(s, 99.9))
        return self

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self.scaler.transform(X))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Severity in [0, 1]: 0 at/below the median normal score, 1 at the 99.9th percentile."""
        s = self.raw_scores(X)
        return np.clip((s - self._lo) / max(self._hi - self._lo, 1e-9), 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.model.predict(self.scaler.transform(X)) == -1).astype(int)
