"""Concept-drift detection via Population Stability Index (PSI) per feature (Section III-H)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(len(ref), 1) + 1e-6
    c = np.histogram(cur, edges)[0] / max(len(cur), 1) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


class DriftMonitor:
    """Keeps a reference window (training distribution) and flags features whose PSI exceeds a threshold.
    PSI < 0.1: stable, 0.1-0.2: moderate shift, > 0.2: significant drift -> schedule retraining."""

    def __init__(self, reference: pd.DataFrame, threshold: float = 0.2, window: int = 2000):
        self.reference = reference.reset_index(drop=True)
        self.threshold = threshold
        self.window = window
        self._buffer: list[dict] = []

    def observe(self, features: dict) -> None:
        self._buffer.append(features)
        if len(self._buffer) > self.window:
            self._buffer = self._buffer[-self.window:]

    def report(self) -> dict:
        if len(self._buffer) < 50:
            return {"status": "warming_up", "n": len(self._buffer), "psi": {}, "drifted": []}
        cur = pd.DataFrame(self._buffer)
        scores = {c: psi(self.reference[c].values, cur[c].values) for c in self.reference.columns if c in cur}
        drifted = [c for c, v in scores.items() if v > self.threshold]
        return {"status": "drift" if drifted else "stable", "n": len(cur), "psi": scores, "drifted": drifted,
                "retrain_recommended": bool(drifted)}
