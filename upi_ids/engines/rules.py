"""Deterministic rule-based detection engine (Section III-E-1).

Each rule returns a severity in [0, 1]; the engine's score is the maximum severity
(a single hard policy violation is enough to flag) plus a small bonus for multiple
simultaneous violations, clipped to 1."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RuleResult:
    score: float
    triggered: list[str] = field(default_factory=list)


class RuleEngine:
    def __init__(self, max_amount: float = 50_000, burst_count_1hr: int = 8, max_failed_attempts: int = 3,
                 impossible_speed_kmph: float = 900.0, far_distance_km: float = 500.0):
        self.max_amount = max_amount
        self.burst_count_1hr = burst_count_1hr
        self.max_failed_attempts = max_failed_attempts
        self.impossible_speed_kmph = impossible_speed_kmph
        self.far_distance_km = far_distance_km

    def evaluate_one(self, f: dict) -> RuleResult:
        hits: list[tuple[str, float]] = []
        if f["Txn_Amount"] >= self.max_amount:
            hits.append(("EXCESSIVE_AMOUNT", 1.0))
        elif f["Txn_Amount"] >= 0.5 * self.max_amount:
            hits.append(("HIGH_AMOUNT", 0.5))
        if f["Txn_Count_1Hr"] >= self.burst_count_1hr:
            hits.append(("TXN_BURST", 0.9))
        if f["Account_History_Idx"] >= self.max_failed_attempts:
            hits.append(("REPEATED_AUTH_FAILURE", 0.8))
        gap_h = max(f["Inter_Txn_Gap"], 1.0) / 3600.0
        if f["Distance_From_Home"] > 50 and f["Distance_From_Home"] / gap_h > self.impossible_speed_kmph:
            hits.append(("IMPOSSIBLE_TRAVEL", 1.0))
        elif f["Distance_From_Home"] >= self.far_distance_km:
            hits.append(("FAR_FROM_HOME", 0.6))
        if f["Device_Fingerprint"] >= 1.0 and f["z_amt"] > 3.0:
            hits.append(("NEW_DEVICE_HIGH_AMOUNT", 0.9))
        if not hits:
            return RuleResult(0.0, [])
        sev = max(s for _, s in hits) + 0.1 * (len(hits) - 1)
        return RuleResult(float(min(sev, 1.0)), [n for n, _ in hits])

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.evaluate_one(r).score for r in X.to_dict("records")], dtype=float)

    def predict(self, X: pd.DataFrame, threshold: float = 0.9) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)
