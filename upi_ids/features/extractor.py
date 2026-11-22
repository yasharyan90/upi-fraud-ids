"""Behavioural feature extraction (Section V-A, Table II).

Produces the engineered feature space from preprocessed transaction rows using a
streaming velocity cache, so the very same code path serves batch training and
real-time scoring.

Features
--------
Txn_Amount          payment value (INR)
z_amt               Z-score vs. the user's rolling 30-day mean / std
Merchant_Type       merchant category risk encoding
Txn_Count_1Hr       rolling 60-minute transaction frequency (prior events of the same user)
Inter_Txn_Gap       seconds since the user's previous UPI event
Distance_From_Home  Haversine km from the user's baseline location
Device_Fingerprint  1 if device unseen for this user, else 0
Sender_Receiver_Rel historical peer interaction score (log1p of prior count)
Account_History_Idx prior failed auth attempts combined with account tenure
Hour_Of_Day         local hour (temporal context)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data.generator import FEATURE_COLUMNS, haversine_km
from .velocity_cache import InMemoryVelocityCache


class FeatureExtractor:
    def __init__(self, cache: InMemoryVelocityCache | None = None, min_history: int = 3):
        self.cache = cache or InMemoryVelocityCache()
        self.min_history = min_history

    def extract_one(self, row: dict, update: bool = True) -> dict:
        user = row["sender_vpa"]
        ts = float(row["timestamp"])
        amt = float(row["amount"])
        lat, lon = row.get("lat", np.nan), row.get("lon", np.nan)
        c = self.cache

        mean, std, n = c.amount_stats(user, ts)
        if n >= self.min_history and std > 1e-6:
            z = (amt - mean) / std
        elif n >= 1:
            z = (amt - mean) / max(mean, 1.0)  # cold-start: relative deviation
        else:
            z = 0.0
        last = c.last_ts(user)
        gap = (ts - last) if last is not None else 86400.0 * 7
        home = c.home(user)
        if home is not None and lat == lat and lon == lon:
            dist = float(haversine_km(home[0], home[1], lat, lon))
        else:
            dist = 0.0
        new_dev = 0.0 if c.known_device(user, row["device_id"]) else 1.0
        if last is None:
            new_dev = 0.0  # first ever event: no device baseline yet
        rel = math.log1p(c.receiver_count(user, row["receiver_vpa"]))
        fails = float(row.get("auth_failures", 0) or 0)
        age = float(row.get("account_age_days", 365) or 365)
        hist_idx = fails + 1.0 / (1.0 + age / 90.0)  # young accounts and failed attempts increase index
        feats = {
            "Txn_Amount": amt,
            "z_amt": float(np.clip(z, -10, 50)),
            "Merchant_Type": float(row.get("merchant_risk", 0.5)),
            "Txn_Count_1Hr": float(c.count_window(user, ts, 3600.0)),
            "Inter_Txn_Gap": float(min(gap, 86400.0 * 7)),
            "Distance_From_Home": dist,
            "Device_Fingerprint": new_dev,
            "Sender_Receiver_Rel": rel,
            "Account_History_Idx": hist_idx,
            "Hour_Of_Day": float(int(ts // 3600) % 24),
        }
        if update:
            c.update(user, ts, amt, row["device_id"], row["receiver_vpa"], lat, lon)
        return feats

    def transform(self, df: pd.DataFrame, update: bool = True) -> pd.DataFrame:
        """Sequentially extract features for a time-ordered frame (no look-ahead leakage)."""
        recs = df.to_dict("records")
        feats = [self.extract_one(r, update=update) for r in recs]
        out = pd.DataFrame(feats, index=df.index)[FEATURE_COLUMNS]
        return out
