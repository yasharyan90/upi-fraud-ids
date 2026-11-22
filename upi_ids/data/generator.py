"""Configurable synthetic UPI transaction generator (Section III-B of the paper).

Emits payment payloads with device metadata, IP addresses, geo-coordinates and
labelled synthetic fraud scenarios for controlled evaluation.  The generator is
modelled after the behaviour of public mobile-money benchmarks (PaySim) but adds
the behavioural / device / geospatial attributes the paper's feature space needs.

Fraud scenarios implemented
---------------------------
* account_takeover   : new device + new IP + large geographic jump + large amount
* velocity_burst     : many rapid small/medium transfers in a short window
* high_value_mule    : unusually large transfer to a never-seen receiver
* night_anomaly      : atypical hour + unknown device + above-normal amount
* subtle             : mild deviations on several axes (hard-to-detect fraud)
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MERCHANT_TYPES = ["grocery", "utility", "food", "transport", "shopping", "p2p", "gaming", "crypto", "wallet_load", "unknown"]
# Static merchant category risk encoding (contextual feature).
MERCHANT_RISK = {
    "grocery": 0.05, "utility": 0.05, "food": 0.08, "transport": 0.10, "shopping": 0.20,
    "p2p": 0.35, "wallet_load": 0.40, "gaming": 0.60, "crypto": 0.80, "unknown": 0.70,
}

RAW_COLUMNS = [
    "txn_id", "timestamp", "sender_vpa", "receiver_vpa", "amount", "merchant_type",
    "device_id", "ip_address", "lat", "lon", "auth_failures", "account_age_days",
    "is_fraud", "fraud_type",
]

# Engineered feature space (Table II of the paper).
FEATURE_COLUMNS = [
    "Txn_Amount", "z_amt", "Merchant_Type", "Txn_Count_1Hr", "Inter_Txn_Gap",
    "Distance_From_Home", "Device_Fingerprint", "Sender_Receiver_Rel", "Account_History_Idx",
    "Hour_Of_Day",
]

# Indian metro centroids used as user home locations.
CITIES = np.array([
    [28.6139, 77.2090], [19.0760, 72.8777], [12.9716, 77.5946], [13.0827, 80.2707],
    [22.5726, 88.3639], [17.3850, 78.4867], [23.0225, 72.5714], [18.5204, 73.8567],
    [26.9124, 75.7873], [21.1702, 72.8311], [26.8467, 80.9462], [28.4595, 77.0266],
])


def haversine_km(lat1, lon1, lat2, lon2):
    """Spherical Haversine distance in kilometres (vectorised)."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class UserProfile:
    uid: int
    vpa: str
    home_lat: float
    home_lon: float
    mean_amount: float
    std_amount: float
    devices: list[str]
    ips: list[str]
    account_age_days: int
    peers: list[str] = field(default_factory=list)
    merchant_pref: np.ndarray | None = None


class UPITransactionGenerator:
    """Synthesises labelled UPI transaction streams with realistic class imbalance."""

    def __init__(self, n_users: int = 2500, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.users = self._make_users()

    # ------------------------------------------------------------------ users
    def _make_users(self) -> list[UserProfile]:
        users = []
        for uid in range(self.n_users):
            city = CITIES[self.rng.integers(len(CITIES))]
            lat = city[0] + self.rng.normal(0, 0.05)
            lon = city[1] + self.rng.normal(0, 0.05)
            mean_amt = float(np.exp(self.rng.normal(6.2, 0.7)))  # ~ INR 500 median
            std_amt = mean_amt * float(self.rng.uniform(0.4, 1.0))
            n_dev = int(self.rng.integers(1, 3))
            devices = [self._hash(f"dev-{uid}-{k}") for k in range(n_dev)]
            ips = [f"10.{self.rng.integers(0,255)}.{self.rng.integers(0,255)}.{self.rng.integers(1,255)}" for _ in range(n_dev)]
            pref = self.rng.dirichlet(np.ones(len(MERCHANT_TYPES)) * np.array([6, 4, 5, 4, 4, 5, 0.6, 0.3, 2, 0.4]))
            users.append(UserProfile(uid, f"user{uid}@upi", lat, lon, mean_amt, std_amt, devices, ips,
                                     int(self.rng.integers(5, 2000)), merchant_pref=pref))
        for u in users:
            k = int(self.rng.integers(2, 9))
            u.peers = [users[i].vpa for i in self.rng.choice(self.n_users, size=k, replace=False) if i != u.uid]
        return users

    @staticmethod
    def _hash(s: str) -> str:
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    # ------------------------------------------------------------- generation
    def generate(self, n: int, fraud_count: int | None = None, fraud_rate: float = 0.01,
                 start_ts: float = 1_700_000_000.0, horizon_days: int = 30) -> pd.DataFrame:
        """Return a DataFrame of `n` transactions with exactly `fraud_count` frauds
        (or approximately `fraud_rate` if fraud_count is None), sorted by time."""
        if fraud_count is None:
            fraud_count = int(round(n * fraud_rate))
        n_legit = n - fraud_count
        legit = self._legit(n_legit, start_ts, horizon_days)
        fraud = self._fraud(fraud_count, start_ts, horizon_days)
        df = pd.concat([legit, fraud], ignore_index=True)
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        df["txn_id"] = [f"TXN{start_ts:.0f}{i:07d}" for i in range(len(df))]
        return df[RAW_COLUMNS]

    def _legit(self, n: int, start_ts: float, horizon_days: int) -> pd.DataFrame:
        rows = []
        uids = self.rng.integers(0, self.n_users, size=n)
        # daytime-heavy hour distribution
        hours = self.rng.choice(24, size=n, p=self._hour_probs())
        days = self.rng.integers(0, horizon_days, size=n)
        for i in range(n):
            u = self.users[uids[i]]
            amt = max(1.0, self.rng.normal(u.mean_amount, u.std_amount))
            if self.rng.random() < 0.03:  # occasional legitimately large bill
                amt *= self.rng.uniform(2, 5)
            ts = start_ts + days[i] * 86400 + hours[i] * 3600 + self.rng.uniform(0, 3600)
            k = int(self.rng.integers(len(u.devices)))
            far = self.rng.random() < 0.02  # legitimate travel
            lat = u.home_lat + self.rng.normal(0, 0.03 if not far else 2.0)
            lon = u.home_lon + self.rng.normal(0, 0.03 if not far else 2.0)
            mtype = MERCHANT_TYPES[self.rng.choice(len(MERCHANT_TYPES), p=u.merchant_pref)]
            recv = (self.rng.choice(u.peers) if (u.peers and mtype == "p2p" and self.rng.random() < 0.85)
                    else f"merchant{self.rng.integers(0, 4000)}@upi")
            rows.append(("", ts, u.vpa, recv, round(amt, 2), mtype, u.devices[k], u.ips[k], lat, lon,
                         int(self.rng.random() < 0.04), u.account_age_days, 0, "none"))
        return pd.DataFrame(rows, columns=RAW_COLUMNS)

    def _fraud(self, n: int, start_ts: float, horizon_days: int) -> pd.DataFrame:
        rows = []
        scenarios = ["account_takeover", "velocity_burst", "high_value_mule", "night_anomaly", "subtle"]
        probs = [0.28, 0.22, 0.18, 0.12, 0.20]
        while len(rows) < n:
            kind = str(self.rng.choice(scenarios, p=probs))
            u = self.users[self.rng.integers(self.n_users)]
            ts = start_ts + self.rng.uniform(0, horizon_days * 86400)
            dev, ip = u.devices[0], u.ips[0]
            lat, lon = u.home_lat + self.rng.normal(0, 0.03), u.home_lon + self.rng.normal(0, 0.03)
            fails, mtype = 0, "p2p"
            recv = f"mule{self.rng.integers(0, 900)}@upi"
            if kind == "account_takeover":
                dev, ip = self._hash(f"rogue-{self.rng.integers(1e9)}"), f"103.{self.rng.integers(0,255)}.{self.rng.integers(0,255)}.{self.rng.integers(1,255)}"
                city = CITIES[self.rng.integers(len(CITIES))]
                lat, lon = city[0] + self.rng.normal(0, 0.5), city[1] + self.rng.normal(0, 0.5)
                amt = u.mean_amount * self.rng.uniform(4, 15)
                fails = int(self.rng.integers(1, 6))
                mtype = str(self.rng.choice(["p2p", "crypto", "wallet_load", "unknown"]))
            elif kind == "velocity_burst":
                amt = u.mean_amount * self.rng.uniform(0.8, 3.0)
                mtype = str(self.rng.choice(["p2p", "gaming", "wallet_load"]))
                if self.rng.random() < 0.5:
                    dev = self._hash(f"rogue-{self.rng.integers(1e9)}")
                # burst = several transactions within a few minutes (bounded by the remaining fraud budget)
                burst_n = min(int(self.rng.integers(4, 9)), n - len(rows))
                for b in range(burst_n):
                    rows.append(("", ts + b * self.rng.uniform(20, 240), u.vpa, f"mule{self.rng.integers(0, 900)}@upi",
                                 round(amt * self.rng.uniform(0.7, 1.3), 2), mtype, dev, ip, lat, lon, fails,
                                 u.account_age_days, 1, kind))
                continue
            elif kind == "high_value_mule":
                amt = u.mean_amount * self.rng.uniform(8, 40)
                mtype = str(self.rng.choice(["p2p", "crypto", "unknown"]))
                fails = int(self.rng.integers(0, 3))
            elif kind == "night_anomaly":
                day = math.floor((ts - start_ts) / 86400)
                ts = start_ts + day * 86400 + self.rng.uniform(1, 4) * 3600  # 01:00-04:00
                dev = self._hash(f"rogue-{self.rng.integers(1e9)}")
                amt = u.mean_amount * self.rng.uniform(2, 6)
                mtype = str(self.rng.choice(["gaming", "crypto", "p2p"]))
            else:  # subtle: looks almost legitimate
                amt = u.mean_amount * self.rng.uniform(1.2, 2.5)
                mtype = str(self.rng.choice(["p2p", "shopping", "wallet_load"]))
                if self.rng.random() < 0.3:
                    dev = self._hash(f"rogue-{self.rng.integers(1e9)}")
                if self.rng.random() < 0.3:
                    lat += self.rng.normal(0, 1.0)
                    lon += self.rng.normal(0, 1.0)
            rows.append(("", ts, u.vpa, recv, round(amt, 2), mtype, dev, ip, lat, lon, fails,
                         u.account_age_days, 1, kind))
        return pd.DataFrame(rows[:n], columns=RAW_COLUMNS)

    def _hour_probs(self) -> np.ndarray:
        p = np.array([0.5, 0.3, 0.2, 0.2, 0.3, 0.8, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 5.5, 5.0, 4.5, 4.5,
                      5.0, 5.5, 6.0, 6.0, 5.5, 4.5, 3.0, 1.5])
        return p / p.sum()
