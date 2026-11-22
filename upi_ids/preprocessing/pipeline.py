"""Deterministic preprocessing (Section III-D): schema validation, PII anonymisation,
missing-value imputation, duplicate suppression and numerical encoding."""
from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pandas as pd

from ..data.generator import MERCHANT_RISK, RAW_COLUMNS

REQUIRED = ["txn_id", "timestamp", "sender_vpa", "receiver_vpa", "amount", "merchant_type",
            "device_id", "ip_address", "lat", "lon"]

_PII_KEY = b"upi-ids-anonymisation-key"  # in production: HSM-backed secret


def anonymize_value(value: str, key: bytes = _PII_KEY) -> str:
    """Keyed one-way hash (HMAC-SHA256) so identifiers stay joinable but not reversible."""
    return hmac.new(key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()[:16]


class SchemaError(ValueError):
    pass


class Preprocessor:
    """Validate, anonymise, impute, de-duplicate and encode raw transaction payloads."""

    PII_FIELDS = ("sender_vpa", "receiver_vpa", "device_id", "ip_address")

    def __init__(self, anonymize: bool = True):
        self.anonymize = anonymize
        self._seen_ids: set[str] = set()

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise SchemaError(f"payload missing required fields: {missing}")
        if (pd.to_numeric(df["amount"], errors="coerce") < 0).any():
            raise SchemaError("negative transaction amount")

    def transform(self, df: pd.DataFrame, stateful_dedup: bool = False) -> pd.DataFrame:
        self.validate(df)
        out = df.copy()
        # duplicate suppression (idempotent ingestion)
        out = out.drop_duplicates(subset="txn_id", keep="first")
        if stateful_dedup:
            out = out[~out["txn_id"].isin(self._seen_ids)]
            self._seen_ids.update(out["txn_id"].tolist())
        # numeric coercion + imputation
        for col in ("amount", "lat", "lon", "auth_failures", "account_age_days", "timestamp"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out["amount"] = out["amount"].fillna(out["amount"].median() if len(out) else 0.0)
        out["auth_failures"] = out.get("auth_failures", pd.Series(0, index=out.index)).fillna(0).astype(int)
        out["account_age_days"] = out.get("account_age_days", pd.Series(365, index=out.index)).fillna(365).astype(int)
        out["merchant_type"] = out["merchant_type"].fillna("unknown").where(out["merchant_type"].isin(MERCHANT_RISK), "unknown")
        # geo imputation: unknown coordinates -> NaN handled later by the feature extractor
        # PII anonymisation (keyed hash) before any scoring / persistence
        if self.anonymize:
            for f in self.PII_FIELDS:
                out[f] = out[f].astype(str).map(anonymize_value)
        # numerical encoding of contextual category
        out["merchant_risk"] = out["merchant_type"].map(MERCHANT_RISK).astype(float)
        for c in ("is_fraud", "fraud_type"):
            if c not in out.columns:
                out[c] = 0 if c == "is_fraud" else "none"
        return out.reset_index(drop=True)
