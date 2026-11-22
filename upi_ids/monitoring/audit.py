"""Immutable, hash-chained audit log (Section III-H): every scored transaction records
the anonymised features, engine scores, risk score, SHAP attributions and policy action."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev = self._last_hash()

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = None
        with open(self.path, "rb") as fh:
            for line in fh:
                if line.strip():
                    last = line
        return json.loads(last)["hash"] if last else "0" * 64

    def append(self, record: dict) -> dict:
        with self._lock:
            entry = {"ts": time.time(), "prev_hash": self._prev, **record}
            payload = json.dumps(entry, sort_keys=True, default=str).encode()
            entry["hash"] = hashlib.sha256(payload).hexdigest()
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
            self._prev = entry["hash"]
            return entry

    def verify(self) -> bool:
        """Re-compute the hash chain; returns False if any record was altered or removed."""
        prev = "0" * 64
        if not self.path.exists():
            return True
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                h = e.pop("hash")
                if e["prev_hash"] != prev or hashlib.sha256(json.dumps(e, sort_keys=True, default=str).encode()).hexdigest() != h:
                    return False
                prev = h
        return True

    def tail(self, n: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        return lines[-n:]
