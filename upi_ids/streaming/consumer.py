"""Stream consumer: pulls transaction events from the bus, scores them with the HybridDetector
and publishes enforcement decisions to the `upi.decisions` topic (plus the alert topic)."""
from __future__ import annotations

import time

from ..monitoring.audit import AuditLog
from ..pipeline import HybridDetector
from .bus import InMemoryBus

TOPIC_IN, TOPIC_OUT, TOPIC_ALERTS = "upi.transactions", "upi.decisions", "upi.alerts"


def run_consumer(bus: InMemoryBus, detector: HybridDetector, audit: AuditLog | None = None,
                 max_events: int | None = None, timeout: float = 1.0, verbose: bool = True) -> dict:
    counts: dict[str, int] = {}
    lat = []
    n = 0
    for event in bus.consume(TOPIC_IN, timeout=timeout):
        t0 = time.perf_counter()
        res = detector.score_transaction(event, explain=(event.get("_explain", True)))
        lat.append((time.perf_counter() - t0) * 1000)
        counts[res["decision"]] = counts.get(res["decision"], 0) + 1
        bus.publish(TOPIC_OUT, res)
        if res["decision"] in ("Alert", "Block"):
            bus.publish(TOPIC_ALERTS, res)
        if audit is not None:
            audit.append({"actor": "stream-consumer", **{k: res[k] for k in ("txn_id", "decision", "risk_score", "engines", "features")}})
        n += 1
        if verbose and n % 500 == 0:
            print(f"  consumed {n} events  decisions={counts}  mean latency={sum(lat)/len(lat):.2f} ms")
        if max_events and n >= max_events:
            break
    return {"events": n, "decisions": counts, "mean_latency_ms": (sum(lat) / len(lat)) if lat else 0.0,
            "p95_latency_ms": (sorted(lat)[int(0.95 * len(lat)) - 1] if len(lat) > 20 else (max(lat) if lat else 0.0))}
