"""Command-line entry point: `python -m upi_ids.cli <command>` or `upi-ids <command>`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_experiment(a):
    from .evaluation.benchmark import run
    run(a.config, a.out, make_figures=not a.no_figures)


def cmd_train(a):
    from .config import load_config
    from .pipeline import HybridDetector
    det = HybridDetector.train(load_config(a.config))
    det.save(a.out)
    print(f"saved -> {a.out}")


def cmd_serve(a):
    import os
    import uvicorn
    if a.model:
        os.environ["UPI_IDS_MODEL"] = a.model
    uvicorn.run("upi_ids.api.app:get_app", factory=True, host=a.host, port=a.port, reload=False)


def cmd_simulate(a):
    from .config import load_config
    from .data.generator import UPITransactionGenerator
    from .monitoring.audit import AuditLog
    from .pipeline import HybridDetector
    from .streaming.bus import make_bus
    from .streaming.consumer import TOPIC_IN, run_consumer
    cfg = load_config(a.config)
    det = HybridDetector.load(a.model) if a.model and Path(a.model).exists() else HybridDetector.train(cfg)
    gen = UPITransactionGenerator(n_users=cfg["data"]["n_users"], seed=cfg["seed"] + 7)
    df = gen.generate(a.n, fraud_rate=a.fraud_rate, start_ts=1_700_000_000.0 + 40 * 86400)
    bus = make_bus(a.kafka)
    for rec in df.drop(columns=["is_fraud", "fraud_type"]).to_dict("records"):
        bus.publish(TOPIC_IN, rec)
    print(f"published {len(df)} events to '{TOPIC_IN}' ({df['is_fraud'].sum()} synthetic frauds hidden in the stream)")
    audit = AuditLog(cfg["monitoring"]["audit_log_path"])
    summary = run_consumer(bus, det, audit=audit, max_events=a.n)
    # measure interception of the hidden frauds
    decisions = {}
    for ev in bus.consume("upi.decisions", timeout=0.2):
        decisions[ev["txn_id"]] = ev["decision"]
    df["decision"] = df["txn_id"].map(decisions)
    intercepted = df[(df.is_fraud == 1) & df.decision.isin(["Alert", "Block"])].shape[0]
    challenged = df[(df.is_fraud == 1) & (df.decision == "Challenge")].shape[0]
    friction = df[(df.is_fraud == 0) & df.decision.isin(["Challenge", "Alert", "Block"])].shape[0]
    summary.update({"frauds_in_stream": int(df.is_fraud.sum()), "frauds_intercepted(Alert|Block)": intercepted,
                    "frauds_challenged": challenged, "legit_with_friction": friction,
                    "legit_friction_rate": friction / max(int((df.is_fraud == 0).sum()), 1),
                    "audit_chain_valid": audit.verify()})
    print(json.dumps(summary, indent=2))


def cmd_score(a):
    from .pipeline import HybridDetector
    det = HybridDetector.load(a.model) if a.model and Path(a.model).exists() else HybridDetector.train()
    payload = json.loads(Path(a.json).read_text()) if Path(a.json).exists() else json.loads(a.json)
    print(json.dumps(det.score_transaction(payload), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="upi-ids", description="Real-Time Intrusion Detection System for UPI Payments")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("experiment", help="reproduce the paper's experiments (tables, SHAP, latency, figures)")
    e.add_argument("--config"); e.add_argument("--out", default="results"); e.add_argument("--no-figures", action="store_true"); e.set_defaults(fn=cmd_experiment)
    t = sub.add_parser("train", help="train and persist the hybrid detector")
    t.add_argument("--config"); t.add_argument("--out", default="results/models/hybrid_detector.pkl"); t.set_defaults(fn=cmd_train)
    s = sub.add_parser("serve", help="run the REST API gateway + dashboard")
    s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8000); s.add_argument("--model"); s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("simulate", help="stream synthetic transactions through the bus + consumer")
    m.add_argument("--n", type=int, default=2000); m.add_argument("--fraud-rate", type=float, default=0.01); m.add_argument("--config")
    m.add_argument("--model"); m.add_argument("--kafka", help="bootstrap servers, e.g. localhost:9092 (default: in-memory bus)"); m.set_defaults(fn=cmd_simulate)
    c = sub.add_parser("score", help="score one transaction from a JSON file or string")
    c.add_argument("json"); c.add_argument("--model"); c.set_defaults(fn=cmd_score)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
