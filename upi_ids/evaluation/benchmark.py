"""End-to-end experiment: data generation -> preprocessing -> features -> training ->
comparative evaluation (Tables III-V), SHAP importance (VII-A), latency (VII-B) and figures (Fig. 2-4)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.generator import FEATURE_COLUMNS, UPITransactionGenerator
from ..engines.anomaly import AnomalyEngine
from ..engines.rules import RuleEngine
from ..engines.supervised import SupervisedEngine, build_baselines, build_xgboost
from ..explain.shap_explainer import ShapExplainer
from ..features.extractor import FeatureExtractor
from ..features.velocity_cache import InMemoryVelocityCache
from ..preprocessing.pipeline import Preprocessor
from ..scoring.risk import PolicyThresholds, RiskEngine, RiskWeights
from .metrics import best_threshold_by_f1, classification_metrics


def _latency_ms(fn, X: pd.DataFrame, n: int = 300, warmup: int = 20) -> dict:
    """Per-transaction inference latency (single-row calls, like a real gateway)."""
    rows = [X.iloc[[i]] for i in range(min(n + warmup, len(X)))]
    for r in rows[:warmup]:
        fn(r)
    t = []
    for r in rows[warmup:]:
        s = time.perf_counter()
        fn(r)
        t.append((time.perf_counter() - s) * 1000)
    return {"mean_ms": float(np.mean(t)), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}


def build_datasets(cfg: dict):
    d = cfg["data"]
    gen = UPITransactionGenerator(n_users=d["n_users"], seed=cfg["seed"])
    train_raw = gen.generate(d["n_train"], fraud_rate=d["train_fraud_rate"], start_ts=1_700_000_000.0)
    # evaluation batch happens strictly *after* training data (temporal split, no leakage)
    eval_raw = gen.generate(d["n_eval"], fraud_count=d["eval_fraud_count"], start_ts=1_700_000_000.0 + 31 * 86400)
    pre = Preprocessor()
    train = pre.transform(train_raw)
    evalb = pre.transform(eval_raw)
    fx = FeatureExtractor(InMemoryVelocityCache())
    X_train = fx.transform(train)       # cache warms up on training traffic ...
    X_eval = fx.transform(evalb)        # ... and keeps rolling into the evaluation month
    return train, evalb, X_train, X_eval


def run(config_path: str | None = None, out_dir: str | Path = "results", make_figures: bool = True) -> dict:
    cfg = load_config(config_path)
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    print("[1/6] generating + preprocessing synthetic UPI streams ...")
    train, evalb, X_train, X_eval = build_datasets(cfg)
    y_train, y_eval = train["is_fraud"].values.astype(int), evalb["is_fraud"].values.astype(int)
    print(f"      train={len(X_train)} (fraud={y_train.sum()})  eval={len(X_eval)} (fraud={y_eval.sum()})")

    # time-ordered hold-out from the *training* month for threshold calibration
    cut = int(len(X_train) * 0.8)
    X_fit, y_fit, X_val, y_val = X_train.iloc[:cut], y_train[:cut], X_train.iloc[cut:], y_train[cut:]

    print("[2/6] training XGBoost (proposed) + baselines ...")
    imb = cfg["imbalance"]
    models: dict[str, SupervisedEngine] = {}
    xgb = SupervisedEngine(build_xgboost(y_fit, cfg["xgboost"], imb["class_weighting"], seed), "XGBoost (Proposed)",
                           cfg["xgboost"]["decision_threshold"], imb["use_smote"], imb["smote_sampling_strategy"],
                           imb["smote_k_neighbors"], seed)
    models[xgb.name] = xgb
    for name, m in build_baselines(seed).items():
        models[name] = SupervisedEngine(m, name, 0.5, imb["use_smote"], imb["smote_sampling_strategy"], imb["smote_k_neighbors"], seed)
    train_time = {}
    for name, eng in models.items():
        s = time.perf_counter()
        eng.fit(X_fit, y_fit)
        eng.threshold, _ = best_threshold_by_f1(y_val, eng.predict_proba(X_val))
        train_time[name] = time.perf_counter() - s
        print(f"      {name:<22} trained in {train_time[name]:.1f}s  threshold={eng.threshold:.2f}")

    iso = AnomalyEngine(cfg["isolation_forest"]["n_estimators"], cfg["isolation_forest"]["contamination"], seed).fit(X_fit, y_fit)
    rules = RuleEngine(**cfg["rules"])
    risk = RiskEngine(RiskWeights(**cfg["risk"]["weights"]), PolicyThresholds(**cfg["risk"]["thresholds"]))

    print("[3/6] evaluating on the 10k batch ...")
    results = {}
    for name, eng in models.items():
        results[name] = classification_metrics(y_eval, eng.predict(X_eval))
    results["Isolation Forest"] = classification_metrics(y_eval, iso.predict(X_eval))
    results["Rule Engine"] = classification_metrics(y_eval, rules.predict(X_eval))

    # hybrid fusion
    p_xgb = xgb.predict_proba(X_eval)
    r_score = rules.score(X_eval)
    a_score = iso.score(X_eval)
    R = risk.aggregate(r_score, p_xgb, a_score)
    decisions = risk.decide(R)
    hybrid_pred = np.array([risk.is_intercepted(d) for d in decisions]).astype(int)
    results["Hybrid Risk Engine"] = classification_metrics(y_eval, hybrid_pred)
    decision_dist = pd.Series([d.value for d in decisions]).value_counts().to_dict()
    decision_by_class = pd.crosstab(pd.Series([d.value for d in decisions], name="decision"),
                                    pd.Series(y_eval, name="is_fraud")).to_dict()

    for name, m in results.items():
        print(f"      {name:<22} acc={m['accuracy']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} FPR={m['fpr']*100:.2f}% FNR={m['fnr']*100:.1f}%")

    print("[4/6] SHAP feature attribution ...")
    shap_exp = ShapExplainer(xgb.model, FEATURE_COLUMNS)
    importance = shap_exp.global_importance(X_eval.iloc[:2000]).round(5).to_dict()
    sample_frauds = X_eval[y_eval == 1].head(3)
    sample_explanations = shap_exp.explain(sample_frauds)

    print("[5/6] latency benchmark (single-transaction calls) ...")
    latency = {
        "Rule Engine": _latency_ms(rules.score, X_eval),
        "XGBoost (Proposed)": _latency_ms(xgb.predict_proba, X_eval),
        "Isolation Forest": _latency_ms(iso.score, X_eval),
        "Random Forest": _latency_ms(models["Random Forest"].predict_proba, X_eval),
        "SVM (RBF)": _latency_ms(models["SVM (RBF)"].predict_proba, X_eval),
    }
    # parallel branches => T_total bounded by slowest branch (+ aggregation overhead)
    def _full(row):
        risk.decide(risk.aggregate(rules.score(row), xgb.predict_proba(row), iso.score(row)))
    latency["Hybrid (sequential 3 engines + policy)"] = _latency_ms(_full, X_eval)
    latency["Hybrid (parallel bound = slowest branch)"] = {
        k: max(latency[e][k] for e in ("Rule Engine", "XGBoost (Proposed)", "Isolation Forest"))
        for k in ("mean_ms", "p50_ms", "p95_ms")
    }
    for k, v in latency.items():
        print(f"      {k:<42} mean={v['mean_ms']:.2f} ms  p95={v['p95_ms']:.2f} ms")

    summary = {
        "dataset": {"n_train": int(len(X_train)), "train_fraud": int(y_train.sum()), "n_eval": int(len(X_eval)),
                    "eval_fraud": int(y_eval.sum()), "eval_legit": int((y_eval == 0).sum()),
                    "features": FEATURE_COLUMNS, "fraud_types_eval": evalb["fraud_type"].value_counts().to_dict()},
        "thresholds": {n: e.threshold for n, e in models.items()},
        "metrics": results,
        "hybrid": {"weights": cfg["risk"]["weights"], "policy_thresholds": cfg["risk"]["thresholds"],
                   "decision_distribution": decision_dist, "decision_by_class": {str(k): v for k, v in decision_by_class.items()}},
        "shap_global_importance": importance,
        "shap_sample_fraud_explanations": sample_explanations,
        "latency_ms": latency,
        "train_time_s": train_time,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    _write_tables_md(summary, out / "results_tables.md")

    if make_figures:
        print("[6/6] rendering figures ...")
        from .figures import render_all
        render_all(summary, out / "figures")
    print(f"done -> {out/'metrics.json'}")
    return summary


def _write_tables_md(s: dict, path: Path) -> None:
    m = s["metrics"]
    lines = ["# Experimental results (auto-generated)", "",
             f"Evaluation batch: N = {s['dataset']['n_eval']:,} transactions "
             f"({s['dataset']['eval_legit']:,} legitimate / {s['dataset']['eval_fraud']} fraud).", "",
             "## Confusion matrices", "", "| Algorithm | TP | FN | FP | TN |", "|---|---|---|---|---|"]
    for k, v in m.items():
        lines.append(f"| {k} | {v['TP']} | {v['FN']} | {v['FP']} | {v['TN']} |")
    lines += ["", "## Comparative performance", "", "| Algorithm | Acc. | Prec. | Recall | F1 | FPR | FNR |", "|---|---|---|---|---|---|---|"]
    for k, v in m.items():
        lines.append(f"| {k} | {v['accuracy']*100:.2f}% | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['fpr']*100:.2f}% | {v['fnr']*100:.1f}% |")
    lines += ["", "## Per-transaction latency", "", "| Engine | mean (ms) | p50 (ms) | p95 (ms) |", "|---|---|---|---|"]
    for k, v in s["latency_ms"].items():
        lines.append(f"| {k} | {v['mean_ms']:.2f} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} |")
    lines += ["", "## Policy decision distribution (hybrid engine)", "", "| Decision | Count |", "|---|---|"]
    for k, v in s["hybrid"]["decision_distribution"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Global SHAP importance (mean |phi|)", "", "| Feature | mean abs SHAP |", "|---|---|"]
    for k, v in s["shap_global_importance"].items():
        lines.append(f"| {k} | {v:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
