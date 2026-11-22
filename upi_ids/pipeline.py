"""HybridDetector: the trained, deployable unit (rules + XGBoost + Isolation Forest + risk policy + SHAP).

`HybridDetector.train(cfg)` builds everything from the synthetic generator;
`score_transaction(payload)` performs real-time scoring for one incoming UPI event
through the exact same preprocessing / feature path used in training."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data.generator import FEATURE_COLUMNS, UPITransactionGenerator
from .engines.anomaly import AnomalyEngine
from .engines.rules import RuleEngine
from .engines.supervised import SupervisedEngine, build_xgboost
from .evaluation.metrics import best_threshold_by_f1
from .explain.shap_explainer import ShapExplainer
from .features.extractor import FeatureExtractor
from .features.velocity_cache import make_velocity_cache
from .monitoring.drift import DriftMonitor
from .preprocessing.pipeline import Preprocessor
from .scoring.risk import PolicyThresholds, RiskEngine, RiskWeights


class HybridDetector:
    def __init__(self, cfg: dict, xgb: SupervisedEngine, iso: AnomalyEngine, rules: RuleEngine, risk: RiskEngine,
                 reference_features: pd.DataFrame, redis_url: str | None = None):
        self.cfg = cfg
        self.xgb, self.iso, self.rules, self.risk = xgb, iso, rules, risk
        self.pre = Preprocessor()
        self.fx = FeatureExtractor(make_velocity_cache(redis_url))
        self.explainer: ShapExplainer | None = None
        self.drift = DriftMonitor(reference_features, cfg["monitoring"]["drift_psi_threshold"])
        self.reference_features = reference_features

    # ------------------------------------------------------------ training
    @classmethod
    def train(cls, cfg: dict | None = None, verbose: bool = True) -> "HybridDetector":
        cfg = cfg or load_config()
        d, seed = cfg["data"], cfg["seed"]
        gen = UPITransactionGenerator(n_users=d["n_users"], seed=seed)
        raw = gen.generate(d["n_train"], fraud_rate=d["train_fraud_rate"])
        pre, fx = Preprocessor(), FeatureExtractor()
        df = pre.transform(raw)
        X, y = fx.transform(df), df["is_fraud"].values.astype(int)
        cut = int(len(X) * 0.8)
        imb = cfg["imbalance"]
        xgb = SupervisedEngine(build_xgboost(y[:cut], cfg["xgboost"], imb["class_weighting"], seed), "XGBoost",
                               cfg["xgboost"]["decision_threshold"], imb["use_smote"], imb["smote_sampling_strategy"],
                               imb["smote_k_neighbors"], seed).fit(X.iloc[:cut], y[:cut])
        xgb.threshold, f1 = best_threshold_by_f1(y[cut:], xgb.predict_proba(X.iloc[cut:]))
        iso = AnomalyEngine(cfg["isolation_forest"]["n_estimators"], cfg["isolation_forest"]["contamination"], seed).fit(X.iloc[:cut], y[:cut])
        rules = RuleEngine(**cfg["rules"])
        risk = RiskEngine(RiskWeights(**cfg["risk"]["weights"]), PolicyThresholds(**cfg["risk"]["thresholds"]))
        if verbose:
            print(f"trained XGBoost (val F1={f1:.3f}, threshold={xgb.threshold:.2f}) on {len(X)} synthetic transactions")
        det = cls(cfg, xgb, iso, rules, risk, X.iloc[cut:].reset_index(drop=True))
        det.explainer = ShapExplainer(xgb.model, FEATURE_COLUMNS)
        # warm the velocity cache with the training traffic so user baselines exist
        det.fx.transform(df)
        return det

    # ------------------------------------------------------------ scoring
    def score_transaction(self, payload: dict, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        raw = pd.DataFrame([{**payload}])
        if "txn_id" not in raw or not raw["txn_id"].iloc[0]:
            raw["txn_id"] = f"TXN{int(time.time()*1000)}"
        if "timestamp" not in raw or pd.isna(raw["timestamp"].iloc[0]):
            raw["timestamp"] = time.time()
        df = self.pre.transform(raw)
        X = self.fx.transform(df)
        t_feat = time.perf_counter()
        rule_res = self.rules.evaluate_one(X.iloc[0].to_dict())
        t_rule = time.perf_counter()
        p = float(self.xgb.predict_proba(X)[0])
        t_xgb = time.perf_counter()
        a = float(self.iso.score(X)[0])
        t_iso = time.perf_counter()
        R = float(self.risk.aggregate([rule_res.score], [p], [a])[0])
        decision = self.risk.decide_one(R)
        self.drift.observe(X.iloc[0].to_dict())
        shap = self.explainer.explain(X)[0] if (explain and self.explainer is not None) else None
        t_end = time.perf_counter()
        return {
            "txn_id": str(raw["txn_id"].iloc[0]),
            "decision": decision.value,
            "risk_score": round(R, 4),
            "engines": {"rule_score": round(rule_res.score, 4), "rules_triggered": rule_res.triggered,
                        "xgboost_probability": round(p, 4), "isolation_forest_anomaly": round(a, 4)},
            "features": {k: round(float(v), 4) for k, v in X.iloc[0].to_dict().items()},
            "explanation": shap,
            "latency_ms": {"features": round((t_feat - t0) * 1000, 3), "rules": round((t_rule - t_feat) * 1000, 3),
                           "xgboost": round((t_xgb - t_rule) * 1000, 3), "isolation_forest": round((t_iso - t_xgb) * 1000, 3),
                           "total": round((t_end - t0) * 1000, 3)},
        }

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        explainer, fx, drift = self.explainer, self.fx, self.drift
        self.explainer, self.fx, self.drift = None, None, None
        try:
            with open(path, "wb") as fh:
                pickle.dump(self, fh)
        finally:
            self.explainer, self.fx, self.drift = explainer, fx, drift

    @classmethod
    def load(cls, path: str | Path, redis_url: str | None = None) -> "HybridDetector":
        with open(path, "rb") as fh:
            det: HybridDetector = pickle.load(fh)
        det.fx = FeatureExtractor(make_velocity_cache(redis_url))
        det.explainer = ShapExplainer(det.xgb.model, FEATURE_COLUMNS)
        det.drift = DriftMonitor(det.reference_features, det.cfg["monitoring"]["drift_psi_threshold"])
        return det
