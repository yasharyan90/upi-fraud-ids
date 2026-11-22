"""Supervised detection engine: XGBoost (proposed) plus comparative baselines
(Random Forest, SVM-RBF, Decision Tree, Logistic Regression, MLP)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .imbalance import positive_class_weight, resample_training


class SupervisedEngine:
    """Wraps a probabilistic classifier with the paper's training protocol
    (SMOTE + class weighting on the training partition only)."""

    def __init__(self, model, name: str = "XGBoost", threshold: float = 0.5, use_smote: bool = True,
                 smote_strategy: float = 0.1, smote_k: int = 5, seed: int = 42):
        self.model = model
        self.name = name
        self.threshold = threshold
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy
        self.smote_k = smote_k
        self.seed = seed
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        Xr, yr = resample_training(X.values, np.asarray(y), self.use_smote, self.smote_strategy, self.smote_k, self.seed)
        self.model.fit(Xr, yr)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


def build_xgboost(y_train: np.ndarray, cfg: dict, class_weighting: bool = True, seed: int = 42) -> XGBClassifier:
    spw = positive_class_weight(y_train) if class_weighting else 1.0
    # after SMOTE the ratio shrinks, so temper the weight accordingly
    return XGBClassifier(
        n_estimators=cfg.get("n_estimators", 400),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        subsample=cfg.get("subsample", 0.9),
        colsample_bytree=cfg.get("colsample_bytree", 0.9),
        min_child_weight=cfg.get("min_child_weight", 2),
        reg_lambda=cfg.get("reg_lambda", 1.0),
        reg_alpha=cfg.get("reg_alpha", 0.0),
        scale_pos_weight=min(spw, 10.0),
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        eval_metric="aucpr",
    )


def build_baselines(seed: int = 42) -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                                class_weight="balanced_subsample", n_jobs=4, random_state=seed),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale",
                                                        class_weight="balanced", probability=True, random_state=seed)),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed)),
    }
