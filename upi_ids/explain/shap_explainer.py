"""SHAP (TreeExplainer) per-transaction feature attribution (Section III-G).

f(x) = phi_0 + sum_j phi_j   where phi_0 is the baseline expectation and phi_j are
the Shapley attributions of the M engineered features."""
from __future__ import annotations

import numpy as np
import pandas as pd


class ShapExplainer:
    def __init__(self, model, feature_names: list[str]):
        import shap

        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
        vals = self.explainer.shap_values(np.asarray(X))
        if isinstance(vals, list):  # older shap versions return a list for binary classifiers
            vals = vals[-1]
        base = self.explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.ravel(base)[-1])
        out = []
        for i in range(len(X)):
            row = vals[i]
            order = np.argsort(-np.abs(row))[:top_k]
            out.append({
                "base_value": float(base),
                "attributions": [
                    {"feature": self.feature_names[j], "value": float(X.iloc[i, j]), "shap": float(row[j])}
                    for j in order
                ],
            })
        return out

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        vals = self.explainer.shap_values(np.asarray(X))
        if isinstance(vals, list):
            vals = vals[-1]
        return pd.Series(np.abs(vals).mean(axis=0), index=self.feature_names).sort_values(ascending=False)
