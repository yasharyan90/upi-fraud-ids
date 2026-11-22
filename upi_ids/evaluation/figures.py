"""Figures 2-4 of the paper plus SHAP importance, rendered with matplotlib (headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRIMARY = ["XGBoost (Proposed)", "Random Forest", "SVM (RBF)", "Isolation Forest", "Hybrid Risk Engine"]
COLORS = {"XGBoost (Proposed)": "#1f77b4", "Random Forest": "#2ca02c", "SVM (RBF)": "#ff7f0e",
          "Isolation Forest": "#9467bd", "Hybrid Risk Engine": "#d62728"}


def _models(summary):
    return [m for m in PRIMARY if m in summary["metrics"]]


def fig_f1(summary, path):
    ms = _models(summary)
    f1 = [summary["metrics"][m]["f1"] for m in ms]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(ms, f1, color=[COLORS[m] for m in ms])
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.05); ax.set_title("Fig. 2. F1-score of candidate detection models")
    plt.xticks(rotation=15, ha="right"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_fpr_fnr(summary, path):
    ms = _models(summary)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in ms:
        r = summary["metrics"][m]
        ax.scatter(r["fpr"] * 100, r["fnr"] * 100, s=120, color=COLORS[m], label=m, edgecolor="k")
        ax.annotate(m, (r["fpr"] * 100, r["fnr"] * 100), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("False Positive Rate (%)  — customer friction"); ax.set_ylabel("False Negative Rate (%)  — missed fraud")
    ax.set_title("Fig. 3. FPR vs FNR trade-off"); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_pr(summary, path):
    ms = _models(summary)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    p = np.linspace(0.01, 1, 200)
    for f in (0.3, 0.5, 0.7, 0.8, 0.9):
        r = f * p / (2 * p - f)
        mask = (r > 0) & (r <= 1)
        ax.plot(p[mask], r[mask], "--", color="grey", lw=0.8)
        ax.annotate(f"F1={f}", (0.98, (f * 0.98 / (2 * 0.98 - f))), fontsize=7, color="grey")
    for m in ms:
        r = summary["metrics"][m]
        ax.scatter(r["precision"], r["recall"], s=120, color=COLORS[m], label=m, edgecolor="k", zorder=3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xlabel("Precision"); ax.set_ylabel("Recall")
    ax.set_title("Fig. 4. Precision vs Recall with iso-F1 contours"); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_shap(summary, path):
    imp = summary["shap_global_importance"]
    names, vals = list(imp.keys())[::-1], list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(names, vals, color="#1f77b4"); ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global SHAP feature importance (XGBoost)"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_latency(summary, path):
    lat = summary["latency_ms"]
    keys = [k for k in lat if "parallel" not in k]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(keys, [lat[k]["mean_ms"] for k in keys], color="#17becf")
    ax.axvline(50, color="red", ls="--", lw=1); ax.text(50, -0.4, " 50 ms budget", color="red", fontsize=8)
    ax.set_xlabel("mean per-transaction latency (ms, log scale)"); ax.set_xscale("log")
    ax.set_title("Inference latency per engine"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def render_all(summary, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_f1(summary, out / "fig2_f1_comparison.png")
    fig_fpr_fnr(summary, out / "fig3_fpr_vs_fnr.png")
    fig_pr(summary, out / "fig4_precision_recall_isoF1.png")
    fig_shap(summary, out / "shap_global_importance.png")
    fig_latency(summary, out / "latency_per_engine.png")
