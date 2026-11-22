"""Configuration loading (YAML with sane defaults)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config file. Falls back to configs/default.yaml."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return copy.deepcopy(cfg)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """cfg lookup with dotted path, e.g. get(cfg, 'risk.weights.rule')."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
