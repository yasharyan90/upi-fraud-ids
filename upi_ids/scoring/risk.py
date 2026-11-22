"""Weighted risk aggregation and policy decision engine (Section III-F).

R(x) = w_r * rule(x) + w_s * p_xgb(x) + w_a * anomaly(x)

Approve   : R(x) <  tau1
Challenge : tau1 <= R(x) < tau2   (step-up MFA / OTP)
Alert     : tau2 <= R(x) < tau3   (analyst triage)
Block     : R(x) >= tau3
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Decision(str, Enum):
    APPROVE = "Approve"
    CHALLENGE = "Challenge"
    ALERT = "Alert"
    BLOCK = "Block"


@dataclass
class RiskWeights:
    rule: float = 0.30
    supervised: float = 0.50
    anomaly: float = 0.20

    def normalised(self) -> "RiskWeights":
        s = self.rule + self.supervised + self.anomaly
        return RiskWeights(self.rule / s, self.supervised / s, self.anomaly / s)


@dataclass
class PolicyThresholds:
    challenge: float = 0.30  # tau1
    alert: float = 0.60      # tau2
    block: float = 0.85      # tau3

    def __post_init__(self):
        if not (0 <= self.challenge < self.alert < self.block <= 1):
            raise ValueError("thresholds must satisfy 0 <= tau1 < tau2 < tau3 <= 1")


class RiskEngine:
    def __init__(self, weights: RiskWeights | None = None, thresholds: PolicyThresholds | None = None):
        self.w = (weights or RiskWeights()).normalised()
        self.t = thresholds or PolicyThresholds()

    def aggregate(self, rule: np.ndarray, supervised: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        r = self.w.rule * np.asarray(rule) + self.w.supervised * np.asarray(supervised) + self.w.anomaly * np.asarray(anomaly)
        return np.clip(r, 0.0, 1.0)

    def decide_one(self, r: float) -> Decision:
        if r >= self.t.block:
            return Decision.BLOCK
        if r >= self.t.alert:
            return Decision.ALERT
        if r >= self.t.challenge:
            return Decision.CHALLENGE
        return Decision.APPROVE

    def decide(self, risk: np.ndarray) -> list[Decision]:
        return [self.decide_one(float(r)) for r in np.asarray(risk)]

    def is_intercepted(self, decision: Decision) -> bool:
        """Alert and Block are treated as fraud interceptions for confusion-matrix purposes."""
        return decision in (Decision.ALERT, Decision.BLOCK)
