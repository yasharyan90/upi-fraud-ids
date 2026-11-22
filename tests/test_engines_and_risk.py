import numpy as np
import pandas as pd
import pytest

from upi_ids.engines.rules import RuleEngine
from upi_ids.evaluation.metrics import classification_metrics, confusion_counts
from upi_ids.scoring.risk import Decision, PolicyThresholds, RiskEngine, RiskWeights


def feats(**kw):
    base = {"Txn_Amount": 500, "z_amt": 0.1, "Merchant_Type": 0.1, "Txn_Count_1Hr": 1, "Inter_Txn_Gap": 5000,
            "Distance_From_Home": 2, "Device_Fingerprint": 0, "Sender_Receiver_Rel": 1, "Account_History_Idx": 0.2, "Hour_Of_Day": 12}
    base.update(kw)
    return base


def test_rules_fire_on_policy_violations():
    re = RuleEngine()
    assert re.evaluate_one(feats()).score == 0.0
    r = re.evaluate_one(feats(Txn_Amount=60000))
    assert r.score == 1.0 and "EXCESSIVE_AMOUNT" in r.triggered
    r = re.evaluate_one(feats(Distance_From_Home=1200, Inter_Txn_Gap=600))
    assert "IMPOSSIBLE_TRAVEL" in r.triggered
    r = re.evaluate_one(feats(Txn_Count_1Hr=12, Account_History_Idx=4))
    assert set(r.triggered) >= {"TXN_BURST", "REPEATED_AUTH_FAILURE"} and r.score == 1.0


def test_policy_tiers_and_threshold_validation():
    eng = RiskEngine(RiskWeights(0.3, 0.5, 0.2), PolicyThresholds(0.3, 0.6, 0.85))
    assert eng.decide_one(0.1) == Decision.APPROVE
    assert eng.decide_one(0.3) == Decision.CHALLENGE
    assert eng.decide_one(0.6) == Decision.ALERT
    assert eng.decide_one(0.9) == Decision.BLOCK
    R = eng.aggregate(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    assert R[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        PolicyThresholds(0.6, 0.3, 0.9)


def test_metrics_match_paper_worked_example():
    # TP=88, FN=12, FP=35, TN=9865 (Section VI-A of the paper)
    y_true = np.array([1] * 100 + [0] * 9900)
    y_pred = np.array([1] * 88 + [0] * 12 + [1] * 35 + [0] * 9865)
    c = confusion_counts(y_true, y_pred)
    assert c == {"TP": 88, "FN": 12, "FP": 35, "TN": 9865}
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.9953, abs=1e-4)
    assert m["precision"] == pytest.approx(0.7154, abs=1e-3)
    assert m["recall"] == pytest.approx(0.880, abs=1e-3)
    assert m["f1"] == pytest.approx(0.789, abs=1e-3)
    assert m["fpr"] == pytest.approx(0.0035, abs=1e-4)
    assert m["fnr"] == pytest.approx(0.12, abs=1e-4)
