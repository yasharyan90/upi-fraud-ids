import json

import pytest
from fastapi.testclient import TestClient

from upi_ids.api.app import create_app
from upi_ids.config import load_config
from upi_ids.monitoring.audit import AuditLog
from upi_ids.pipeline import HybridDetector


@pytest.fixture(scope="module")
def detector():
    cfg = load_config()
    cfg["data"].update({"n_train": 6000, "n_users": 400})
    cfg["xgboost"].update({"n_estimators": 60})
    cfg["isolation_forest"]["n_estimators"] = 50
    return HybridDetector.train(cfg, verbose=False)


@pytest.fixture(scope="module")
def client(detector, tmp_path_factory):
    cfg = load_config()
    cfg["monitoring"]["audit_log_path"] = str(tmp_path_factory.mktemp("audit") / "audit.jsonl")
    cfg["api"]["rate_limit_per_minute"] = 10000
    return TestClient(create_app(detector, cfg))


def test_score_transaction_structure(detector):
    res = detector.score_transaction({"sender_vpa": "u1@upi", "receiver_vpa": "m@upi", "amount": 250.0, "merchant_type": "grocery",
                                      "device_id": "d1", "ip_address": "1.1.1.1", "lat": 28.6, "lon": 77.2, "timestamp": 1_800_000_000})
    assert res["decision"] in {"Approve", "Challenge", "Alert", "Block"}
    assert 0 <= res["risk_score"] <= 1 and res["explanation"]["attributions"]
    assert res["latency_ms"]["total"] < 500


def test_obvious_fraud_is_blocked_or_alerted(detector):
    for i in range(5):  # warm a baseline for this user
        detector.score_transaction({"sender_vpa": "victim@upi", "receiver_vpa": "shop@upi", "amount": 300.0, "merchant_type": "grocery",
                                    "device_id": "home-phone", "ip_address": "10.0.0.1", "lat": 28.6, "lon": 77.2, "timestamp": 1_800_000_000 + i * 7200}, explain=False)
    res = detector.score_transaction({"sender_vpa": "victim@upi", "receiver_vpa": "mule@upi", "amount": 95000.0, "merchant_type": "crypto",
                                      "device_id": "rogue", "ip_address": "103.1.1.1", "lat": 12.97, "lon": 77.59, "auth_failures": 5,
                                      "timestamp": 1_800_000_000 + 5 * 7200 + 60})
    assert res["decision"] in {"Alert", "Block"}
    assert "EXCESSIVE_AMOUNT" in res["engines"]["rules_triggered"]


def test_api_auth_mfa_scoring_and_audit(client):
    assert client.post("/api/v1/score", json={}).status_code == 401
    r = client.post("/auth/login", json={"username": "gateway", "password": "bad"})
    assert r.status_code == 401
    tok = client.post("/auth/login", json={"username": "gateway", "password": "gateway-secret"}).json()["access_token"]
    txn = {"sender_vpa": "a@upi", "receiver_vpa": "b@upi", "amount": 120.0, "device_id": "d", "lat": 19.0, "lon": 72.8}
    r = client.post("/api/v1/score", json=txn, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["decision"] in {"Approve", "Challenge", "Alert", "Block"}
    # service account cannot read analyst endpoints
    assert client.get("/api/v1/alerts", headers={"Authorization": f"Bearer {tok}"}).status_code == 403
    # analyst must complete MFA
    step = client.post("/auth/login", json={"username": "analyst", "password": "analyst-secret"}).json()
    assert step["mfa_required"]
    assert client.post("/auth/mfa/verify", json={"username": "analyst", "code": "000000"}).status_code in (401, 200)
    atok = client.post("/auth/mfa/verify", json={"username": "analyst", "code": step["demo_code"]}).json().get("access_token")
    if atok is None:  # the 000000 guess above could theoretically have consumed the code
        step = client.post("/auth/login", json={"username": "analyst", "password": "analyst-secret"}).json()
        atok = client.post("/auth/mfa/verify", json={"username": "analyst", "code": step["demo_code"]}).json()["access_token"]
    assert client.get("/api/v1/alerts", headers={"Authorization": f"Bearer {atok}"}).status_code == 200
    assert client.get("/api/v1/drift", headers={"Authorization": f"Bearer {atok}"}).json()["status"] in {"warming_up", "stable", "drift"}
    assert client.get("/").status_code == 200 and "Dashboard" in client.get("/").text


def test_audit_log_hash_chain(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append({"txn_id": "1", "decision": "Approve"})
    log.append({"txn_id": "2", "decision": "Block"})
    assert log.verify()
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    e = json.loads(lines[0]); e["decision"] = "Approve-tampered"
    (tmp_path / "a.jsonl").write_text(json.dumps(e) + "\n" + lines[1] + "\n")
    assert not AuditLog(tmp_path / "a.jsonl").verify()
