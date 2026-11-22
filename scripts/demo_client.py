"""Minimal API client: logs in as the gateway service account and scores two sample transactions."""
import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
tok = httpx.post(f"{BASE}/auth/login", json={"username": "gateway", "password": "gateway-secret"}).json()["access_token"]
H = {"Authorization": f"Bearer {tok}"}
normal = {"sender_vpa": "user42@upi", "receiver_vpa": "grocer@upi", "amount": 450.0, "merchant_type": "grocery",
          "device_id": "dev-42-0", "ip_address": "10.1.2.3", "lat": 28.61, "lon": 77.21, "timestamp": time.time()}
suspicious = {"sender_vpa": "user42@upi", "receiver_vpa": "mule77@upi", "amount": 85000.0, "merchant_type": "crypto",
              "device_id": "rogue-device", "ip_address": "103.5.6.7", "lat": 12.97, "lon": 77.59, "auth_failures": 4,
              "timestamp": time.time() + 60}
for t in (normal, suspicious):
    r = httpx.post(f"{BASE}/api/v1/score", json=t, headers=H).json()
    print(json.dumps({k: r[k] for k in ("txn_id", "decision", "risk_score", "engines", "latency_ms")}, indent=2))
