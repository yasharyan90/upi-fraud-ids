"""REST API gateway (Sections III-B/C and IV) built with FastAPI.

Endpoints
---------
POST /auth/login             -> JWT (or MFA challenge for MFA-enabled accounts)
POST /auth/mfa/verify        -> JWT after one-time code verification
POST /api/v1/score           -> real-time risk scoring of a single UPI transaction
POST /api/v1/score/batch     -> batch scoring
GET  /api/v1/alerts          -> recent Challenge/Alert/Block events (analyst)
GET  /api/v1/stats           -> live counters for the dashboard
GET  /api/v1/drift           -> PSI drift report
GET  /api/v1/audit/verify    -> hash-chain integrity check (admin)
GET  /                       -> React dashboard
GET  /health
"""
from __future__ import annotations

import os
import threading
from collections import Counter, deque
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ..config import load_config
from ..monitoring.audit import AuditLog
from ..pipeline import HybridDetector
from .security import JWTManager, MFAEmulator, TokenBucketLimiter, UserStore

DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"


class Transaction(BaseModel):
    txn_id: str | None = None
    timestamp: float | None = None
    sender_vpa: str
    receiver_vpa: str
    amount: float = Field(ge=0)
    merchant_type: str = "p2p"
    device_id: str
    ip_address: str = "0.0.0.0"
    lat: float | None = None
    lon: float | None = None
    auth_failures: int = 0
    account_age_days: int = 365


class LoginRequest(BaseModel):
    username: str
    password: str


class MFARequest(BaseModel):
    username: str
    code: str


class State:
    def __init__(self, cfg: dict, detector: HybridDetector):
        self.cfg = cfg
        self.detector = detector
        self.users = UserStore()
        self.mfa = MFAEmulator()
        self.jwt = JWTManager(os.environ.get("UPI_IDS_JWT_SECRET", cfg["api"]["jwt_secret"]), cfg["api"]["jwt_exp_minutes"])
        self.limiter = TokenBucketLimiter(cfg["api"]["rate_limit_per_minute"])
        self.audit = AuditLog(os.environ.get("UPI_IDS_AUDIT_PATH", cfg["monitoring"]["audit_log_path"]))
        self.recent: deque = deque(maxlen=500)
        self.counts = Counter()
        self.lock = threading.Lock()


def create_app(detector: HybridDetector | None = None, cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("UPI_IDS_CONFIG"))
    if detector is None:
        model_path = os.environ.get("UPI_IDS_MODEL", "results/models/hybrid_detector.pkl")
        detector = HybridDetector.load(model_path, os.environ.get("REDIS_URL")) if Path(model_path).exists() else HybridDetector.train(cfg)
    st = State(cfg, detector)
    app = FastAPI(title="UPI Real-Time Intrusion Detection API", version="1.0.0",
                  description="Hybrid rules + XGBoost + Isolation Forest risk scoring with SHAP explainability.")
    app.state.st = st
    bearer = HTTPBearer(auto_error=False)

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        key = request.client.host if request.client else "anon"
        if not st.limiter.allow(key):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        return await call_next(request)

    def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
        if creds is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        try:
            return st.jwt.verify(creds.credentials)
        except jwt.PyJWTError as e:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}")

    def require_role(*roles):
        def dep(user: dict = Depends(current_user)):
            if user["role"] not in roles:
                raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient role")
            if user["role"] in ("admin", "analyst") and cfg["api"]["mfa_required_for_admin"] and not user.get("mfa"):
                raise HTTPException(status.HTTP_403_FORBIDDEN, "MFA verification required")
            return user
        return dep

    # ------------------------------------------------------------------ auth
    @app.post("/auth/login")
    def login(req: LoginRequest):
        u = st.users.authenticate(req.username, req.password)
        if not u:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad credentials")
        if u.mfa_enabled:
            code = st.mfa.issue(u.username)
            # emulated delivery channel: code is logged + returned (prototype only)
            print(f"[MFA] one-time code for {u.username}: {code}")
            return {"mfa_required": True, "username": u.username, "demo_code": code}
        return {"mfa_required": False, "access_token": st.jwt.issue(u, mfa_verified=False), "token_type": "bearer"}

    @app.post("/auth/mfa/verify")
    def mfa_verify(req: MFARequest):
        u = st.users.users.get(req.username)
        if not u or not st.mfa.verify(req.username, req.code):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired MFA code")
        return {"access_token": st.jwt.issue(u, mfa_verified=True), "token_type": "bearer"}

    # --------------------------------------------------------------- scoring
    def _score(txn: Transaction, user: dict) -> dict:
        res = st.detector.score_transaction(txn.model_dump(exclude_none=True))
        with st.lock:
            st.counts[res["decision"]] += 1
            st.counts["total"] += 1
            if res["decision"] != "Approve":
                st.recent.appendleft(res)
        st.audit.append({"actor": user["sub"], "txn_id": res["txn_id"], "decision": res["decision"],
                         "risk_score": res["risk_score"], "engines": res["engines"], "features": res["features"],
                         "shap": res["explanation"]})
        return res

    @app.post("/api/v1/score")
    def score(txn: Transaction, user: dict = Depends(current_user)):
        return _score(txn, user)

    @app.post("/api/v1/score/batch")
    def score_batch(txns: list[Transaction], user: dict = Depends(current_user)):
        return [_score(t, user) for t in txns]

    # ------------------------------------------------------------- dashboard
    @app.get("/api/v1/alerts")
    def alerts(limit: int = 50, user: dict = Depends(require_role("analyst", "admin"))):
        return list(st.recent)[:limit]

    @app.get("/api/v1/stats")
    def stats(user: dict = Depends(current_user)):
        return {"counts": dict(st.counts), "policy": st.cfg["risk"], "model": "XGBoost + IsolationForest + Rules"}

    @app.get("/api/v1/drift")
    def drift(user: dict = Depends(require_role("analyst", "admin"))):
        return st.detector.drift.report()

    @app.get("/api/v1/audit/verify")
    def audit_verify(user: dict = Depends(require_role("admin"))):
        return {"chain_valid": st.audit.verify(), "records": len(st.audit.tail(10**9))}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD.read_text(encoding="utf-8")

    return app


def get_app() -> FastAPI:  # uvicorn entry point: `uvicorn upi_ids.api.app:get_app --factory`
    return create_app()
