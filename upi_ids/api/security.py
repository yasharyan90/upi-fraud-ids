"""Access-security layer (Section III-C): JWT authentication, salted password hashing,
emulated MFA (TOTP-like one-time codes) and a token-bucket rate limiter."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

import jwt


# ----------------------------------------------------------------- passwords
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000)
    return hmac.compare_digest(digest.hex(), digest_hex)


@dataclass
class User:
    username: str
    password_hash: str
    role: str = "service"  # service | analyst | admin
    mfa_enabled: bool = False


class UserStore:
    """In-memory user directory seeded with demo accounts (replace with PostgreSQL in production)."""

    def __init__(self):
        self.users: dict[str, User] = {}
        self.add("gateway", "gateway-secret", "service", mfa=False)
        self.add("analyst", "analyst-secret", "analyst", mfa=True)
        self.add("admin", "admin-secret", "admin", mfa=True)

    def add(self, username: str, password: str, role: str, mfa: bool):
        self.users[username] = User(username, hash_password(password), role, mfa)

    def authenticate(self, username: str, password: str) -> User | None:
        u = self.users.get(username)
        if u and verify_password(password, u.password_hash):
            return u
        return None


# ----------------------------------------------------------------------- MFA
class MFAEmulator:
    """Issues short-lived 6-digit codes. In the prototype the code is returned in the response
    (and logged) to stand in for an SMS / authenticator-app delivery channel."""

    def __init__(self, ttl_s: int = 300):
        self.ttl = ttl_s
        self._codes: dict[str, tuple[str, float]] = {}

    def issue(self, username: str) -> str:
        code = f"{secrets.randbelow(1_000_000):06d}"
        self._codes[username] = (code, time.time() + self.ttl)
        return code

    def verify(self, username: str, code: str) -> bool:
        item = self._codes.get(username)
        if not item:
            return False
        stored, exp = item
        ok = hmac.compare_digest(stored, code) and time.time() <= exp
        if ok:
            del self._codes[username]
        return ok


# ----------------------------------------------------------------------- JWT
@dataclass
class JWTManager:
    secret: str
    exp_minutes: int = 60
    algorithm: str = "HS256"

    def issue(self, user: User, mfa_verified: bool) -> str:
        now = int(time.time())
        payload = {"sub": user.username, "role": user.role, "mfa": mfa_verified, "iat": now,
                   "exp": now + self.exp_minutes * 60, "jti": secrets.token_hex(8)}
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def verify(self, token: str) -> dict:
        return jwt.decode(token, self.secret, algorithms=[self.algorithm])


# ---------------------------------------------------------------- rate limit
@dataclass
class TokenBucketLimiter:
    rate_per_minute: int = 120
    _buckets: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.rate_per_minute), now))
            tokens = min(float(self.rate_per_minute), tokens + (now - last) * self.rate_per_minute / 60.0)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True
