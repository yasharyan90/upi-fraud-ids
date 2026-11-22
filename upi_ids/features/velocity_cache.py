"""Low-latency per-user velocity / profile cache.

`InMemoryVelocityCache` is the default; `RedisVelocityCache` is used when a Redis
URL is supplied (mirrors the paper's "Redis-cached user profiles" design)."""
from __future__ import annotations

import bisect
from collections import defaultdict, deque


class InMemoryVelocityCache:
    def __init__(self):
        self._events: dict[str, deque] = defaultdict(deque)       # sorted timestamps per user
        self._amounts: dict[str, deque] = defaultdict(deque)      # (ts, amount) rolling window
        self._devices: dict[str, set] = defaultdict(set)
        self._receivers: dict[str, dict] = defaultdict(dict)
        self._home: dict[str, tuple] = {}

    # velocity -------------------------------------------------------------
    def count_window(self, user: str, ts: float, window_s: float) -> int:
        q = self._events[user]
        while q and q[0] < ts - window_s:
            q.popleft()
        return bisect.bisect_right(q, ts) if q else 0

    def last_ts(self, user: str) -> float | None:
        q = self._events[user]
        return q[-1] if q else None

    # rolling amount stats --------------------------------------------------
    def amount_stats(self, user: str, ts: float, window_s: float = 30 * 86400) -> tuple[float, float, int]:
        q = self._amounts[user]
        while q and q[0][0] < ts - window_s:
            q.popleft()
        n = len(q)
        if n == 0:
            return 0.0, 0.0, 0
        vals = [a for _, a in q]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return mean, var ** 0.5, n

    # device / peers / home -----------------------------------------------
    def known_device(self, user: str, device: str) -> bool:
        return device in self._devices[user]

    def receiver_count(self, user: str, receiver: str) -> int:
        return self._receivers[user].get(receiver, 0)

    def home(self, user: str) -> tuple | None:
        return self._home.get(user)

    # update ---------------------------------------------------------------
    def update(self, user: str, ts: float, amount: float, device: str, receiver: str, lat: float, lon: float):
        self._events[user].append(ts)
        self._amounts[user].append((ts, amount))
        self._devices[user].add(device)
        self._receivers[user][receiver] = self._receivers[user].get(receiver, 0) + 1
        if user not in self._home and lat == lat and lon == lon:  # first seen location is baseline
            self._home[user] = (lat, lon)


class RedisVelocityCache(InMemoryVelocityCache):
    """Redis-backed variant: keeps hot counters in Redis sorted sets, falls back to memory for stats."""

    def __init__(self, url: str):
        super().__init__()
        import redis  # optional dependency

        self.r = redis.Redis.from_url(url, decode_responses=True)

    def count_window(self, user: str, ts: float, window_s: float) -> int:
        key = f"vel:{user}"
        self.r.zremrangebyscore(key, 0, ts - window_s)
        return int(self.r.zcount(key, ts - window_s, ts))

    def update(self, user, ts, amount, device, receiver, lat, lon):
        super().update(user, ts, amount, device, receiver, lat, lon)
        self.r.zadd(f"vel:{user}", {f"{ts}": ts})
        self.r.expire(f"vel:{user}", 86400)


def make_velocity_cache(redis_url: str | None = None):
    if redis_url:
        try:
            return RedisVelocityCache(redis_url)
        except Exception:  # pragma: no cover - fall back gracefully
            pass
    return InMemoryVelocityCache()
