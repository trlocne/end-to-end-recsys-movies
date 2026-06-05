import time
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SimpleCache:
    """In-memory cache with TTL — fallback when Redis is unavailable."""

    def __init__(self):
        self._store: dict = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if expire_at > 0 and time.time() > expire_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 60):
        expire_at = time.time() + ttl if ttl > 0 else 0
        self._store[key] = (value, expire_at)

    def delete(self, key: str):
        self._store.pop(key, None)

    def invalidate_user(self, user_id: int):
        keys = [k for k in self._store if f":{user_id}:" in k]
        for k in keys:
            del self._store[k]


class RedisCache:
    """Shared cache backed by Redis — shared across all serving pods."""

    def __init__(self, connection_string: str):
        import redis as redis_lib
        # Parse Aiven-style connection string:
        # host:port,password=xxx,ssl=true,ssl_cert_reqs=none
        parts = connection_string.split(",")
        host_port = parts[0].split(":")
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 6379
        password = None
        ssl = False
        for part in parts[1:]:
            if part.startswith("password="):
                password = part[len("password="):]
            elif part.startswith("ssl=true"):
                ssl = True

        self._client = redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            ssl=ssl,
            ssl_cert_reqs=None,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._prefix = "recsys:rec:"

    def get(self, key: str) -> Optional[Any]:
        try:
            val = self._client.get(self._prefix + key)
            if val is None:
                return None
            return json.loads(val)
        except Exception as e:
            logger.warning("Redis get failed: %s", e)
            return None

    def set(self, key: str, value: Any, ttl: int = 60):
        try:
            self._client.setex(self._prefix + key, ttl, json.dumps(value))
        except Exception as e:
            logger.warning("Redis set failed: %s", e)

    def delete(self, key: str):
        try:
            self._client.delete(self._prefix + key)
        except Exception as e:
            logger.warning("Redis delete failed: %s", e)

    def invalidate_user(self, user_id: int):
        try:
            pattern = f"{self._prefix}*:{user_id}:*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            logger.warning("Redis invalidate_user failed: %s", e)


def build_cache() -> SimpleCache:
    """Return RedisCache if REDIS_URL is set, else fall back to SimpleCache."""
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            cache = RedisCache(redis_url)
            cache._client.ping()
            logger.info("Using Redis cache at %s", redis_url.split(",")[0])
            return cache
        except Exception as e:
            logger.warning("Redis unavailable (%s), falling back to in-memory cache", e)
    return SimpleCache()
