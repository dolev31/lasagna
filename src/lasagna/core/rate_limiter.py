"""Adaptive rate limiter with priority-based throttling.

Implements a token-bucket per tenant that adjusts refill rate based on
system load, ensuring fair resource distribution under heavy traffic while
maximizing throughput when the system is idle.

Dual-mode:
    - **Disabled** (default, single-user): All requests pass through.
    - **Enabled** (production): Token bucket + load-aware throttling.

Priority tiers allow critical internal operations (health checks, admin)
to bypass throttling while background/low-priority tenants are throttled
first during overload.

Usage::

    from lasagna.core.rate_limiter import get_rate_limiter

    rl = get_rate_limiter()
    allowed, retry_after = rl.try_acquire("session_abc", "/api/v1/search")
    if not allowed:
        return 429, {"Retry-After": retry_after}
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Priority tiers
# ---------------------------------------------------------------------------


class RateTier(enum.Enum):
    """Tenant priority tier — determines throttling order."""

    CRITICAL = "critical"  # Never throttled (admin, health probes)
    HIGH = "high"  # Throttled last
    NORMAL = "normal"  # Default tier
    LOW = "low"  # Throttled first


_TIER_MULTIPLIERS: Dict[RateTier, float] = {
    RateTier.CRITICAL: 999.0,
    RateTier.HIGH: 1.5,
    RateTier.NORMAL: 1.0,
    RateTier.LOW: 0.5,
}


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


@dataclass
class _TokenBucket:
    """Per-tenant token bucket with fractional tokens."""

    capacity: float
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def refill(self, rate: float, now: float) -> None:
        """Add tokens based on elapsed time and current rate."""
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * rate)
        self.last_refill = now

    def try_consume(self, cost: float = 1.0) -> bool:
        """Try to consume *cost* tokens.  Returns True on success."""
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


# ---------------------------------------------------------------------------
# Load estimator
# ---------------------------------------------------------------------------


class _LoadEstimator:
    """Lightweight system load estimator.

    Uses a simple exponential moving average of request latency to
    estimate system load without depending on OS-level metrics (which
    vary across platforms and container runtimes).
    """

    def __init__(self, overload_threshold: float = 0.8, idle_threshold: float = 0.3) -> None:
        self._overload_threshold = overload_threshold
        self._idle_threshold = idle_threshold
        # EMA of normalised latency (0.0 = fast, 1.0 = at target ceiling)
        self._ema: float = 0.0
        self._alpha: float = 0.1  # EMA smoothing factor
        self._target_latency_ms: float = 500.0  # Latency ceiling for "fully loaded"
        self._lock = threading.Lock()

    @property
    def load_score(self) -> float:
        """Current load estimate in [0.0, 1.0]."""
        return min(1.0, max(0.0, self._ema))

    @property
    def is_overloaded(self) -> bool:
        return self.load_score >= self._overload_threshold

    @property
    def is_idle(self) -> bool:
        return self.load_score <= self._idle_threshold

    def record_latency(self, latency_ms: float) -> None:
        """Feed a request latency sample into the estimator."""
        normalised = min(1.0, latency_ms / self._target_latency_ms)
        with self._lock:
            self._ema = self._alpha * normalised + (1.0 - self._alpha) * self._ema


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


@dataclass
class RateLimiterConfig:
    """Rate limiter settings."""

    enabled: bool = False
    strategy: str = "adaptive"  # "adaptive" or "fixed"
    requests_per_minute: float = 60.0  # Base rate per tenant
    burst_capacity: float = 10.0  # Max burst tokens
    burst_multiplier: float = 2.0  # Capacity multiplier when idle
    overload_threshold: float = 0.8
    idle_threshold: float = 0.3


class RateLimiter:
    """Adaptive rate limiter with per-tenant token buckets.

    Thread-safe.

    In **fixed** mode, every tenant gets ``requests_per_minute / 60`` tokens
    per second, regardless of system load.

    In **adaptive** mode:
      - When the system is idle (load < idle_threshold), tenants get up to
        ``burst_multiplier`` times their base rate.
      - When the system is overloaded (load > overload_threshold), tenants
        are throttled to ``0.5×`` their base rate (tier-adjusted).
      - Priority tiers further scale the effective rate.
    """

    _MAX_BUCKETS: int = 50_000  # Eviction threshold for in-memory buckets

    def __init__(self, config: Optional[RateLimiterConfig] = None) -> None:
        self._config = config or RateLimiterConfig()
        self._buckets: Dict[str, _TokenBucket] = {}
        self._tenant_tiers: Dict[str, RateTier] = {}
        self._load_estimator = _LoadEstimator(
            overload_threshold=self._config.overload_threshold,
            idle_threshold=self._config.idle_threshold,
        )
        self._lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def load_estimator(self) -> _LoadEstimator:
        return self._load_estimator

    # -----------------------------------------------------------------------
    # Tenant tier management
    # -----------------------------------------------------------------------

    def set_tenant_tier(self, session_id: str, tier: RateTier) -> None:
        """Override priority tier for a specific tenant."""
        with self._lock:
            self._tenant_tiers[session_id] = tier

    def get_tenant_tier(self, session_id: str) -> RateTier:
        """Get the priority tier for a tenant (default: NORMAL)."""
        return self._tenant_tiers.get(session_id, RateTier.NORMAL)

    # -----------------------------------------------------------------------
    # Core rate limiting
    # -----------------------------------------------------------------------

    def try_acquire(
        self,
        session_id: str,
        path: str = "",
    ) -> Tuple[bool, Optional[float]]:
        """Try to acquire a token for the given tenant.

        Args:
            session_id: Tenant identifier.
            path: Request path (for logging/metrics).

        Returns:
            (allowed, retry_after_seconds).
            If allowed is True, retry_after is None.
            If allowed is False, retry_after is the estimated seconds
            until a token becomes available.
        """
        if not self._config.enabled:
            return True, None

        tier = self.get_tenant_tier(session_id)

        # CRITICAL tier is never throttled
        if tier == RateTier.CRITICAL:
            return True, None

        now = time.monotonic()

        # Compute effective rate and refill+consume atomically under a single
        # lock acquisition so the load-score snapshot used for the rate cannot
        # change between rate computation and token consumption.
        with self._lock:
            effective_rate = self._effective_rate(tier)
            bucket = self._get_or_create_bucket_locked(session_id)
            bucket.refill(effective_rate, now)
            if bucket.try_consume(1.0):
                return True, None

        # Estimate retry time: tokens needed / refill rate
        retry_after = max(0.1, 1.0 / effective_rate) if effective_rate > 0 else 1.0
        logger.debug(
            f"Rate limited session={session_id[:12]}... path={path} "
            f"tier={tier.value} load={self._load_estimator.load_score:.2f} "
            f"retry_after={retry_after:.1f}s"
        )
        return False, retry_after

    def record_request_latency(self, latency_ms: float) -> None:
        """Feed a request latency sample for load estimation."""
        self._load_estimator.record_latency(latency_ms)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _get_or_create_bucket_locked(self, session_id: str) -> _TokenBucket:
        """Get or create a token bucket for *session_id*.

        Must be called while ``self._lock`` is held.  Evicts the oldest
        bucket when the in-memory store exceeds ``_MAX_BUCKETS`` to
        prevent unbounded memory growth from ephemeral sessions.
        """
        if session_id not in self._buckets:
            # Evict oldest bucket when limit is reached
            if len(self._buckets) >= self._MAX_BUCKETS:
                oldest = next(iter(self._buckets))
                del self._buckets[oldest]
            capacity = self._config.burst_capacity
            self._buckets[session_id] = _TokenBucket(
                capacity=capacity,
                tokens=capacity,  # Start full
            )
        return self._buckets[session_id]

    def _effective_rate(self, tier: RateTier) -> float:
        """Compute effective token refill rate (tokens/second)."""
        base_rate = self._config.requests_per_minute / 60.0
        tier_multiplier = _TIER_MULTIPLIERS.get(tier, 1.0)

        if self._config.strategy == "fixed":
            return base_rate * tier_multiplier

        # Adaptive: adjust based on load
        load = self._load_estimator.load_score

        if self._load_estimator.is_idle:
            # Idle system: allow bursting
            load_factor = self._config.burst_multiplier
        elif self._load_estimator.is_overloaded:
            # Overloaded: throttle
            load_factor = 0.5
        else:
            # Linear interpolation between idle and overload thresholds
            idle = self._config.idle_threshold
            overload = self._config.overload_threshold
            t = (load - idle) / (overload - idle) if overload > idle else 0.5
            load_factor = self._config.burst_multiplier * (1 - t) + 0.5 * t

        return base_rate * tier_multiplier * load_factor


# ---------------------------------------------------------------------------
# Redis-backed Rate Limiter
# ---------------------------------------------------------------------------


class RedisRateLimiter(RateLimiter):
    """Rate limiter using Redis for cross-worker token bucket state.

    Stores token bucket state (tokens + last_refill timestamp) in Redis
    using a Lua script for atomic refill + consume.  The ``_LoadEstimator``
    stays in-memory per worker since each worker has its own latency view.

    Falls back to in-memory buckets if Redis is unavailable.
    """

    # Lua script for atomic token bucket refill + consume
    _LUA_TOKEN_BUCKET = """
    local tokens_key = KEYS[1]
    local ts_key = KEYS[2]
    local capacity = tonumber(ARGV[1])
    local rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local cost = tonumber(ARGV[4])
    local ttl = tonumber(ARGV[5])

    -- Read current state
    local tokens = tonumber(redis.call('GET', tokens_key) or capacity)
    local last_refill = tonumber(redis.call('GET', ts_key) or now)

    -- Refill based on elapsed time
    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * rate)

    -- Try consume
    if tokens >= cost then
        tokens = tokens - cost
        redis.call('SET', tokens_key, tostring(tokens))
        redis.call('SET', ts_key, tostring(now))
        redis.call('EXPIRE', tokens_key, ttl)
        redis.call('EXPIRE', ts_key, ttl)
        return 1
    else
        -- Update refill timestamp even on failure
        redis.call('SET', tokens_key, tostring(tokens))
        redis.call('SET', ts_key, tostring(now))
        redis.call('EXPIRE', tokens_key, ttl)
        redis.call('EXPIRE', ts_key, ttl)
        return 0
    end
    """

    def __init__(
        self,
        config: Optional[RateLimiterConfig] = None,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        super().__init__(config=config)
        self._redis = None
        self._lua_bucket = None
        try:
            import redis

            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._lua_bucket = self._redis.register_script(self._LUA_TOKEN_BUCKET)
            from lasagna.core.provenance import redact_url

            logger.info(f"RedisRateLimiter connected: {redact_url(redis_url)}")
        except Exception as exc:
            logger.warning(f"RedisRateLimiter Redis connect failed, using in-memory: {exc}")
            self._redis = None

    def try_acquire(
        self,
        session_id: str,
        path: str = "",
    ) -> Tuple[bool, Optional[float]]:
        if not self._config.enabled:
            return True, None

        tier = self.get_tenant_tier(session_id)
        if tier == RateTier.CRITICAL:
            return True, None

        if self._redis is None or self._lua_bucket is None:
            return super().try_acquire(session_id, path)

        try:
            # Use wall clock (not monotonic) because this timestamp is shared
            # across workers via Redis.  time.monotonic() is process-local.
            now = time.time()
            effective_rate = self._effective_rate(tier)
            capacity = self._config.burst_capacity

            tokens_key = f"rag:rl:{session_id}:tokens"
            ts_key = f"rag:rl:{session_id}:ts"
            ttl = 300  # 5 minute TTL for inactive buckets

            result = self._lua_bucket(
                keys=[tokens_key, ts_key],
                args=[capacity, effective_rate, now, 1.0, ttl],
            )

            if result == 1:
                return True, None

            retry_after = max(0.1, 1.0 / effective_rate) if effective_rate > 0 else 1.0
            logger.debug(
                f"Rate limited session={session_id[:12]}... path={path} "
                f"tier={tier.value} load={self._load_estimator.load_score:.2f} "
                f"retry_after={retry_after:.1f}s (Redis)"
            )
            return False, retry_after

        except Exception as exc:
            logger.warning(f"Redis rate limit failed, using in-memory: {exc}")
            return super().try_acquire(session_id, path)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[RateLimiter] = None
_instance_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Return the global RateLimiter singleton.

    Uses ``RedisRateLimiter`` when ``RAG_CACHE_BACKEND=redis``.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                from lasagna.core.config import rag_config

                cfg = RateLimiterConfig(
                    enabled=getattr(rag_config, "rate_limiting_enabled", False),
                    strategy=getattr(rag_config, "rate_limiting_strategy", "adaptive"),
                    requests_per_minute=getattr(rag_config, "rate_limiting_rpm", 60.0),
                    burst_capacity=getattr(rag_config, "rate_limiting_burst", 10.0),
                    burst_multiplier=getattr(rag_config, "rate_limiting_burst_multiplier", 2.0),
                    overload_threshold=getattr(rag_config, "rate_limiting_overload_threshold", 0.8),
                    idle_threshold=getattr(rag_config, "rate_limiting_idle_threshold", 0.3),
                )
                cache_backend = getattr(rag_config, "cache_backend", "memory")
                if cache_backend == "redis" and cfg.enabled:
                    redis_url = getattr(rag_config, "redis_url", "redis://localhost:6379")
                    _instance = RedisRateLimiter(config=cfg, redis_url=redis_url)
                else:
                    _instance = RateLimiter(config=cfg)
                logger.debug(
                    f"RateLimiter initialised: enabled={cfg.enabled}, "
                    f"strategy={cfg.strategy}, type={type(_instance).__name__}"
                )
    return _instance


def configure_rate_limiter(config: Optional[RateLimiterConfig] = None) -> None:
    """Reconfigure the global RateLimiter, honouring the cache backend.

    This unconditionally constructed the in-memory `RateLimiter`, and
    `RAGBackend._do_init` calls it on every startup -- so it overwrote whatever
    `get_rate_limiter()` would have selected and `RedisRateLimiter` was
    unreachable in the served app. The shipped k8s manifests set
    `RAG_CACHE_BACKEND: redis` and run two replicas behind an HPA, so each
    replica enforced its own private bucket: the effective limit was N times the
    configured one, and ~250 lines of Lua token-bucket never executed.
    """
    global _instance

    use_redis = False
    redis_url = "redis://localhost:6379"
    try:
        from lasagna.core.config import rag_config

        use_redis = getattr(rag_config, "cache_backend", "memory") == "redis"
        redis_url = getattr(rag_config, "redis_url", redis_url)
    except Exception as exc:  # noqa: BLE001 - config must not break startup
        logger.debug(f"Could not read cache backend for the rate limiter: {exc}")

    with _instance_lock:
        if use_redis and config is not None and config.enabled:
            _instance = RedisRateLimiter(config=config, redis_url=redis_url)
            logger.info("RateLimiter using Redis (shared across replicas)")
        else:
            _instance = RateLimiter(config=config)
    if config:
        logger.info(f"RateLimiter configured: enabled={config.enabled}, strategy={config.strategy}")
