"""Per-tenant resource quota enforcement.

Tracks and limits per-tenant usage of documents, chunks, storage, and API
calls to prevent "noisy neighbour" problems in multi-tenant deployments.

Dual-mode:
    - **Disabled** (default, single-user): No enforcement — all operations allowed.
    - **Enabled** (production): Quotas checked before document uploads and searches.

Usage::

    from lasagna.core.quota import get_quota_manager

    qm = get_quota_manager()
    qm.check_upload(session_id="user_abc", file_size_bytes=5_000_000)
    qm.record_upload(session_id="user_abc", file_size_bytes=5_000_000, chunks=42)
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from loguru import logger


@dataclass
class TenantQuota:
    """Configurable limits for a single tenant."""

    max_documents: int = 1_000
    max_storage_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB
    max_chunks: int = 100_000
    max_searches_per_minute: int = 600
    max_uploads_per_hour: int = 200


@dataclass
class TenantUsage:
    """Tracked usage counters for a single tenant."""

    documents: int = 0
    storage_bytes: int = 0
    chunks: int = 0
    # Sliding window for search rate limiting — deque(maxlen) auto-evicts
    # oldest entries on append, preventing unbounded memory growth.
    _search_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    # Upload timestamps keyed by reservation_id for correct rollback.
    # Dict is bounded by quota enforcement + time-window pruning.
    _upload_timestamps: Dict[str, float] = field(default_factory=dict)

    def searches_last_minute(self) -> int:
        now = time.monotonic()
        cutoff = now - 60
        while self._search_timestamps and self._search_timestamps[0] <= cutoff:
            self._search_timestamps.popleft()
        return len(self._search_timestamps)

    def uploads_last_hour(self) -> int:
        now = time.monotonic()
        cutoff = now - 3600
        expired = [k for k, v in self._upload_timestamps.items() if v <= cutoff]
        for k in expired:
            del self._upload_timestamps[k]
        return len(self._upload_timestamps)


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds their resource quota."""

    def __init__(self, resource: str, limit: int, current: int, session_id: str):
        self.resource = resource
        self.limit = limit
        self.current = current
        self.session_id = session_id
        super().__init__(
            f"Quota exceeded for session {session_id!r}: "
            f"{resource} limit is {limit}, current usage is {current}"
        )


class QuotaManager:
    """Enforces per-tenant resource quotas.

    Thread-safe.  Usage counters are stored in-memory (reset on restart).
    For persistent quotas across restarts and workers, use ``RedisQuotaManager``.
    """

    def __init__(self, enabled: bool = False, default_quota: Optional[TenantQuota] = None) -> None:
        self._enabled = enabled
        self._default_quota = default_quota or TenantQuota()
        self._usage: Dict[str, TenantUsage] = {}
        self._tenant_quotas: Dict[str, TenantQuota] = {}
        self._lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _get_usage(self, session_id: str) -> TenantUsage:
        if session_id not in self._usage:
            with self._lock:
                if session_id not in self._usage:
                    self._usage[session_id] = TenantUsage()
        return self._usage[session_id]

    def _get_usage_locked(self, session_id: str) -> TenantUsage:
        """Get or create usage record. Must be called while ``self._lock`` is held.

        Search deque maxlen is sized to 2x the tenant's quota so that
        custom quotas above the default (600 searches/min) are tracked correctly.
        Upload timestamps use a dict (keyed by reservation_id) — no maxlen needed.
        """
        if session_id not in self._usage:
            quota = self._get_quota(session_id)
            self._usage[session_id] = TenantUsage(
                _search_timestamps=deque(maxlen=max(1200, quota.max_searches_per_minute * 2)),
            )
        return self._usage[session_id]

    def _get_quota(self, session_id: str) -> TenantQuota:
        return self._tenant_quotas.get(session_id, self._default_quota)

    def set_tenant_quota(self, session_id: str, quota: TenantQuota) -> None:
        """Override quota for a specific tenant.

        If usage tracking already exists for this session, resizes the search
        deque to match the new quota limits so that higher quotas are tracked
        correctly.
        """
        with self._lock:
            self._tenant_quotas[session_id] = quota
            if session_id in self._usage:
                usage = self._usage[session_id]
                new_maxlen = max(1200, quota.max_searches_per_minute * 2)
                if usage._search_timestamps.maxlen != new_maxlen:
                    usage._search_timestamps = deque(
                        usage._search_timestamps,
                        maxlen=new_maxlen,
                    )

    # -----------------------------------------------------------------------
    # Check methods (raise QuotaExceededError)
    # -----------------------------------------------------------------------

    def check_upload(self, session_id: str, file_size_bytes: int = 0) -> None:
        """Check if an upload is allowed.  Raises QuotaExceededError if not.

        Holds ``_lock`` for the entire check to prevent TOCTOU races
        where two concurrent uploads both pass the check before either
        records usage.
        """
        if not self._enabled:
            return

        with self._lock:
            usage = self._get_usage_locked(session_id)
            quota = self._get_quota(session_id)

            if usage.documents >= quota.max_documents:
                raise QuotaExceededError(
                    "max_documents", quota.max_documents, usage.documents, session_id
                )

            if (
                file_size_bytes > 0
                and (usage.storage_bytes + file_size_bytes) > quota.max_storage_bytes
            ):
                raise QuotaExceededError(
                    "max_storage_bytes",
                    quota.max_storage_bytes,
                    usage.storage_bytes + file_size_bytes,
                    session_id,
                )

            if usage.uploads_last_hour() >= quota.max_uploads_per_hour:
                raise QuotaExceededError(
                    "max_uploads_per_hour",
                    quota.max_uploads_per_hour,
                    usage.uploads_last_hour(),
                    session_id,
                )

    def check_search(self, session_id: str) -> None:
        """Check if a search is allowed.  Raises QuotaExceededError if not."""
        if not self._enabled:
            return

        with self._lock:
            usage = self._get_usage_locked(session_id)
            quota = self._get_quota(session_id)

            if usage.searches_last_minute() >= quota.max_searches_per_minute:
                raise QuotaExceededError(
                    "max_searches_per_minute",
                    quota.max_searches_per_minute,
                    usage.searches_last_minute(),
                    session_id,
                )

    def check_chunks(self, session_id: str, additional_chunks: int) -> None:
        """Check if adding chunks would exceed quota."""
        if not self._enabled:
            return

        with self._lock:
            usage = self._get_usage_locked(session_id)
            quota = self._get_quota(session_id)

            if (usage.chunks + additional_chunks) > quota.max_chunks:
                raise QuotaExceededError(
                    "max_chunks",
                    quota.max_chunks,
                    usage.chunks + additional_chunks,
                    session_id,
                )

    # -----------------------------------------------------------------------
    # Atomic check-and-reserve (eliminates TOCTOU between check and record)
    # -----------------------------------------------------------------------

    def check_and_reserve_upload(self, session_id: str, file_size_bytes: int = 0) -> str:
        """Atomically check quota and reserve capacity for an upload.

        Increments document count and storage bytes immediately so that
        concurrent uploads see the reserved capacity.  Call
        ``finalize_upload_reservation()`` after processing to record chunks,
        or ``release_upload_reservation()`` if processing fails.

        Returns:
            A reservation_id string.  Pass this to ``release_upload_reservation()``
            to roll back the correct reservation on failure.

        Raises:
            QuotaExceededError: If the upload would exceed any quota limit.
        """
        if not self._enabled:
            return ""

        reservation_id = uuid.uuid4().hex

        with self._lock:
            usage = self._get_usage_locked(session_id)
            quota = self._get_quota(session_id)

            if usage.documents >= quota.max_documents:
                raise QuotaExceededError(
                    "max_documents", quota.max_documents, usage.documents, session_id
                )

            if (
                file_size_bytes > 0
                and (usage.storage_bytes + file_size_bytes) > quota.max_storage_bytes
            ):
                raise QuotaExceededError(
                    "max_storage_bytes",
                    quota.max_storage_bytes,
                    usage.storage_bytes + file_size_bytes,
                    session_id,
                )

            if usage.uploads_last_hour() >= quota.max_uploads_per_hour:
                raise QuotaExceededError(
                    "max_uploads_per_hour",
                    quota.max_uploads_per_hour,
                    usage.uploads_last_hour(),
                    session_id,
                )

            # Reserve capacity (visible to concurrent checks)
            usage.documents += 1
            usage.storage_bytes += file_size_bytes
            usage._upload_timestamps[reservation_id] = time.monotonic()

        return reservation_id

    def finalize_upload_reservation(self, session_id: str, chunks: int) -> None:
        """Record chunk count after successful document processing.

        Called after ``check_and_reserve_upload()`` once the chunk count is known.
        """
        if not self._enabled:
            return
        with self._lock:
            usage = self._get_usage_locked(session_id)
            usage.chunks += chunks

    def release_upload_reservation(
        self,
        session_id: str,
        file_size_bytes: int = 0,
        reservation_id: str = "",
    ) -> None:
        """Roll back a reservation if document processing fails.

        Only decrements counters when the ``reservation_id`` is found in the
        upload timestamps.  Unknown or empty IDs are a no-op with a warning
        to prevent counter drift from duplicate/invalid releases.

        Args:
            reservation_id: The ID returned by ``check_and_reserve_upload()``.
                If empty or not found, logs a warning and returns without
                modifying any counters.
        """
        if not self._enabled:
            return
        with self._lock:
            usage = self._get_usage_locked(session_id)
            if not reservation_id or reservation_id not in usage._upload_timestamps:
                logger.warning(
                    f"release_upload_reservation: unknown reservation_id "
                    f"{reservation_id!r} for session {session_id!r}: no-op"
                )
                return
            del usage._upload_timestamps[reservation_id]
            usage.documents = max(0, usage.documents - 1)
            usage.storage_bytes = max(0, usage.storage_bytes - file_size_bytes)

    # -----------------------------------------------------------------------
    # Record methods (update counters after successful operations)
    # -----------------------------------------------------------------------

    def record_upload(self, session_id: str, file_size_bytes: int, chunks: int) -> None:
        """Record a successful upload.

        Prefer ``check_and_reserve_upload()`` + ``finalize_upload_reservation()``
        for TOCTOU-safe quota enforcement.  This method is kept for backward
        compatibility and non-critical paths (e.g. Celery task completion).
        """
        if not self._enabled:
            return
        with self._lock:
            usage = self._get_usage_locked(session_id)
            usage.documents += 1
            usage.storage_bytes += file_size_bytes
            usage.chunks += chunks
            usage._upload_timestamps[uuid.uuid4().hex] = time.monotonic()

    def record_delete(self, session_id: str, file_size_bytes: int, chunks: int) -> None:
        """Record a document deletion (frees quota)."""
        if not self._enabled:
            return
        with self._lock:
            usage = self._get_usage_locked(session_id)
            usage.documents = max(0, usage.documents - 1)
            usage.storage_bytes = max(0, usage.storage_bytes - file_size_bytes)
            usage.chunks = max(0, usage.chunks - chunks)

    def record_search(self, session_id: str) -> None:
        """Record a search operation."""
        if not self._enabled:
            return
        with self._lock:
            usage = self._get_usage_locked(session_id)
            usage._search_timestamps.append(time.monotonic())

    def sync_from_counters(
        self,
        session_id: str,
        documents: int,
        storage_bytes: int,
        chunks: int,
    ) -> None:
        """Overwrite in-memory document/storage/chunk counters from DB values.

        Used by GC Phase 6 to repair counter drift after process restarts.
        Counter population is unconditional — ``_enabled`` only controls
        enforcement (``check_*`` / ``record_*``), not the ability to seed
        counters so that enabling quota later starts from accurate values.
        Rate-limiting sliding windows (searches/min, uploads/hour) are NOT
        affected — they are transient and intentionally reset on restart.
        """
        with self._lock:
            usage = self._get_usage_locked(session_id)
            usage.documents = max(0, documents)
            usage.storage_bytes = max(0, storage_bytes)
            usage.chunks = max(0, chunks)

    def get_usage(self, session_id: str) -> Dict[str, int]:
        """Return current usage for a tenant (for admin/stats APIs)."""
        with self._lock:
            usage = self._get_usage_locked(session_id)
            quota = self._get_quota(session_id)
            return {
                "documents": usage.documents,
                "documents_limit": quota.max_documents,
                "storage_bytes": usage.storage_bytes,
                "storage_limit": quota.max_storage_bytes,
                "chunks": usage.chunks,
                "chunks_limit": quota.max_chunks,
                "searches_last_minute": usage.searches_last_minute(),
                "searches_limit": quota.max_searches_per_minute,
            }


# ---------------------------------------------------------------------------
# Redis-backed Quota Manager
# ---------------------------------------------------------------------------


class RedisQuotaManager(QuotaManager):
    """Persistent quota manager using Redis for cross-worker consistency.

    Extends ``QuotaManager`` with Redis-backed counters.  Sliding window
    counters use sorted sets; document/storage/chunk counters use atomic
    INCR operations.

    Falls back to in-memory counters if Redis is unavailable.
    """

    # Lua script for atomic check-and-increment (single counter)
    _LUA_CHECK_AND_INCR = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local current = tonumber(redis.call('GET', key) or '0')
    if current >= limit then
        return 0
    end
    redis.call('INCR', key)
    return 1
    """

    # Lua script for atomic release-if-exists.
    # Only decrements counters when the reservation member exists in the sorted set.
    # KEYS: [1]=docs_key, [2]=storage_key, [3]=uploads_hourly_key
    # ARGV: [1]=reservation_id, [2]=file_size
    # Returns: 1 if released, 0 if reservation_id not found (no-op).
    _LUA_RELEASE_RESERVATION = """
    local removed = redis.call('ZREM', KEYS[3], ARGV[1])
    if removed == 0 then
        return 0
    end
    local docs = tonumber(redis.call('GET', KEYS[1]) or '0')
    if docs > 0 then
        redis.call('DECRBY', KEYS[1], 1)
    end
    local file_size = tonumber(ARGV[2])
    if file_size > 0 then
        local storage = tonumber(redis.call('GET', KEYS[2]) or '0')
        if storage >= file_size then
            redis.call('DECRBY', KEYS[2], file_size)
        else
            redis.call('SET', KEYS[2], '0')
        end
    end
    return 1
    """

    # Lua script for atomic check-and-reserve upload.
    # Checks all three quota dimensions and reserves in one atomic operation.
    # KEYS: [1]=docs_key, [2]=storage_key, [3]=uploads_hourly_key
    # ARGV: [1]=max_docs, [2]=max_storage, [3]=file_size, [4]=max_uploads,
    #        [5]=now, [6]=cutoff (now-3600), [7]=reservation_id
    # Returns: {status, resource_name, current_value}
    #   status=1 means success, status=0 means quota exceeded.
    _LUA_CHECK_AND_RESERVE_UPLOAD = """
    local docs = tonumber(redis.call('GET', KEYS[1]) or '0')
    if docs >= tonumber(ARGV[1]) then
        return {0, 'max_documents', docs}
    end

    local storage = tonumber(redis.call('GET', KEYS[2]) or '0')
    local file_size = tonumber(ARGV[3])
    if file_size > 0 and (storage + file_size) > tonumber(ARGV[2]) then
        return {0, 'max_storage_bytes', storage + file_size}
    end

    redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[6])
    local count = redis.call('ZCARD', KEYS[3])
    if count >= tonumber(ARGV[4]) then
        return {0, 'max_uploads_per_hour', count}
    end

    redis.call('INCR', KEYS[1])
    if file_size > 0 then
        redis.call('INCRBY', KEYS[2], file_size)
    end
    redis.call('ZADD', KEYS[3], ARGV[5], ARGV[7])
    redis.call('EXPIRE', KEYS[3], 3600)
    return {1, 'ok', 0}
    """

    def __init__(
        self,
        enabled: bool = False,
        default_quota: Optional[TenantQuota] = None,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        super().__init__(enabled=enabled, default_quota=default_quota)
        self._redis = None
        self._lua_check_incr = None
        self._lua_check_reserve = None
        self._lua_release = None
        try:
            import redis

            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            # Register Lua scripts
            self._lua_check_incr = self._redis.register_script(self._LUA_CHECK_AND_INCR)
            self._lua_check_reserve = self._redis.register_script(
                self._LUA_CHECK_AND_RESERVE_UPLOAD
            )
            self._lua_release = self._redis.register_script(self._LUA_RELEASE_RESERVATION)
            from lasagna.core.provenance import redact_url

            logger.info(f"RedisQuotaManager connected: {redact_url(redis_url)}")
        except Exception as exc:
            logger.warning(f"RedisQuotaManager Redis connect failed, using in-memory: {exc}")
            self._redis = None

    def _rkey(self, session_id: str, resource: str) -> str:
        return f"rag:quota:{session_id}:{resource}"

    def check_upload(self, session_id: str, file_size_bytes: int = 0) -> None:
        if not self._enabled:
            return
        if self._redis is None:
            return super().check_upload(session_id, file_size_bytes)

        quota = self._get_quota(session_id)
        try:
            docs = int(self._redis.get(self._rkey(session_id, "documents")) or 0)
            if docs >= quota.max_documents:
                raise QuotaExceededError("max_documents", quota.max_documents, docs, session_id)

            storage = int(self._redis.get(self._rkey(session_id, "storage_bytes")) or 0)
            if file_size_bytes > 0 and (storage + file_size_bytes) > quota.max_storage_bytes:
                raise QuotaExceededError(
                    "max_storage_bytes",
                    quota.max_storage_bytes,
                    storage + file_size_bytes,
                    session_id,
                )

            # Sliding window for uploads/hour
            now = time.time()
            key = self._rkey(session_id, "uploads_hourly")
            self._redis.zremrangebyscore(key, "-inf", now - 3600)
            count = self._redis.zcard(key)
            if count >= quota.max_uploads_per_hour:
                raise QuotaExceededError(
                    "max_uploads_per_hour",
                    quota.max_uploads_per_hour,
                    count,
                    session_id,
                )
        except QuotaExceededError:
            raise
        except Exception as exc:
            logger.warning(f"Redis quota check failed, using in-memory: {exc}")
            return super().check_upload(session_id, file_size_bytes)

    def check_search(self, session_id: str) -> None:
        if not self._enabled:
            return
        if self._redis is None:
            return super().check_search(session_id)

        quota = self._get_quota(session_id)
        try:
            now = time.time()
            key = self._rkey(session_id, "searches_minute")
            self._redis.zremrangebyscore(key, "-inf", now - 60)
            count = self._redis.zcard(key)
            if count >= quota.max_searches_per_minute:
                raise QuotaExceededError(
                    "max_searches_per_minute",
                    quota.max_searches_per_minute,
                    count,
                    session_id,
                )
        except QuotaExceededError:
            raise
        except Exception as exc:
            logger.warning(f"Redis quota check failed, using in-memory: {exc}")
            return super().check_search(session_id)

    def check_chunks(self, session_id: str, additional_chunks: int) -> None:
        if not self._enabled:
            return
        if self._redis is None:
            return super().check_chunks(session_id, additional_chunks)

        quota = self._get_quota(session_id)
        try:
            chunks = int(self._redis.get(self._rkey(session_id, "chunks")) or 0)
            if (chunks + additional_chunks) > quota.max_chunks:
                raise QuotaExceededError(
                    "max_chunks",
                    quota.max_chunks,
                    chunks + additional_chunks,
                    session_id,
                )
        except QuotaExceededError:
            raise
        except Exception as exc:
            logger.warning(f"Redis quota check failed, using in-memory: {exc}")
            return super().check_chunks(session_id, additional_chunks)

    def check_and_reserve_upload(self, session_id: str, file_size_bytes: int = 0) -> str:
        """Atomically check quota and reserve capacity via Redis Lua script.

        A single Lua script checks all three quota dimensions (documents,
        storage, hourly uploads) and reserves atomically — no TOCTOU race
        between concurrent workers.  Falls back to in-memory on Redis failure.

        Returns:
            A reservation_id string for use with ``release_upload_reservation()``.
        """
        if not self._enabled:
            return ""
        if self._redis is None:
            return super().check_and_reserve_upload(session_id, file_size_bytes)

        quota = self._get_quota(session_id)
        reservation_id = uuid.uuid4().hex
        try:
            now = time.time()
            result = self._lua_check_reserve(
                keys=[
                    self._rkey(session_id, "documents"),
                    self._rkey(session_id, "storage_bytes"),
                    self._rkey(session_id, "uploads_hourly"),
                ],
                args=[
                    quota.max_documents,
                    quota.max_storage_bytes,
                    file_size_bytes,
                    quota.max_uploads_per_hour,
                    now,
                    now - 3600,
                    reservation_id,
                ],
            )
            status = int(result[0])
            if status == 0:
                resource = result[1] if isinstance(result[1], str) else result[1].decode()
                current = int(result[2])
                limit = getattr(quota, resource, 0)
                raise QuotaExceededError(resource, limit, current, session_id)
            return reservation_id
        except QuotaExceededError:
            raise
        except Exception as exc:
            logger.warning(f"Redis quota reserve failed, using in-memory: {exc}")
            return super().check_and_reserve_upload(session_id, file_size_bytes)

    def finalize_upload_reservation(self, session_id: str, chunks: int) -> None:
        """Record chunk count in Redis after successful processing."""
        if not self._enabled:
            return
        if self._redis is None:
            return super().finalize_upload_reservation(session_id, chunks)

        try:
            self._redis.incrby(self._rkey(session_id, "chunks"), chunks)
        except Exception as exc:
            logger.warning(f"Redis quota finalize failed, using in-memory: {exc}")
            super().finalize_upload_reservation(session_id, chunks)

    def release_upload_reservation(
        self,
        session_id: str,
        file_size_bytes: int = 0,
        reservation_id: str = "",
    ) -> None:
        """Roll back a reservation in Redis if processing fails.

        Uses an atomic Lua script that checks the reservation member exists
        in the sorted set before decrementing counters.  Unknown or empty
        reservation IDs are a no-op with a warning — counters are never
        decremented for unrecognised reservations.
        """
        if not self._enabled:
            return
        if self._redis is None:
            return super().release_upload_reservation(session_id, file_size_bytes, reservation_id)

        if not reservation_id:
            logger.warning(
                f"release_upload_reservation: empty reservation_id for "
                f"session {session_id!r}: no-op"
            )
            return

        try:
            released = self._lua_release(
                keys=[
                    self._rkey(session_id, "documents"),
                    self._rkey(session_id, "storage_bytes"),
                    self._rkey(session_id, "uploads_hourly"),
                ],
                args=[reservation_id, file_size_bytes],
            )
            if not int(released):
                logger.warning(
                    f"release_upload_reservation: reservation_id "
                    f"{reservation_id!r} not found in Redis for session "
                    f"{session_id!r}: no-op"
                )
        except Exception as exc:
            logger.warning(f"Redis quota release failed, using in-memory: {exc}")
            super().release_upload_reservation(session_id, file_size_bytes, reservation_id)

    def record_upload(self, session_id: str, file_size_bytes: int, chunks: int) -> None:
        if not self._enabled:
            return
        if self._redis is None:
            return super().record_upload(session_id, file_size_bytes, chunks)

        try:
            pipe = self._redis.pipeline()
            pipe.incr(self._rkey(session_id, "documents"))
            pipe.incrby(self._rkey(session_id, "storage_bytes"), file_size_bytes)
            pipe.incrby(self._rkey(session_id, "chunks"), chunks)
            now = time.time()
            member = uuid.uuid4().hex  # unique member avoids collision
            pipe.zadd(self._rkey(session_id, "uploads_hourly"), {member: now})
            pipe.expire(self._rkey(session_id, "uploads_hourly"), 3600)
            pipe.execute()
        except Exception as exc:
            logger.warning(f"Redis quota record failed, using in-memory: {exc}")
            super().record_upload(session_id, file_size_bytes, chunks)

    def record_delete(self, session_id: str, file_size_bytes: int, chunks: int) -> None:
        if not self._enabled:
            return
        if self._redis is None:
            return super().record_delete(session_id, file_size_bytes, chunks)

        try:
            pipe = self._redis.pipeline()
            pipe.decrby(self._rkey(session_id, "documents"), 1)
            pipe.decrby(self._rkey(session_id, "storage_bytes"), file_size_bytes)
            pipe.decrby(self._rkey(session_id, "chunks"), chunks)
            pipe.execute()
        except Exception as exc:
            logger.warning(f"Redis quota record failed, using in-memory: {exc}")
            super().record_delete(session_id, file_size_bytes, chunks)

    def sync_from_counters(
        self,
        session_id: str,
        documents: int,
        storage_bytes: int,
        chunks: int,
    ) -> None:
        """Overwrite Redis quota counters from authoritative DB values.

        Unconditional — see ``QuotaManager.sync_from_counters`` for rationale.
        """
        if self._redis is None:
            return super().sync_from_counters(session_id, documents, storage_bytes, chunks)

        try:
            pipe = self._redis.pipeline()
            pipe.set(self._rkey(session_id, "documents"), max(0, documents))
            pipe.set(self._rkey(session_id, "storage_bytes"), max(0, storage_bytes))
            pipe.set(self._rkey(session_id, "chunks"), max(0, chunks))
            pipe.execute()
            # Also sync the in-memory fallback so they stay consistent
            super().sync_from_counters(session_id, documents, storage_bytes, chunks)
        except Exception as exc:
            logger.warning(f"Redis quota sync failed, using in-memory: {exc}")
            super().sync_from_counters(session_id, documents, storage_bytes, chunks)

    def record_search(self, session_id: str) -> None:
        if not self._enabled:
            return
        if self._redis is None:
            return super().record_search(session_id)

        try:
            now = time.time()
            key = self._rkey(session_id, "searches_minute")
            pipe = self._redis.pipeline()
            pipe.zadd(key, {f"{now}": now})
            pipe.expire(key, 60)
            pipe.execute()
        except Exception as exc:
            logger.warning(f"Redis quota record failed, using in-memory: {exc}")
            super().record_search(session_id)

    def get_usage(self, session_id: str) -> Dict[str, int]:
        if self._redis is None:
            return super().get_usage(session_id)

        quota = self._get_quota(session_id)
        try:
            now = time.time()
            docs = int(self._redis.get(self._rkey(session_id, "documents")) or 0)
            storage = int(self._redis.get(self._rkey(session_id, "storage_bytes")) or 0)
            chunks = int(self._redis.get(self._rkey(session_id, "chunks")) or 0)
            self._redis.zremrangebyscore(
                self._rkey(session_id, "searches_minute"), "-inf", now - 60
            )
            searches = self._redis.zcard(self._rkey(session_id, "searches_minute"))
            return {
                "documents": docs,
                "documents_limit": quota.max_documents,
                "storage_bytes": storage,
                "storage_limit": quota.max_storage_bytes,
                "chunks": chunks,
                "chunks_limit": quota.max_chunks,
                "searches_last_minute": searches,
                "searches_limit": quota.max_searches_per_minute,
            }
        except Exception:
            return super().get_usage(session_id)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[QuotaManager] = None
_instance_lock = threading.Lock()


def get_quota_manager() -> QuotaManager:
    """Return the global QuotaManager singleton.

    Uses ``RedisQuotaManager`` when ``RAG_CACHE_BACKEND=redis``.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                from lasagna.core.config import rag_config

                enabled = getattr(rag_config, "quota_enabled", False)
                cache_backend = getattr(rag_config, "cache_backend", "memory")
                if cache_backend == "redis" and enabled:
                    redis_url = getattr(rag_config, "redis_url", "redis://localhost:6379")
                    _instance = RedisQuotaManager(
                        enabled=enabled,
                        redis_url=redis_url,
                    )
                else:
                    _instance = QuotaManager(enabled=enabled)
                logger.debug(
                    f"QuotaManager initialised: enabled={enabled}, type={type(_instance).__name__}"
                )
    return _instance


def configure_quota_manager(
    enabled: bool = False, default_quota: Optional[TenantQuota] = None
) -> None:
    """Reconfigure the global QuotaManager, honouring the cache backend.

    Same defect as `configure_rate_limiter`: this always built the in-memory
    manager, so `RedisQuotaManager` was unreachable in the served app and
    per-tenant quota counters were per-replica and reset on every pod restart.
    """
    global _instance

    use_redis = False
    redis_url = "redis://localhost:6379"
    try:
        from lasagna.core.config import rag_config

        use_redis = getattr(rag_config, "cache_backend", "memory") == "redis"
        redis_url = getattr(rag_config, "redis_url", redis_url)
    except Exception as exc:  # noqa: BLE001 - config must not break startup
        logger.debug(f"Could not read cache backend for the quota manager: {exc}")

    with _instance_lock:
        if use_redis and enabled:
            _instance = RedisQuotaManager(
                enabled=enabled, default_quota=default_quota, redis_url=redis_url
            )
            logger.info("QuotaManager using Redis (shared across replicas)")
        else:
            _instance = QuotaManager(enabled=enabled, default_quota=default_quota)
    logger.info(f"QuotaManager configured: enabled={enabled}")
