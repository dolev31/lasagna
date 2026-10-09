"""Lifecycle policy configuration for document garbage collection.

``LifecyclePolicy`` is a plain dataclass that controls all GC thresholds and
scheduling intervals.  It is read by ``GarbageCollector`` and
``LifecycleScheduler`` at runtime and can be overridden per-deployment via
environment variables or explicit instantiation.

Example::

    from lasagna.lifecycle import LifecyclePolicy
    policy = LifecyclePolicy(soft_delete_grace_hours=48, document_ttl_days=30)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LifecyclePolicy:
    """Configures all document lifecycle and GC behaviour.

    Attributes:
        soft_delete_grace_hours:
            How long a soft-deleted document survives before GC hard-deletes it.
            During this window the document can be restored.  Default: 24 h.
        document_ttl_days:
            Auto-expire documents older than this many days (sets ``expires_at``
            at upload time).  ``None`` or ``0`` (default ``None``) both disable
            TTL-based expiry.  Via env: ``RAG_GC_TTL_DAYS=0`` also disables.
        error_retention_days:
            GC Phase 3 soft-deletes ``status=error`` documents older than this.
            Set to 0 to disable.  Default: 7 days.
        session_idle_days:
            GC Phase 5 removes sessions not accessed within this window.
            Default: 30 days.
        gc_cycle_hours:
            How often the full 6-phase GC cycle runs.  Default: 24 h (daily).
        orphan_scan_hours:
            How often ``OrphanDetector`` scans for cross-store inconsistencies.
            Default: 168 h (weekly).
        safety_threshold_percent:
            GC aborts Phase 1 with a warning if the fraction of documents
            marked for hard deletion exceeds this threshold.
            Default: 0.25 (25 %).
        batch_size:
            Documents processed per batch in each GC phase.  Smaller values
            reduce memory pressure but increase round-trips.  Default: 100.
        reconcile_quota_on_startup:
            If True, rebuild quota counters from the DB on ``RAGBackend``
            initialisation.  Corrects counter drift after process restarts.
            Default: True.
    """

    # ---- Soft-delete -------------------------------------------------------
    soft_delete_grace_hours: int = 24

    # ---- TTL / auto-expiry -------------------------------------------------
    document_ttl_days: Optional[int] = None
    error_retention_days: int = 7

    # ---- Session cleanup ---------------------------------------------------
    session_idle_days: int = 30

    # ---- Scheduling --------------------------------------------------------
    gc_cycle_hours: int = 24
    orphan_scan_hours: int = 168

    # ---- Safety / performance ----------------------------------------------
    safety_threshold_percent: float = 0.25
    batch_size: int = 100
    reconcile_quota_on_startup: bool = True

    @classmethod
    def load(cls) -> "LifecyclePolicy":
        """Load policy from TOML settings, then apply env-var overrides.

        Priority (highest wins):
        1. Environment variables (``RAG_GC_*`` — see below)
        2. ``[rag.lifecycle]`` section of ``settings.lasagna.toml``
        3. Dataclass defaults

        Supported env variables (all optional):

        * ``RAG_GC_GRACE_HOURS``
        * ``RAG_GC_TTL_DAYS``
        * ``RAG_GC_ERROR_RETENTION_DAYS``
        * ``RAG_GC_SESSION_IDLE_DAYS``
        * ``RAG_GC_CYCLE_HOURS``
        * ``RAG_GC_ORPHAN_SCAN_HOURS``
        * ``RAG_GC_SAFETY_THRESHOLD``
        * ``RAG_GC_BATCH_SIZE``
        * ``RAG_GC_RECONCILE_QUOTA``
        """
        # --- Step 1: load TOML base values via _get_knowledge_setting ----------
        # RAGConfig has no flat "lifecycle" attribute; the canonical way to
        # traverse the Dynaconf settings tree is _get_knowledge_setting(*keys).
        try:
            from lasagna.core.config import rag_config

            def _toml(key: str, default):
                """Read rag_config.rag.lifecycle.<key> from TOML/env priority chain."""
                return rag_config._get_knowledge_setting("rag", "lifecycle", key, default=default)

            ttl_raw = _toml("document_ttl_days", None)
            base = cls(
                soft_delete_grace_hours=int(_toml("soft_delete_grace_hours", 24)),
                document_ttl_days=(int(ttl_raw) if ttl_raw not in (None, 0, "", False) else None),
                error_retention_days=int(_toml("error_retention_days", 7)),
                session_idle_days=int(_toml("session_idle_days", 30)),
                gc_cycle_hours=int(_toml("gc_cycle_hours", 24)),
                orphan_scan_hours=int(_toml("orphan_scan_hours", 168)),
                safety_threshold_percent=float(_toml("safety_threshold_percent", 0.25)),
                batch_size=int(_toml("batch_size", 100)),
                reconcile_quota_on_startup=bool(_toml("reconcile_quota_on_startup", True)),
            )
        except Exception:
            base = cls()  # fall back to pure dataclass defaults

        # --- Step 2: apply env-var overrides (env wins over TOML) ------------
        return cls._apply_env_overrides(base)

    @classmethod
    def from_env(cls) -> "LifecyclePolicy":
        """Backward-compatible alias for ``load()``."""
        return cls.load()

    @classmethod
    def _apply_env_overrides(cls, base: "LifecyclePolicy") -> "LifecyclePolicy":
        """Apply RAG_GC_* environment variable overrides to *base* and return a new instance."""
        import dataclasses

        def _int(key: str, default: Optional[int]) -> Optional[int]:
            val = os.environ.get(key)
            return int(val) if val is not None else default

        def _float(key: str, default: float) -> float:
            val = os.environ.get(key)
            return float(val) if val is not None else default

        def _bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("1", "true", "yes"):
                return True
            if val in ("0", "false", "no"):
                return False
            return default

        # Build overrides for non-None fields only (None means "not set in env").
        overrides = {
            k: v
            for k, v in {
                "soft_delete_grace_hours": _int("RAG_GC_GRACE_HOURS", None),
                "error_retention_days": _int("RAG_GC_ERROR_RETENTION_DAYS", None),
                "session_idle_days": _int("RAG_GC_SESSION_IDLE_DAYS", None),
                "gc_cycle_hours": _int("RAG_GC_CYCLE_HOURS", None),
                "orphan_scan_hours": _int("RAG_GC_ORPHAN_SCAN_HOURS", None),
                "safety_threshold_percent": _float("RAG_GC_SAFETY_THRESHOLD", None),  # type: ignore[arg-type]
                "batch_size": _int("RAG_GC_BATCH_SIZE", None),
                "reconcile_quota_on_startup": (
                    _bool("RAG_GC_RECONCILE_QUOTA", base.reconcile_quota_on_startup)
                    if "RAG_GC_RECONCILE_QUOTA" in os.environ
                    else None
                ),
            }.items()
            if v is not None
        }

        # Handle document_ttl_days separately: both 0 and null/None mean "disabled".
        # Must not be filtered by `if v is not None` — an explicit 0 must override
        # a TOML value (e.g. document_ttl_days=90) and disable TTL.
        _ttl_env = os.environ.get("RAG_GC_TTL_DAYS")
        if _ttl_env is not None:
            overrides["document_ttl_days"] = None if int(_ttl_env) == 0 else int(_ttl_env)

        if overrides:
            return dataclasses.replace(base, **overrides)
        return base
