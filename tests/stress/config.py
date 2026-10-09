"""Stress test configuration — all tunables driven by environment variables.

Usage::

    from lasagna.tests.stress.config import StressConfig

    cfg = StressConfig.from_env()
    backend_cfg = cfg.for_backend("milvus-lite")  # backend-specific overrides
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class StressConfig:
    """Immutable stress-test configuration.

    Backend-specific fields (celery_enabled, compose_profiles, worker_mode)
    are defaults from ``from_env()``. Use ``for_backend()`` to get a copy
    with correct overrides for a specific vector backend.
    """

    # Infrastructure -----------------------------------------------------------
    api_base_url: str = "http://localhost:8000"
    compose_dir: str = ""  # resolved at construction
    vector_backend: str = "pgvector"

    # Scale parameters ---------------------------------------------------------
    num_tenants: int = 50
    docs_per_tenant: int = 10
    concurrent_searches: int = 100
    search_rounds: int = 5
    endurance_minutes: int = 10

    # Timing thresholds (seconds) ----------------------------------------------
    upload_timeout: float = 120.0
    search_p50_threshold: float = 2.0
    search_p95_threshold: float = 10.0
    ready_timeout: float = 180.0  # embedding model load can be slow
    latency_degradation_max: float = 5.0  # max acceptable degradation factor

    # Features -----------------------------------------------------------------
    enable_prometheus: bool = True
    enable_live_dashboard: bool = True

    # Run mode (derived from backend) ------------------------------------------
    celery_enabled: bool = True
    compose_profiles: List[str] = field(default_factory=list)
    worker_mode: str = "gunicorn-multi"  # or "uvicorn-single"

    # Architecture -------------------------------------------------------------
    architecture: str = field(default_factory=platform.machine)

    @classmethod
    def from_env(cls) -> StressConfig:
        """Build base config from environment variables.

        This creates a base config without backend-specific overrides.
        Call ``for_backend(name)`` to get a backend-specific copy.
        """
        default_compose = str(Path(__file__).resolve().parents[2] / "deploy")

        return cls(
            api_base_url=os.environ.get("STRESS_API_URL", "http://localhost:8000"),
            compose_dir=os.environ.get("STRESS_COMPOSE_DIR", default_compose),
            vector_backend="",  # filled by for_backend()
            num_tenants=int(os.environ.get("STRESS_NUM_TENANTS", "50")),
            docs_per_tenant=int(os.environ.get("STRESS_DOCS_PER_TENANT", "10")),
            concurrent_searches=int(os.environ.get("STRESS_CONCURRENT_SEARCHES", "100")),
            search_rounds=int(os.environ.get("STRESS_SEARCH_ROUNDS", "5")),
            endurance_minutes=int(os.environ.get("STRESS_ENDURANCE_MINUTES", "10")),
            upload_timeout=float(os.environ.get("STRESS_UPLOAD_TIMEOUT", "120")),
            search_p50_threshold=float(os.environ.get("STRESS_SEARCH_P50", "2.0")),
            search_p95_threshold=float(os.environ.get("STRESS_SEARCH_P95", "10.0")),
            ready_timeout=float(os.environ.get("STRESS_READY_TIMEOUT", "180")),
            enable_prometheus=True,
            enable_live_dashboard=os.environ.get("STRESS_DASHBOARD", "1") == "1",
        )

    def for_backend(self, backend: str) -> StressConfig:
        """Return a copy with backend-specific run mode overrides.

        This is the critical method that ensures celery_enabled, worker_mode,
        and compose_profiles are correct for the specific backend under test.
        """
        celery_enabled = backend != "milvus-lite"
        worker_mode = "uvicorn-single" if backend == "milvus-lite" else "gunicorn-multi"
        profiles: List[str] = ["celery"] if celery_enabled else []

        return replace(
            self,
            vector_backend=backend,
            celery_enabled=celery_enabled,
            worker_mode=worker_mode,
            compose_profiles=profiles,
        )

    @property
    def total_documents(self) -> int:
        return self.num_tenants * self.docs_per_tenant

    @staticmethod
    def _is_docker_gpu_available() -> bool:
        """Detect if the Docker container has GPU access.

        Note: this checks the TARGET environment (Docker), not the test host.
        MPS on macOS is NOT available inside Docker containers. CUDA requires
        nvidia-container-toolkit. Defaults to False (CPU-only) unless
        STRESS_GPU_AVAILABLE=1 is set.
        """
        return os.environ.get("STRESS_GPU_AVAILABLE", "0") == "1"

    @staticmethod
    def _is_lightweight_model() -> bool:
        """Check if a lightweight embedding model is configured.

        Lightweight models (all-MiniLM-L6-v2, etc.) are ~100x faster than
        bge-m3 on CPU, so CPU-mode threshold multipliers aren't needed.
        """
        model = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
        lightweight_models = {"all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-MiniLM-L12-v2"}
        return any(lm in model for lm in lightweight_models)

    @staticmethod
    def _is_lightweight_reranker() -> bool:
        """Check if a lightweight reranker model is configured.

        Lightweight rerankers (ms-marco-MiniLM, etc.) are ~20x faster than
        mxbai-rerank-large-v1 on CPU (~0.1s vs ~2-3s per query).
        """
        model = os.environ.get("RAG_RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-large-v1")
        lightweight = {"ms-marco-MiniLM", "ms-marco-TinyBERT", "MiniLM"}
        return any(lm in model for lm in lightweight)

    @property
    def _cpu_multiplier(self) -> float:
        """Adjust thresholds for model size and compute mode.

        Lightweight embedding + lightweight reranker: near GPU-speed, 1.0x.
        Lightweight embedding + heavy reranker: reranker still ~2-3s, 2.0x.
        Heavy embedding + heavy reranker: ~8-12s per query, 6.0x.
        """
        if self._is_docker_gpu_available():
            return 1.0
        if self._is_lightweight_model() and self._is_lightweight_reranker():
            return 1.0  # both models fast on CPU
        if self._is_lightweight_model():
            return 2.0  # reranker still CPU-bound
        return 6.0

    @property
    def search_p50_threshold_adjusted(self) -> float:
        """Higher thresholds for single-worker backends and CPU-only mode."""
        base = self.search_p50_threshold * self._cpu_multiplier
        if self.worker_mode == "uvicorn-single":
            return base * 2.0
        return base

    @property
    def search_p95_threshold_adjusted(self) -> float:
        base = self.search_p95_threshold * self._cpu_multiplier
        if self.worker_mode == "uvicorn-single":
            return base * 2.0
        return base

    @property
    def latency_degradation_max_adjusted(self) -> float:
        """CPU-only mode has higher latency variance under load."""
        if self._is_docker_gpu_available():
            return self.latency_degradation_max
        return self.latency_degradation_max * 2.0
