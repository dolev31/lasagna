"""Thread-safe singleton registry for benchmark adapters.

Mirrors the ``ComponentRegistry`` pattern from
:mod:`lasagna.core.registry`.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Type

import os

from loguru import logger

from ._base import BaseBenchmarkAdapter, BenchmarkInfo


class BenchmarkRegistry:
    """Global registry of benchmark adapters.

    Usage::

        from lasagna.knowledge_evaluation_sdk.benchmarks import (
            benchmark_registry,
        )

        # Discover all registered benchmarks
        for name, info in benchmark_registry.list_benchmarks().items():
            print(f"{name}: {info.description}")

        # Load a benchmark
        adapter = benchmark_registry.get("beir/nfcorpus")
        samples = adapter.load(n_samples=100)

        # Register a custom benchmark
        @benchmark_registry.register("my_bench")
        class MyBenchmark(BaseBenchmarkAdapter):
            ...
    """

    _instance: Optional["BenchmarkRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "BenchmarkRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._adapters: Dict[str, BaseBenchmarkAdapter] = {}
        self._data_lock = threading.Lock()
        self._builtins_registered = False
        # Names the package itself registers, as opposed to plugins and files
        # registered at run time; what the docs count.
        self._builtin_names: frozenset = frozenset()

    # ------------------------------------------------------------------
    # Built-in registration (lazy, thread-safe)
    # ------------------------------------------------------------------

    def _ensure_builtins(self) -> None:
        if self._builtins_registered:
            return
        with self._lock:
            if self._builtins_registered:
                return
            self._register_builtins()
            self._discover_entry_points()
            self._builtins_registered = True

    def _register_builtins(self) -> None:
        """Import and register built-in adapters."""
        try:
            from .adapters import register_all

            register_all(self)
            with self._data_lock:
                self._builtin_names = frozenset(self._adapters)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to register built-in benchmarks: {exc}")

    def _discover_entry_points(self) -> None:
        """Discover third-party adapters via ``lasagna_benchmarks`` entry-point group."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            # Python 3.12+ returns a SelectableGroups; 3.10-3.11 returns dict
            if isinstance(eps, dict):
                group = eps.get("lasagna_benchmarks", [])
            else:
                group = eps.select(group="lasagna_benchmarks")

            for ep in group:
                try:
                    from lasagna.core.registry import _record_discovery

                    adapter_cls = ep.load()
                    if isinstance(adapter_cls, type) and issubclass(
                        adapter_cls, BaseBenchmarkAdapter
                    ):
                        self.register_adapter(ep.name, adapter_cls())
                    elif isinstance(adapter_cls, BaseBenchmarkAdapter):
                        self.register_adapter(ep.name, adapter_cls)
                    else:
                        raise TypeError(
                            f"expected a BaseBenchmarkAdapter subclass or instance, "
                            f"got {type(adapter_cls).__name__}"
                        )
                    _record_discovery("lasagna_benchmarks", ep.name, ep.value, ok=True)
                except Exception as exc:  # noqa: BLE001
                    # Recorded so `lasagna plugins list` can show a benchmark
                    # that failed to load. Otherwise the dataset is simply
                    # absent and its author has nothing to debug from.
                    try:
                        from lasagna.core.registry import _record_discovery as _rec

                        _rec("lasagna_benchmarks", ep.name, ep.value, ok=False, error=exc)
                    except Exception:
                        pass
                    logger.warning(f"Failed to load entry point '{ep.name}': {exc}")
                    if os.environ.get("LASAGNA_STRICT_PLUGINS"):
                        raise
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Entry-point discovery skipped: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
    ) -> Callable[[Type[BaseBenchmarkAdapter]], Type[BaseBenchmarkAdapter]]:
        """Decorator to register a benchmark adapter class.

        Example::

            @benchmark_registry.register("my_bench")
            class MyBenchmark(BaseBenchmarkAdapter):
                ...
        """

        def decorator(cls: Type[BaseBenchmarkAdapter]) -> Type[BaseBenchmarkAdapter]:
            self.register_adapter(name, cls())
            return cls

        return decorator

    def register_adapter(self, name: str, adapter: BaseBenchmarkAdapter) -> None:
        """Register an adapter instance under *name*."""
        with self._data_lock:
            if name in self._adapters:
                logger.debug(f"Overwriting benchmark adapter: {name}")
            self._adapters[name] = adapter

    def get(self, name: str) -> BaseBenchmarkAdapter:
        """Retrieve a registered adapter by name.

        Raises:
            KeyError: If no adapter is registered under *name*.
        """
        self._ensure_builtins()
        with self._data_lock:
            if name not in self._adapters:
                available = ", ".join(sorted(self._adapters.keys()))
                raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
            return self._adapters[name]

    def list_benchmarks(self) -> Dict[str, BenchmarkInfo]:
        """Return ``{name: BenchmarkInfo}`` for all registered benchmarks."""
        self._ensure_builtins()
        with self._data_lock:
            return {name: adapter.info() for name, adapter in sorted(self._adapters.items())}

    def names(self) -> List[str]:
        """Return sorted list of registered benchmark names."""
        self._ensure_builtins()
        with self._data_lock:
            return sorted(self._adapters.keys())

    def is_builtin(self, name: str) -> bool:
        """Whether *name* ships with the package (not a plugin or a local file)."""
        self._ensure_builtins()
        return name in self._builtin_names

    def __contains__(self, name: str) -> bool:
        self._ensure_builtins()
        with self._data_lock:
            return name in self._adapters

    def __len__(self) -> int:
        self._ensure_builtins()
        with self._data_lock:
            return len(self._adapters)


# Module-level singleton
benchmark_registry = BenchmarkRegistry()
