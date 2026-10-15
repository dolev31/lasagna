"""The docs count the benchmarks the package ships, not what is installed beside it.

With the plugin template installed, its `template_tiny` benchmark joined the
registry and three tests that check the README's benchmark count failed on a
correct README. A user's own plugins or local data files must not make the
package's documentation tests fail either.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.smoke import SmokeBenchmarkAdapter


def test_a_benchmark_registered_at_run_time_is_not_builtin():
    assert benchmark_registry.is_builtin("smoke")
    benchmark_registry.register_adapter("test_run_time_only", SmokeBenchmarkAdapter())
    try:
        assert "test_run_time_only" in benchmark_registry
        assert not benchmark_registry.is_builtin("test_run_time_only")
    finally:
        with benchmark_registry._data_lock:
            benchmark_registry._adapters.pop("test_run_time_only", None)
