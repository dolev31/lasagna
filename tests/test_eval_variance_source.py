"""What `± std` in a report actually measures.

Every markdown and LaTeX table this framework produces shows `mean ± std` over
`n_runs`. A reader takes that as run-to-run stability — "run it again and you
get roughly this". It was not: the runner drew a *different* query subset each
run (`seed + run_index`), so the spread was sampling variance of the metric over
the dataset, and the guide's advice that "retrieval metrics move between runs"
was true for a reason no reader would guess.

Both quantities are worth having. They are not the same quantity, so the choice
is explicit and recorded.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask
from lasagna.knowledge_evaluation_sdk.seeding import seed_everything


def _task(**kw) -> BenchmarkTask:
    base = dict(benchmark_name="smoke", config_name="c", config_overrides={}, run_index=0)
    base.update(kw)
    return BenchmarkTask(**base)


class TestVarianceSourceIsExplicit:
    def test_default_is_run_variance(self):
        """The default must match what the reports have always implied."""
        from lasagna.knowledge_evaluation_sdk.experiment import ExperimentConfig

        assert ExperimentConfig(name="x").variance_source == "run"
        assert _task().variance_source == "run"

    def test_run_variance_holds_the_query_subset_fixed(self):
        """Same queries every run, so any spread is the pipeline."""
        from tests.test_event_loop_safety import executable_source
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkRunner

        src = executable_source(BenchmarkRunner._execute_task)
        assert 'task.seed if task.variance_source == "run"' in src, (
            'with variance_source="run" the sampling seed must not include '
            "run_index, or every run draws a different subset"
        )

    def test_sample_variance_still_available(self):
        src_task = _task(variance_source="sample")
        assert src_task.variance_source == "sample"

    def test_config_threads_through_to_the_task(self):
        """A setting that never reaches the runner is decoration."""
        from tests.test_event_loop_safety import executable_source
        from lasagna.knowledge_evaluation_sdk.experiment import Experiment

        src = executable_source(Experiment.generate_tasks)
        assert "variance_source=cfg.variance_source" in src


class TestSeedingCoversTheProcess:
    def test_seed_everything_reports_what_it_seeded(self):
        """A provenance record must distinguish 'tried' from 'did'."""
        out = seed_everything(1234, quiet=True)
        assert out["seed"] == 1234
        assert out["python"] is True
        # numpy/torch are optional; the record says which were present.
        assert set(out) >= {"seed", "python", "numpy", "torch"}

    def test_python_rng_is_actually_seeded(self):
        import random

        seed_everything(7, quiet=True)
        first = [random.random() for _ in range(5)]
        seed_everything(7, quiet=True)
        assert [random.random() for _ in range(5)] == first

    def test_numpy_rng_is_actually_seeded(self):
        np = pytest.importorskip("numpy")

        seed_everything(11, quiet=True)
        first = np.random.rand(5).tolist()
        seed_everything(11, quiet=True)
        assert np.random.rand(5).tolist() == first

    def test_torch_rng_is_actually_seeded(self):
        torch = pytest.importorskip("torch")

        seed_everything(13, quiet=True)
        first = torch.rand(5).tolist()
        seed_everything(13, quiet=True)
        assert torch.rand(5).tolist() == first

    def test_the_runner_seeds_before_loading_samples(self):
        from tests.test_event_loop_safety import executable_source
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkRunner

        src = executable_source(BenchmarkRunner._execute_task)
        assert "seed_everything(" in src, "the process must be seeded per task"
        assert src.index("seed_everything(") < src.index("adapter.load("), (
            "seeding must happen before anything random runs"
        )

    def test_does_not_claim_bit_determinism(self):
        """`use_deterministic_algorithms` raises on ops sentence-transformers
        uses, and an API judge is outside our control. The honest claim is
        'seeded'."""
        from tests.test_event_loop_safety import executable_source
        from lasagna.knowledge_evaluation_sdk import seeding

        src = executable_source(seeding.seed_everything)
        assert "use_deterministic_algorithms" not in src
