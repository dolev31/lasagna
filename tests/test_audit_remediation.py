"""Regression cover for the remaining audit findings.

Each class corresponds to a finding that survived adversarial refutation. The
docstrings record what was wrong, because the value of these is mostly in
stopping the same reasoning error being made again.
"""

from __future__ import annotations

import pytest


class TestVerifiedMeansReproduced:
    """The leaderboard badge compared different systems.

    `mark_verified` gated on submitter, comparability key and the ±1σ interval —
    but the key *deliberately* excludes the dimension under test, so every arm
    of a sweep shares one key by construction. A baseline row could therefore be
    stamped "verified" because another lab's reranker arm landed inside its
    interval. `hub/README.md` publishes the badge as "a real reproduction
    signal rather than a badge".
    """

    def _row(self, name, submitter, mean, levels, key="k1"):
        return {
            "config": name,
            "submitter": submitter,
            "mean": mean,
            "std": 0.01,
            "n_runs": 3,
            "comparability_key": key,
            "varied_levels": levels,
            "benchmark": "b",
            # A current row records what each stage held fixed; rows that do
            # not are never verified across bundles.
            "held_fixed": {"fixed_stages": {"retriever": "r1"}},
        }

    def test_a_different_arm_does_not_verify_you(self):
        from lasagna.knowledge_evaluation_sdk.leaderboard import mark_verified

        rows = [
            self._row("baseline", "lab-a", 0.50, {"reranker": "none"}),
            self._row("with_reranker", "lab-b", 0.505, {"reranker": "bge"}),
        ]
        mark_verified(rows)
        assert not rows[0]["verified"], "a different system was accepted as a reproduction"
        assert not rows[1]["verified"]

    def test_the_same_arm_from_another_lab_does_verify_you(self):
        """The badge must still be earnable, or the fix is just a disable."""
        from lasagna.knowledge_evaluation_sdk.leaderboard import mark_verified

        rows = [
            self._row("baseline", "lab-a", 0.50, {"reranker": "none"}),
            self._row("baseline", "lab-b", 0.505, {"reranker": "none"}),
        ]
        mark_verified(rows)
        assert rows[0]["verified"]
        assert rows[1]["verified"]

    def test_your_own_second_run_still_does_not_count(self):
        from lasagna.knowledge_evaluation_sdk.leaderboard import mark_verified

        rows = [
            self._row("baseline", "lab-a", 0.50, {"reranker": "none"}),
            self._row("baseline", "lab-a", 0.505, {"reranker": "none"}),
        ]
        mark_verified(rows)
        assert not any(r["verified"] for r in rows)


class TestAggregatorWeightsNameRealMetrics:
    """Half the IR weight and all of the generation weight resolved to nothing.

    The defaults named ``map``/``mrr`` while the harness emits ``map@10``/
    ``mrr@10``, and ``judge_score``/``factual_accuracy``/``completeness``, none
    of which exist anywhere in this package. So ``gen_score`` was always 0.0 and
    the composite was silently halved.
    """

    def test_ir_weights_match_the_emitted_metric_names(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.aggregator import (
            _DEFAULT_IR_WEIGHTS,
        )

        emitted = {"ndcg@10", "map@10", "mrr@10", "recall@10", "precision@10", "hit_rate@10"}
        assert set(_DEFAULT_IR_WEIGHTS) <= emitted, (
            f"weights name metrics the harness never emits: "
            f"{sorted(set(_DEFAULT_IR_WEIGHTS) - emitted)}"
        )

    def test_generation_weights_match_ragas_metric_names(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.aggregator import (
            _DEFAULT_GEN_WEIGHTS,
        )

        ragas = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        assert set(_DEFAULT_GEN_WEIGHTS) <= ragas, (
            f"weights name metrics RAGAS never emits: {sorted(set(_DEFAULT_GEN_WEIGHTS) - ragas)}"
        )

    def test_the_weights_still_sum_to_one(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.aggregator import (
            _DEFAULT_GEN_WEIGHTS,
            _DEFAULT_IR_WEIGHTS,
        )

        assert sum(_DEFAULT_IR_WEIGHTS.values()) == pytest.approx(1.0)
        assert sum(_DEFAULT_GEN_WEIGHTS.values()) == pytest.approx(1.0)


class TestCostIsCountedPerQueryNotPerRow:
    """A live end-to-end run emits two rows per query — retrieval and
    generation — each with its own ``latency_ms``. Averaging over rows halved
    every cost figure in the table whose entire purpose is the cost axis.
    Measured on committed e2e results: 1.77x understated."""

    def _rows(self):
        return [
            {"_sample_id": "q1", "latency_ms": 100.0, "input_tokens": 10, "output_tokens": 5},
            {"_sample_id": "q1", "latency_ms": 300.0, "input_tokens": 20, "output_tokens": 5},
            {"_sample_id": "q2", "latency_ms": 200.0, "input_tokens": 10, "output_tokens": 10},
        ]

    def test_rows_sharing_a_sample_id_are_one_query(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.tradeoff_analyzer import (
            _per_query_totals,
        )

        totals = _per_query_totals(self._rows(), ("latency_ms",))
        assert totals == {"q1": 400.0, "q2": 200.0}

    def test_the_average_is_over_queries(self):
        import statistics

        from lasagna.knowledge_evaluation_sdk.private_eval.tradeoff_analyzer import (
            _per_query_totals,
        )

        per_query = statistics.mean(_per_query_totals(self._rows(), ("latency_ms",)).values())
        per_row = statistics.mean(r["latency_ms"] for r in self._rows())
        assert per_query == 300.0
        assert per_row == pytest.approx(200.0)
        assert per_query > per_row, "the bug understated cost; the fix must raise it"

    def test_tokens_sum_across_both_fields_and_both_rows(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.tradeoff_analyzer import (
            _per_query_totals,
        )

        assert _per_query_totals(self._rows(), ("input_tokens", "output_tokens")) == {
            "q1": 40,
            "q2": 20,
        }

    def test_offline_rows_without_a_sample_id_stay_one_query_each(self):
        """Retrieval-only runs have one row per query already; grouping them all
        under a single missing key would collapse the whole run into one."""
        from lasagna.knowledge_evaluation_sdk.private_eval.tradeoff_analyzer import (
            _per_query_totals,
        )

        rows = [{"latency_ms": 10.0}, {"latency_ms": 20.0}]
        assert sorted(_per_query_totals(rows, ("latency_ms",)).values()) == [10.0, 20.0]


class TestLatexDaggersAreCorrected:
    def test_it_uses_the_corrected_test(self):
        """The daggers were N-1 uncorrected comparisons against whichever arm
        scored highest — the textbook way to manufacture a false positive, in a
        table headed for a paper, from a package built to prevent that."""
        import ast
        import inspect
        import textwrap

        from lasagna.knowledge_evaluation_sdk.reporting import ReportGenerator

        tree = ast.parse(textwrap.dedent(inspect.getsource(ReportGenerator.latex_table)))
        called = {
            n.func.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "pairwise_significance" in called
        assert "significance_test" not in called


class TestCredentialsDoNotReachTheLog:
    """`session_token` is a signed JWT valid for seven days by default and
    `admin_token` authorises instance-wide destruction. Both were logged
    verbatim — by the shared tool-call logger, so by all 33 tools, not only the
    docs endpoint the audit named."""

    def test_tokens_are_redacted(self):
        from lasagna.mcp.server import _redact_credentials

        out = _redact_credentials(
            {"session_id": "s1", "session_token": "eyJ.SECRET", "admin_token": "hunter2"}
        )
        assert "SECRET" not in str(out)
        assert "hunter2" not in str(out)

    def test_presence_is_still_visible(self):
        """ "Was a token supplied?" is a real debugging question, so the key
        stays and only the value goes."""
        from lasagna.mcp.server import _redact_credentials

        assert "session_token" in _redact_credentials({"session_token": "x"})

    def test_non_credentials_are_untouched(self):
        from lasagna.mcp.server import _redact_credentials

        assert _redact_credentials({"top_k": 5, "session_id": "s"}) == {
            "top_k": 5,
            "session_id": "s",
        }

    def test_the_shared_logger_uses_it(self):
        """The defect was in `_log_tool_call`, which every tool routes through."""
        import ast
        import inspect
        import textwrap

        from lasagna.mcp import server

        tree = ast.parse(textwrap.dedent(inspect.getsource(server._log_tool_call)))
        called = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_redact_credentials" in called


class TestRenameProfileChecksOwnership:
    """`save_profile` and `delete_profile` both refuse when the global profile
    belongs to someone else. `rename_profile` mutated the same global store with
    no check, and its REST route did not even resolve a ``user_id``."""

    def test_it_accepts_a_user_id(self):
        import inspect

        from lasagna.core.session_config._profiles import ProfileMixin

        assert "user_id" in inspect.signature(ProfileMixin.rename_profile).parameters

    def test_the_route_resolves_one(self):
        import inspect

        from lasagna.api.rest.routes import knowledge_config

        assert "user_id" in inspect.signature(knowledge_config.rename_profile).parameters

    def test_it_matches_its_peers(self):
        """All three mutate the same global store; all three must gate the same
        way."""
        import inspect

        from lasagna.core.session_config._profiles import ProfileMixin

        for name in ("save_profile", "delete_profile", "rename_profile"):
            params = inspect.signature(getattr(ProfileMixin, name)).parameters
            assert "user_id" in params, f"{name} cannot check ownership"


class TestWindowsIsNotClaimed:
    def test_os_independent_is_not_declared(self):
        """milvus-lite has no Windows wheel and is the default vector store, so
        `pip install` succeeds and the first upload dies."""
        import pathlib

        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib

        pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        classifiers = tomllib.loads(pyproject.read_text())["project"]["classifiers"]
        assert "Operating System :: OS Independent" not in classifiers
        assert any(c.startswith("Operating System ::") for c in classifiers), (
            "no OS is claimed at all, which is also wrong"
        )


class TestDoctorAdviceIsActionable:
    def test_no_check_points_at_an_extra_that_lacks_the_package(self):
        """`pymilvus` mapped to `vector-backends`, which is
        pinecone/qdrant/weaviate. pymilvus is a core dependency, so the advice
        sent users somewhere that could not help them."""
        import pathlib

        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib

        from lasagna.core.diagnostics import _OPTIONAL_IMPORTS

        pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())["project"]
        extras = data["optional-dependencies"]
        core = " ".join(data["dependencies"])

        for module, (extra, _purpose) in _OPTIONAL_IMPORTS.items():
            if extra is None:
                continue
            names = " ".join(extras.get(extra, []))
            assert module.replace("_", "-") in names or module in names, (
                f"doctor sends a user missing {module!r} to the {extra!r} extra, "
                f"which does not contain it"
            )
            assert module.replace("_", "-") not in core, (
                f"{module!r} is a core dependency; no extra can fix it being absent"
            )
