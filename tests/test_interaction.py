"""Whether two techniques compose is the question two papers cannot answer.

Technique A is published against a baseline. Technique B is published against a
baseline. Whether A+B is worth A alone does not follow from those two numbers,
because the interaction is not a function of the main effects: both techniques
may be fixing the same queries, in which case stacking them buys far less than
the sum of their headline gains.

Answering it needs all four cells of a 2x2 measured on the same queries, and
until now this package could not even express the design. The dimension lists
have no *off* level -- a `null` entry leaves the profile's default in place --
and the `config_overrides` branch that can express "off" disables the dimension
lists and names its arms `config_0`. The shipped `beir_sweep.yaml` demonstrated
the failure: `query_enhancers: [hyde, null]` produced two identical arms,
because swapping the component never sets `enable_hyde` and `standard` ships it
false. The sweep read as evidence that HyDE does not help.

README.md quoted an "8.4% composition efficiency" computed by hand, with no code
in the package that could produce it.
"""

from __future__ import annotations

import math

import pytest

from lasagna.knowledge_evaluation_sdk.experiment import (
    FactorialSpec,
    PipelineSweepSpec,
    _generate_pipeline_configs,
)
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult

CELLS = ["neither", "hyde", "reranking", "hyde+reranking"]


def _shipped_experiments():
    """Every worked example this package installs, as (path, pipeline spec)."""
    from pathlib import Path

    import yaml

    import lasagna

    root = Path(lasagna.__file__).parent / "knowledge_evaluation_sdk/research/experiments"
    found = [
        (path, (yaml.safe_load(path.read_text()) or {}).get("pipeline") or {})
        for path in sorted(root.glob("*.yaml"))
    ]
    assert found, f"no shipped experiments found under {root}"
    return found


def _results(per_cell, benchmark="scifact", metric="ndcg@10"):
    """A results object whose four cells score `metric` on the same samples."""
    return BenchmarkResults(
        [
            TaskResult(
                task_key=f"{benchmark}|{name}",
                benchmark_name=benchmark,
                config_name=name,
                run_index=0,
                metrics={metric: sum(scores) / len(scores)},
                per_sample_scores=[
                    {"_sample_id": f"q{i}", metric: s} for i, s in enumerate(scores)
                ],
                n_samples=len(scores),
            )
            for name, scores in per_cell.items()
        ]
    )


def _planted(n=60, base=0.40, eff_a=0.05, eff_b=0.05, interaction=0.0):
    """Four cells built from a known truth, with per-query noise.

    Deterministic: the same query gets the same wobble in all four cells, so
    the paired design sees the planted effect and not the noise.
    """
    wobble = [((i * 37) % 11 - 5) / 100.0 for i in range(n)]
    return {
        "neither": [base + w for w in wobble],
        "hyde": [base + eff_a + w for w in wobble],
        "reranking": [base + eff_b + w for w in wobble],
        "hyde+reranking": [base + eff_a + eff_b + interaction + w for w in wobble],
    }


class TestTheStatisticRecoversWhatWasPlanted:
    def test_a_purely_additive_design_reports_no_interaction(self):
        report = _results(_planted(interaction=0.0)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=2000,
        )
        assert report["interaction"] == pytest.approx(0.0, abs=1e-9)
        assert report["ci_lower"] <= 0.0 <= report["ci_upper"]
        assert not report["significant"]
        assert "additive" in report["verdict"]

    def test_a_planted_synergy_is_recovered(self):
        report = _results(_planted(interaction=0.04)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=2000,
        )
        assert report["interaction"] == pytest.approx(0.04, abs=1e-9)
        assert report["ci_lower"] > 0
        assert "synergistic" in report["verdict"]

    def test_a_planted_redundancy_is_recovered(self):
        """The common case: two techniques fixing the same queries."""
        report = _results(_planted(eff_a=0.05, eff_b=0.05, interaction=-0.04)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=2000,
        )
        assert report["interaction"] == pytest.approx(-0.04, abs=1e-9)
        assert report["ci_upper"] < 0
        assert "redundant" in report["verdict"]

    def test_main_effects_are_reported_alongside(self):
        report = _results(_planted(eff_a=0.05, eff_b=0.02)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=500,
        )
        assert report["main_effect_a"] == pytest.approx(0.05)
        assert report["main_effect_b"] == pytest.approx(0.02)
        assert report["expected_if_additive"] == pytest.approx(0.07)

    def test_composition_efficiency_is_the_readme_number(self):
        """Half the additive prediction delivered reads as 50%."""
        report = _results(_planted(eff_a=0.04, eff_b=0.04, interaction=-0.04)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=500,
        )
        assert report["composition_efficiency"] == pytest.approx(0.5)

    def test_efficiency_is_none_when_nothing_moved(self):
        """0.04/0.0 is not "they composed perfectly"."""
        report = _results(_planted(eff_a=0.0, eff_b=0.0, interaction=0.03)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=500,
        )
        assert report["composition_efficiency"] is None


class TestAlignment:
    """The double difference subtracts quantities measured on four cells.

    If those cells are not the same queries, the result is a number with no
    meaning, and nothing about it looks wrong.
    """

    def test_the_intersection_is_taken_across_all_four_at_once(self):
        cells = _planted(n=10)
        # Each cell is missing a different query. The four-way intersection is
        # 6; a pairwise alignment would find more for some pairs.
        drop = {"neither": 0, "hyde": 1, "reranking": 2, "hyde+reranking": 3}
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key=name,
                    benchmark_name="scifact",
                    config_name=name,
                    run_index=0,
                    per_sample_scores=[
                        {"_sample_id": f"q{i}", "ndcg@10": s}
                        for i, s in enumerate(scores)
                        if i != drop[name]
                    ],
                )
                for name, scores in cells.items()
            ]
        )
        report = results.interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=200,
        )
        assert report["n_paired"] == 6

    def test_the_reported_n_is_the_intersection_not_a_pairs(self):
        """A pairwise n would make the interval look better than the design
        supports."""
        ids, scores = _results(_planted(n=25)).aligned_scores("scifact", CELLS, "ndcg@10")
        assert len(ids) == 25
        assert all(len(v) == 25 for v in scores.values())

    def test_alignment_is_by_sample_id_not_position(self):
        """Reversed row order must not change the answer."""
        cells = _planted(n=12)
        forward = _results(cells)

        def _rows(scores):
            return [{"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)]

        shuffled = BenchmarkResults(
            [
                TaskResult(
                    task_key=name,
                    benchmark_name="scifact",
                    config_name=name,
                    run_index=0,
                    per_sample_scores=(
                        list(reversed(_rows(scores))) if name == "hyde" else _rows(scores)
                    ),
                )
                for name, scores in cells.items()
            ]
        )
        kwargs = dict(
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=200,
        )
        a = forward.interaction_test("scifact", "ndcg@10", **kwargs)
        b = shuffled.interaction_test("scifact", "ndcg@10", **kwargs)
        assert a["interaction"] == pytest.approx(b["interaction"])


class TestItRefusesWhatItCannotEstimate:
    def test_a_repeated_cell_is_rejected(self):
        """A 2x2 with a duplicate reports zero interaction by construction."""
        with pytest.raises(ValueError, match="distinct"):
            _results(_planted()).interaction_test(
                "scifact",
                "ndcg@10",
                neither="neither",
                only_a="hyde",
                only_b="hyde",
                both="hyde+reranking",
            )

    def test_a_missing_cell_names_itself(self):
        with pytest.raises(ValueError, match="no_such_config"):
            _results(_planted()).interaction_test(
                "scifact",
                "ndcg@10",
                neither="neither",
                only_a="no_such_config",
                only_b="reranking",
                both="hyde+reranking",
            )

    def test_a_single_shared_sample_is_refused(self):
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key=name,
                    benchmark_name="scifact",
                    config_name=name,
                    run_index=0,
                    per_sample_scores=[{"_sample_id": "q0", "ndcg@10": 0.5 + i / 10}],
                )
                for i, name in enumerate(CELLS)
            ]
        )
        with pytest.raises(ValueError, match="fewer than two"):
            results.interaction_test(
                "scifact",
                "ndcg@10",
                neither="neither",
                only_a="hyde",
                only_b="reranking",
                both="hyde+reranking",
            )

    def test_a_metric_nobody_scored_is_refused(self):
        with pytest.raises(ValueError, match="mrr@10"):
            _results(_planted()).interaction_test(
                "scifact",
                "mrr@10",
                neither="neither",
                only_a="hyde",
                only_b="reranking",
                both="hyde+reranking",
            )


class TestTheDesignCanBeExpressed:
    def test_four_named_cells_are_generated(self):
        configs = _generate_pipeline_configs(
            PipelineSweepSpec(
                factorial=FactorialSpec(factor_a="enable_hyde", factor_b="enable_reranking")
            )
        )
        assert [c["name"] for c in configs] == CELLS

    def test_both_levels_are_recorded_including_off(self):
        """`off` must be explicit. A cell that merely omits the flag records
        nothing, and "we turned it off" becomes indistinguishable from "we
        forgot" when someone reads the results file."""
        configs = _generate_pipeline_configs(
            PipelineSweepSpec(
                factorial=FactorialSpec(factor_a="enable_hyde", factor_b="enable_reranking")
            )
        )
        by_name = {c["name"]: c["overrides"] for c in configs}
        assert by_name["neither"] == {"enable_hyde": False, "enable_reranking": False}
        assert by_name["hyde"] == {"enable_hyde": True, "enable_reranking": False}
        assert by_name["hyde+reranking"] == {"enable_hyde": True, "enable_reranking": True}

    def test_held_fixed_reaches_every_cell(self):
        configs = _generate_pipeline_configs(
            PipelineSweepSpec(
                factorial=FactorialSpec(
                    factor_a="enable_hyde",
                    factor_b="enable_reranking",
                    held_fixed={"enable_query_classification": False},
                )
            )
        )
        assert all(c["overrides"]["enable_query_classification"] is False for c in configs)

    def test_the_factorial_takes_precedence_over_the_dimension_lists(self):
        configs = _generate_pipeline_configs(
            PipelineSweepSpec(
                rerankers=["cross_encoder", "combined"],
                factorial=FactorialSpec(factor_a="enable_hyde", factor_b="enable_mmr"),
            )
        )
        assert len(configs) == 4

    def test_no_factorial_leaves_the_old_behaviour_alone(self):
        configs = _generate_pipeline_configs(PipelineSweepSpec(rerankers=["a", "b"]))
        assert len(configs) == 2


class TestValidationHappensBeforeTheSweep:
    """A wrong factorial costs a full run and then reports an interaction
    estimate — a number indistinguishable from a real one."""

    def test_a_component_slot_is_refused_with_the_flag_to_use_instead(self):
        with pytest.raises(ValueError, match="enable_reranking"):
            FactorialSpec(factor_a="reranker", factor_b="enable_mmr").validate()

    def test_an_unknown_setting_is_refused_and_lists_candidates(self):
        with pytest.raises(ValueError) as exc:
            FactorialSpec(factor_a="enable_telepathy", factor_b="enable_mmr").validate()
        assert "enable_hyde" in str(exc.value)

    def test_the_same_factor_twice_is_refused(self):
        with pytest.raises(ValueError, match="zero by construction"):
            FactorialSpec(factor_a="enable_mmr", factor_b="enable_mmr").validate()

    def test_a_half_specified_factorial_is_refused(self):
        with pytest.raises(ValueError, match="both"):
            FactorialSpec(factor_a="enable_mmr").validate()

    @pytest.mark.parametrize(
        "flag", ["enable_hyde", "enable_reranking", "enable_mmr", "use_agentic_rag"]
    )
    def test_the_flags_the_docs_name_are_all_real(self, flag):
        """Validation reads the live config, so this catches a renamed setting
        rather than agreeing with a stale hand-written list."""
        FactorialSpec(factor_a=flag, factor_b="enable_crag").validate()


class TestTheNullTrapIsAnError:
    """`null` in a dimension list reads as "off" and is not.

    It leaves the profile default in place while still appearing in the config
    name and in `config_overrides`, so the arm looks like a treatment. When the
    profile already has the feature off, the arms are identical and the sweep
    reads as evidence the technique does not help.
    """

    def test_a_null_entry_is_refused(self):
        with pytest.raises(ValueError, match="null"):
            _generate_pipeline_configs(PipelineSweepSpec(query_enhancers=["hyde", None]))

    def test_the_error_points_at_the_factorial(self):
        with pytest.raises(ValueError, match="factorial"):
            _generate_pipeline_configs(PipelineSweepSpec(rerankers=["cross_encoder", None]))

    def test_no_shipped_example_contains_one(self):
        """A worked example that measures nothing teaches the failure."""
        for path, spec in _shipped_experiments():
            for key in ("rerankers", "embedders", "chunkers", "query_enhancers"):
                assert None not in (spec.get(key) or []), f"{path.name}: null in {key}"

    def test_every_shipped_example_generates_distinct_arms(self):
        """The stronger property. `null` was one way to produce identical arms;
        this catches the next one too."""
        for path, spec in _shipped_experiments():
            configs = _generate_pipeline_configs(
                PipelineSweepSpec(
                    base_profile=spec.get("base_profile", "standard"),
                    rerankers=spec.get("rerankers", []),
                    embedders=spec.get("embedders", []),
                    chunkers=spec.get("chunkers", []),
                    query_enhancers=spec.get("query_enhancers", []),
                    config_overrides=spec.get("config_overrides", []),
                    factorial=FactorialSpec(**(spec.get("factorial") or {})),
                )
            )
            seen = [tuple(sorted(c["overrides"].items())) for c in configs]
            assert len(seen) == len(set(seen)), f"{path.name}: duplicate arms in {seen}"

    def test_every_shipped_factorial_validates(self):
        for path, spec in _shipped_experiments():
            if spec.get("factorial"):
                FactorialSpec(**spec["factorial"]).validate()

    def test_every_shipped_factor_is_read_by_the_path_the_run_takes(self):
        """`validate()` only checks a flag EXISTS in the config.

        A flag can exist, be settable, be recorded in `config_overrides`, and be
        read by nothing on the path the experiment actually runs — in which case
        two of the four cells are behaviourally identical and the interaction is
        exactly 0 by construction, reported with a confident CI.

        `multihop_analysis.yaml` shipped exactly that: `enable_query_decomposition`
        is read only by the agentic graph's `plan_retrieval` node, never by the
        standard search path a live retrieval run takes. The arms differed in
        their override dicts, so `test_every_shipped_example_generates_distinct_arms`
        certified them as distinct.

        **This check is weaker than it looks, and the limit is stated here so
        nobody mistakes it for a proof.** It asserts the flag is *read*, not
        that the two factors vary *independently*. The first version of it
        matched comments, docstrings and REST request models, and passed on the
        exact YAML it was written to reject. It also certified
        `enable_rag_fusion` x `enable_query_expansion`, which are OR'd into one
        branch — the flag is read, and three of the four cells still run an
        identical search. `_COUPLED_FLAG_PAIRS` handles that case; this handles
        the "read by nothing on this path" case.
        """
        import ast
        from pathlib import Path

        import lasagna

        root = Path(lasagna.__file__).parent
        # The search path only. `api/rest/models.py` declares the flag as a
        # request field and `routes/config.py` echoes it back -- neither is a
        # retrieval decision, and matching them is how a dead factor passed.
        consumers = [root / "api" / "_search_orchestrator.py"] + [
            p for p in (root / "retrieval").rglob("*.py")
        ]

        def _names_read(path: Path) -> set:
            """Attribute and string names in executable code, not comments.

            Parsed rather than grepped: a comment mentioning a flag is not a
            reader, and the previous version could not tell the difference.
            """
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found: set = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    found.add(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    found.add(node.value)
            return found

        read_names: set = set()
        for path in consumers:
            read_names |= _names_read(path)

        for path, spec in _shipped_experiments():
            factorial = spec.get("factorial") or {}
            for role in ("factor_a", "factor_b"):
                flag = factorial.get(role)
                if not flag:
                    continue
                assert flag in read_names, (
                    f"{path.name}: {role}={flag!r} is not read in executable code on "
                    f"the search path (_search_orchestrator.py, lasagna/retrieval), so "
                    f"the 2x2 cannot measure it and the interaction would be zero by "
                    f"construction"
                )

    def test_the_reader_check_finds_the_flags_the_examples_depend_on(self):
        """Guards the guard: the check must be able to see a real reader.

        A check that matched nothing would pass vacuously if the shipped YAMLs
        ever lost their `factorial` blocks.

        Deliberately *not* asserting that some flag is absent. The obvious
        candidate — `enable_query_decomposition`, which this file's first fix
        removed — turns out to be read at `retrieval/enhancement/enhancer.py:558`
        and `:820`, so "has no reader in lasagna/retrieval" is false for it. That
        was the stated reason for the first repointing and it was wrong; the
        defensible reason is narrower, and it is in the YAML.
        """
        import ast
        from pathlib import Path

        import lasagna

        root = Path(lasagna.__file__).parent
        consumers = [root / "api" / "_search_orchestrator.py"] + list(
            (root / "retrieval").rglob("*.py")
        )
        names: set = set()
        for path in consumers:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names.add(node.value)

        assert "enable_query_expansion" in names
        assert "enable_reranking" in names
        # A flag that is genuinely not a retrieval setting must not match, or
        # the check passes for anything.
        assert "enable_prometheus" not in names

    def test_coupled_factors_are_refused(self):
        """A flag can be read and still be inert *as a factor*.

        `enable_rag_fusion` and `enable_query_expansion` are OR'd into one
        condition that delegates to `search_multi_query`, and both n-query
        settings default to 3 — so "expansion", "fusion" and "both" run the
        identical search. Three of four cells collapse and the interaction is
        exactly 0, reported with a confident interval.
        """
        with pytest.raises(ValueError, match="zero by construction"):
            FactorialSpec(
                factor_a="enable_rag_fusion", factor_b="enable_query_expansion"
            ).validate()

    def test_the_coupled_list_only_holds_real_flags(self):
        """A typo there would silently stop refusing anything."""
        from lasagna.knowledge_evaluation_sdk.experiment import (
            _boolean_config_flags,
            _COUPLED_FLAG_PAIRS,
        )

        flags = _boolean_config_flags()
        for pair in _COUPLED_FLAG_PAIRS:
            assert pair <= flags, f"{sorted(pair)} are not both real config flags"


class TestReportShape:
    def test_every_field_a_reader_needs_is_present(self):
        report = _results(_planted()).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=200,
        )
        for key in (
            "interaction",
            "ci_lower",
            "ci_upper",
            "p_value",
            "n_paired",
            "effect_size",
            "verdict",
            "main_effect_a",
            "main_effect_b",
        ):
            assert key in report

    def test_the_p_value_can_never_be_exactly_zero(self):
        """A finite resample cannot support p=0; the continuity correction is
        what stops "p < 0.001" claims the data cannot carry."""
        report = _results(_planted(interaction=0.5)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=200,
        )
        assert report["p_value"] > 0

    def test_it_is_reproducible(self):
        kwargs = dict(
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=500,
            seed=7,
        )
        a = _results(_planted()).interaction_test("scifact", "ndcg@10", **kwargs)
        b = _results(_planted()).interaction_test("scifact", "ndcg@10", **kwargs)
        assert a["ci_lower"] == b["ci_lower"] and a["ci_upper"] == b["ci_upper"]

    def test_the_effect_size_does_not_grow_with_sample_size(self):
        """Cohen's d_z, not the observed difference over a bootstrap SD --
        which grows as sqrt(n) and makes the same effect look stronger merely
        for having been measured on more queries."""
        small = _results(_planted(n=30, interaction=0.03)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=200,
        )
        large = _results(_planted(n=240, interaction=0.03)).interaction_test(
            "scifact",
            "ndcg@10",
            neither="neither",
            only_a="hyde",
            only_b="reranking",
            both="hyde+reranking",
            n_bootstrap=200,
        )
        # Planted with no noise on the interaction term, so d_z is unbounded in
        # both; what matters is that they do not differ by a factor of sqrt(8).
        assert math.isinf(small["effect_size"]) == math.isinf(large["effect_size"])


class TestTheInteractionCommand:
    """Driven through the CLI, which is where a reader meets these numbers."""

    def _file(self, tmp_path, cells, retrieved=None):
        import json

        rows = []
        for name, scores in cells.items():
            per_sample = []
            for i, s in enumerate(scores):
                row = {"_sample_id": f"q{i}", "ndcg@10": s}
                if retrieved is not None:
                    row["_retrieved_ids"] = retrieved[name][i]
                per_sample.append(row)
            rows.append(
                {
                    "task_key": name,
                    "run_index": 0,
                    "benchmark_name": "scifact",
                    "config_name": name,
                    "metrics": {},
                    "per_sample_scores": per_sample,
                }
            )
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"results": rows}))
        return str(path)

    def _invoke(self, path, *extra):
        typer_testing = pytest.importorskip("typer.testing")
        from lasagna.cli import app

        return typer_testing.CliRunner().invoke(
            app, ["results", "interaction", path, "--neither", "neither", *extra]
        )

    def test_the_interval_is_labelled_with_its_own_level(self, tmp_path):
        """`--alpha 0.2` printed an 80% interval under the label "95% CI"."""
        path = self._file(tmp_path, _planted(n=30, interaction=0.01))
        out = self._invoke(path, "--alpha", "0.2")
        assert out.exit_code == 0, out.output
        assert "80% CI" in out.output and "95% CI" not in out.output

    def test_a_cell_identical_to_another_is_named(self, tmp_path):
        """In a rerank x MMR factorial on `speed`, the MMR cell returned the
        same ranked lists as `neither` on 12 of 12 queries, and the command
        reported a significant "redundant" interaction from a factor that did
        nothing."""
        cells = _planted(n=12, eff_a=0.05, eff_b=0.0, interaction=-0.03)
        lists = {
            "neither": [[f"d{i}", "x"] for i in range(12)],
            "hyde": [[f"h{i}", "x"] for i in range(12)],
            "reranking": [[f"d{i}", "x"] for i in range(12)],
            "hyde+reranking": [[f"b{i}", "x"] for i in range(12)],
        }
        out = self._invoke(self._file(tmp_path, cells, lists))
        assert out.exit_code == 0, out.output
        assert "identical ranked lists" in out.output
        assert "'neither'" in out.output and "'reranking'" in out.output

    def test_distinct_cells_get_no_warning(self, tmp_path):
        lists = {
            name: [[f"{name}{i}"] for i in range(12)]
            for name in ("neither", "hyde", "reranking", "hyde+reranking")
        }
        out = self._invoke(self._file(tmp_path, _planted(n=12), lists))
        assert out.exit_code == 0, out.output
        assert "identical ranked lists" not in out.output

    def test_config_names_are_printed_literally(self, tmp_path):
        cells = {
            "neither": [0.4 + 0.01 * (i % 5) for i in range(12)],
            "a [old]": [0.45 + 0.01 * (i % 5) for i in range(12)],
            "b": [0.42 + 0.01 * (i % 3) for i in range(12)],
            "both": [0.5 + 0.01 * (i % 4) for i in range(12)],
        }
        out = self._invoke(
            self._file(tmp_path, cells),
            "--only-a",
            "a [old]",
            "--only-b",
            "b",
            "--both",
            "both",
        )
        assert out.exit_code == 0, out.output
        assert "a [old]" in out.output
