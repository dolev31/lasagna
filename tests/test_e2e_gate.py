"""The release gate must be able to fail.

A gate that always passes is decoration, and worse than nothing — it converts
"nobody checked" into "checked and fine". These tests exercise the verdict
logic and the report against deliberately broken runs.

They test `e2e/`, not `lasagna` itself.
"""

import json

import pytest

from e2e.contract import (
    PhaseResult,
    RunResult,
    Severity,
    Status,
    capture_environment,
)
from e2e.report import load_previous, render


def _phase(key="p1", title="Install") -> PhaseResult:
    return PhaseResult(key=key, title=title)


def _run(*phases: PhaseResult) -> RunResult:
    return RunResult(
        started_at="2026-01-01 00:00:00 UTC", environment={"python": "3.12"}, phases=list(phases)
    )


class TestVerdict:
    def test_clean_run_is_go(self):
        p = _phase()
        p.check("wheel builds", True)
        assert _run(p).go is True

    def test_critical_failure_blocks_the_release(self):
        p = _phase()
        p.check("wheel builds", False, detail="build errored")
        run = _run(p)
        assert run.go is False
        assert len(run.blockers) == 1
        assert run.blockers[0].name == "wheel builds"

    def test_warning_failure_does_not_block(self):
        """A busy laptop missing a latency target must not stop a good release."""
        p = _phase()
        p.check("search p95 under 200ms", False, severity=Severity.WARN, detail="310ms")
        run = _run(p)
        assert run.go is True, "a WARN failure must not block"
        assert p.status is Status.WARNED, "but it must still be visible"

    def test_one_critical_failure_among_passes_still_blocks(self):
        p = _phase()
        p.check("a", True)
        p.check("b", False)
        p.check("c", True)
        assert _run(p).go is False

    def test_a_crashed_phase_does_not_hide_other_phases(self):
        """A phase that blows up must not take the run's other evidence with it."""
        good, bad = _phase("p1"), _phase("p2", "Cold start")
        good.check("wheel builds", True)
        good.measure("install time", 12.5, "s")
        bad.check("p2 completed", False, detail="RuntimeError")
        run = _run(good, bad)
        assert run.go is False
        assert run.phases[0].metrics, "metrics from the healthy phase must survive"


class TestUnrunIsNotPassed:
    """The distinction the whole design turns on."""

    def test_skipped_phase_is_unvalidated_not_passed(self):
        p = _phase("p8", "Quality")
        p.skipped_reason = "no LLM API key present"
        run = _run(p)
        assert p.status is Status.SKIPPED
        assert p in run.unvalidated
        assert run.go is True, "a skip must not block..."
        assert "unvalidated" in render(run, None).lower(), "...but must be reported as unknown"

    def test_skipped_check_is_not_counted_as_passing(self):
        p = _phase()
        p.skip("pgvector conformance", "service not reachable")
        assert p.checks[0].status is Status.SKIPPED
        assert p.checks[0].blocks_release is False
        assert p.status is Status.SKIPPED, "a phase whose checks all skipped has verified nothing"

    def test_report_names_why_a_phase_did_not_run(self):
        p = _phase("p8", "Quality")
        p.skipped_reason = "no LLM API key present"
        assert "no LLM API key present" in render(_run(p), None)


class TestReportIsActionable:
    def test_blocker_appears_with_its_repro_command(self):
        p = _phase()
        p.check(
            "import is clean", False, detail="torch loaded", repro="python -m e2e.run --phase p2"
        )
        text = render(_run(p), None)
        assert "NO-GO" in text
        assert "torch loaded" in text
        assert "python -m e2e.run --phase p2" in text, (
            "a failure a reader cannot reproduce is a rumour"
        )

    def test_metrics_render_even_when_the_phase_failed(self):
        """Measure first, assert after — the failed run is the interesting one."""
        p = _phase()
        p.measure("time to first search", 42.0, "s")
        p.check("first search returns", False, detail="timed out")
        text = render(_run(p), None)
        assert "time to first search" in text
        assert "42" in text

    def test_verdict_is_unambiguous(self):
        ok, bad = _phase("p1"), _phase("p2")
        ok.check("x", True)
        assert render(_run(ok), None).startswith("# Lasagna release gate — GO")
        bad.check("y", False)
        assert render(_run(bad), None).startswith("# Lasagna release gate — NO-GO")


class TestRegressionDetection:
    def _previous(self, value: float) -> dict:
        return {
            "phases": [
                {
                    "key": "p2",
                    "status": "passed",
                    "metrics": [
                        {
                            "name": "time to first search",
                            "value": value,
                            "unit": "s",
                            "lower_is_better": True,
                        }
                    ],
                }
            ]
        }

    def test_a_slide_is_called_out_as_worse(self):
        p = _phase("p2", "Cold start")
        p.measure("time to first search", 40.0, "s")  # was 20 → 100% slower
        text = render(_run(p), self._previous(20.0))
        assert "WORSE" in text, "a 100% latency regression must be flagged"

    def test_an_improvement_is_not_flagged_as_worse(self):
        p = _phase("p2", "Cold start")
        p.measure("time to first search", 10.0, "s")  # was 20 → twice as fast
        text = render(_run(p), self._previous(20.0))
        assert "better" in text
        assert "WORSE" not in text

    def test_small_movement_is_treated_as_noise(self):
        p = _phase("p2", "Cold start")
        p.measure("time to first search", 20.8, "s")  # +4%
        assert "noise" in render(_run(p), self._previous(20.0))

    def test_higher_is_better_metrics_are_not_inverted(self):
        """Throughput going up is an improvement, not a regression."""
        p = _phase("p5", "Backends")
        p.measure("ingest throughput", 200.0, "docs/s", lower_is_better=False)
        prev = {
            "phases": [
                {
                    "key": "p5",
                    "status": "passed",
                    "metrics": [
                        {
                            "name": "ingest throughput",
                            "value": 100.0,
                            "unit": "docs/s",
                            "lower_is_better": False,
                        }
                    ],
                }
            ]
        }
        text = render(_run(p), prev)
        assert "better" in text and "WORSE" not in text

    def test_a_phase_that_stops_running_is_called_out(self):
        """A phase quietly ceasing to run looks exactly like success."""
        p = _phase("p1", "Install")
        p.check("x", True)
        prev = {
            "phases": [
                {"key": "p1", "status": "passed", "metrics": []},
                {"key": "p5", "status": "passed", "metrics": []},
            ]
        }
        text = render(_run(p), prev)
        assert "did not run now" in text and "p5" in text


class TestEnvironmentCapture:
    def test_records_what_the_numbers_were_measured_on(self):
        env = capture_environment()
        for key in ("platform", "python", "git_commit", "llm_keys_present"):
            assert key in env, (
                f"report must record {key}; a p95 without a machine is not comparable"
            )

    def test_llm_keys_are_recorded_by_name_not_value(self):
        """The safety property is 'no values', not 'names match a pattern'.

        An earlier version asserted every name ended in `_KEY`, which was an
        incidental naming convention rather than the thing that matters —
        and it failed the moment WATSONX_APIKEY (IBM's own spelling) was
        supported.
        """
        import os

        env = capture_environment()
        names = env["llm_keys_present"]
        assert isinstance(names, list)

        # Env-var shaped: uppercase, underscores, digits.
        assert all(n.replace("_", "").isalnum() and n.isupper() for n in names), (
            f"expected environment variable names, got {names}"
        )

        # The real assertion: no actual secret value appears anywhere in the
        # environment block that gets written into a committed report.
        serialized = " ".join(str(v) for v in env.values())
        for name in names:
            value = os.environ.get(name)
            if value and len(value) > 8:
                assert value not in serialized, (
                    f"the value of {name} leaked into the report's environment block"
                )


class TestPreviousReportLoading:
    def test_returns_none_when_there_is_no_history(self, tmp_path):
        assert load_previous(tmp_path, exclude=tmp_path / "nope") is None

    def test_ignores_the_current_run(self, tmp_path):
        cur = tmp_path / "now"
        cur.mkdir()
        (cur / "results.json").write_text(json.dumps({"phases": []}))
        assert load_previous(tmp_path, exclude=cur) is None

    def test_survives_a_corrupt_report(self, tmp_path):
        """One unreadable report must not break the next run's diffing."""
        bad = tmp_path / "a"
        bad.mkdir()
        (bad / "results.json").write_text("{ not json")
        good = tmp_path / "b"
        good.mkdir()
        (good / "results.json").write_text(json.dumps({"phases": [{"key": "p1"}]}))
        loaded = load_previous(tmp_path, exclude=tmp_path / "none")
        assert loaded is None or loaded.get("phases")


class TestTheGateCannotVerifyNothingAndPass:
    """The failure mode the whole design exists to prevent, applied to itself.

    Both of these were real defects in the first version of the contract: an
    empty phase reported PASSED and an empty run reported GO. A gate that
    verified nothing and said GO is worse than no gate, because it converts
    "nobody checked" into "checked and fine".
    """

    def test_a_phase_that_asserts_nothing_is_not_a_pass(self):
        p = _phase()
        assert p.status is Status.FAILED, "a phase that ran and produced no checks verified nothing"
        assert p.blockers, "and it must block, not merely look bad in the report"

    def test_a_phase_with_nothing_to_do_must_say_so_explicitly(self):
        """The legitimate version of the above: skipped_reason, not silence."""
        p = _phase()
        p.skipped_reason = "requires Docker, which is not running"
        assert p.status is Status.SKIPPED
        assert not p.blockers, "an honest skip must not block"

    def test_a_run_with_no_phases_is_never_go(self):
        """Easy to reach by accident: a --phase filter that matches nothing."""
        run = _run()
        assert run.go is False
        assert "nothing was verified" in run.blockers[0].detail

    def test_an_empty_phase_drags_the_whole_run_to_no_go(self):
        good = _phase("p1")
        good.check("real check", True)
        empty = _phase("p2", "Cold start")
        assert _run(good, empty).go is False


class TestReportsDoNotPublishLocalPaths:
    """Reports are committed, so whatever they contain is published.

    Subprocess output, tracebacks and repro commands all carry absolute paths.
    An earlier set of committed reports held 641 references to the developer's
    home directory and 287 to the unrelated internal monorepo sitting beside
    this checkout — none of it secret, all of it published, and none of it any
    use to a reader.
    """

    def test_redactor_rewrites_the_paths_that_actually_appear(self):
        import os
        import sys
        from pathlib import Path

        from e2e.contract import redact_local_paths

        repo = str(Path(__file__).resolve().parents[1])
        home = os.path.expanduser("~")

        assert redact_local_paths(sys.executable) == "python"
        assert redact_local_paths(f"{repo}/src/x.py") == "<repo>/src/x.py"
        assert redact_local_paths(f"{home}/elsewhere/x") == "<home>/elsewhere/x"
        # macOS per-user temp roots embed a per-user token.
        assert redact_local_paths("/var/folders/xg/abc123/T/lasagna-p1/venv") == (
            "<tmp>/lasagna-p1/venv"
        )
        assert redact_local_paths("/private/var/folders/xg/abc/T/x") == "<tmp>/x"

    def test_redactor_reaches_into_nested_structures(self):
        """A path buried in a check detail is published just as loudly."""
        import os

        from e2e.contract import redact_local_paths

        home = os.path.expanduser("~")
        out = redact_local_paths(
            {"phases": [{"checks": [{"detail": f"failed at {home}/secret/spot"}]}]}
        )
        assert out["phases"][0]["checks"][0]["detail"] == "failed at <home>/secret/spot"

    def test_run_result_redacts_on_serialisation(self):
        """The guarantee has to hold at the point the report is written."""
        import os

        home = os.path.expanduser("~")
        p = _phase()
        p.check("x", False, detail=f"boom in {home}/x", repro=f"cd {home}/x && run")
        blob = json.dumps(_run(p).to_dict())
        assert home not in blob
        assert "<home>/x" in blob

    def test_committed_reports_contain_no_absolute_local_paths(self):
        """The property, asserted against what is actually in the repo."""
        import os
        import re
        from pathlib import Path

        home = os.path.expanduser("~")
        reports = Path(__file__).resolve().parents[1] / "e2e" / "reports"
        if not reports.is_dir():
            pytest.skip("no committed reports in this checkout")

        offenders = []
        for f in sorted(reports.rglob("*")):
            if not f.is_file():
                continue
            text = f.read_text(errors="replace")
            if home in text or re.search(r"(?:/private)?/var/folders/", text):
                offenders.append(f.name)
        assert not offenders, (
            f"committed reports leak local absolute paths: {sorted(set(offenders))}. "
            f"RunResult.to_dict() redacts them; a report written another way must too."
        )


class TestMetricFormatting:
    """A number in a report is read as a claim about precision."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, "0"),  # a count, not a 4-decimal measurement
            (3.0, "3"),
            (1502.0, "1,502"),
            (0.3, "0.3"),
            (0.863, "0.863"),
            (1.2, "1.20"),
            (61.3, "61.30"),
        ],
    )
    def test_renders_without_spurious_precision(self, value, expected):
        from e2e.report import _fmt

        assert _fmt(value) == expected

    def test_a_nonzero_measurement_never_renders_as_zero(self):
        """Rounding 1.2e-05 to "0" would claim an operation took no time."""
        from e2e.report import _fmt

        for tiny in (1.2e-05, 0.00004, -0.00002):
            rendered = _fmt(tiny)
            assert float(rendered) != 0.0, (
                f"{tiny!r} rendered as {rendered!r}, which reads as exactly zero"
            )


class TestSkipListsMatchTheChecksTheyStandIn:
    """A phase that stops early replaces its remaining checks with skips.

    If a name in `_skip_rest` drifts from the `p.check()` it stands in for, the
    report gains a skip for a check nobody ever emits, and the real check simply
    goes missing — which reads as "fine" to anyone skimming. Nothing else would
    catch that: both spellings are valid strings, and the mismatch only shows on
    the early-return path.

    Checked in the direction that has a single correct answer. Every skip-list
    entry must correspond to a real check; the reverse is deliberately not
    asserted, because checks upstream of the hard stop (the build itself) have
    already run and must not be in the list.
    """

    @staticmethod
    def _literal_prefix(node, module):
        """Leading constant text of a check-name expression, or None.

        Handles the three ways a phase spells one, because both are in use and
        both are fine:

        * a string literal, or an f-string (reduced to its constant head — the
          interpolated parts are loop variables, and a prefix is enough to tie
          a skip entry to the check it stands in for);
        * a module-level constant, resolved through the imported module;
        * ``CONST.format(...)``, resolved then truncated at the first field.

        Naming the constant is the better style — it makes drift impossible
        rather than merely detectable — so this must not push phases back
        toward duplicated literals by failing to understand it.
        """
        import ast

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            head = node.values[0] if node.values else None
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                return head.value
            return None
        if isinstance(node, ast.Name):
            value = getattr(module, node.id, None)
            return value.split("{")[0] if isinstance(value, str) else None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # CONST.format(...) -- the text before the first field is fixed
            if node.func.attr == "format" and isinstance(node.func.value, ast.Name):
                value = getattr(module, node.func.value.id, None)
                return value.split("{")[0] if isinstance(value, str) else None
        return None

    @classmethod
    def _check_name_prefixes(cls, module) -> list:
        """Leading constant text of every `p.check(...)`/`p.skip(...)` name."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(module))
        prefixes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr in ("check", "skip")):
                continue
            prefix = cls._literal_prefix(node.args[0], module)
            if prefix:
                prefixes.append(prefix)
        return prefixes

    @pytest.mark.parametrize("module_name", ["p1_install", "p9_lifecycle"])
    def test_every_skip_list_entry_names_a_real_check(self, module_name):
        import importlib
        import inspect

        module = importlib.import_module(f"e2e.phases.{module_name}")
        skip_rest = getattr(module, "_skip_rest", None)
        assert skip_rest is not None, f"{module_name} has no _skip_rest to verify"

        prefixes = self._check_name_prefixes(module)
        assert prefixes, "found no check names at all — the AST walk is broken"

        # The names _skip_rest iterates over — read from the loop's own tuple,
        # not from every string in the function, which would sweep up the
        # docstring.
        import ast
        import textwrap

        listed = []
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(skip_rest)))):
            if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
                for element in node.iter.elts:
                    prefix = self._literal_prefix(element, module)
                    if prefix:
                        listed.append(prefix)
        assert listed, f"{module_name}._skip_rest lists no check names"

        for name in listed:
            assert any(name.startswith(pref) for pref in prefixes if pref), (
                f"{module_name}._skip_rest stands in for {name!r}, but no p.check() "
                f"in that phase emits a name starting that way. The skip would "
                f"appear in the report while the real check silently vanished."
            )


class TestSyntheticPreviousRelease:
    """p9 upgrades from a repacked wheel; the repack must produce a valid one.

    pip validates the hashes in RECORD, so a repack that patched METADATA and
    left RECORD stale would fail the install for a reason that has nothing to
    do with upgrade hygiene — and the phase would report a lifecycle failure
    that was really a bug in its own fixture.
    """

    def _minimal_wheel(self, tmp_path):
        import base64
        import hashlib
        import zipfile

        src = tmp_path / "src"
        (src / "lasagna").mkdir(parents=True)
        (src / "lasagna" / "__init__.py").write_text("VERSION = '0.1.0'\n")
        di = src / "lasagna_rag-0.1.0.dist-info"
        di.mkdir()
        (di / "METADATA").write_text("Metadata-Version: 2.1\nName: lasagna-rag\nVersion: 0.1.0\n")
        (di / "WHEEL").write_text("Wheel-Version: 1.0\n")

        lines = []
        for f in sorted(src.rglob("*")):
            if f.is_file():
                data = f.read_bytes()
                h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                lines.append(f"{f.relative_to(src).as_posix()},sha256={h},{len(data)}")
        lines.append("lasagna_rag-0.1.0.dist-info/RECORD,,")
        (di / "RECORD").write_text("\n".join(lines) + "\n")

        wheel = tmp_path / "lasagna_rag-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as zf:
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(src).as_posix())
        return wheel

    def test_repack_renames_the_version_everywhere_it_appears(self, tmp_path):
        import zipfile

        from e2e.phases.p9_lifecycle import PROBE_MODULE, _repack

        out, why = _repack(self._minimal_wheel(tmp_path), "0.0.9", tmp_path / "out")
        assert out is not None, f"repack failed: {why}"
        assert out.name == "lasagna_rag-0.0.9-py3-none-any.whl"

        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            meta = zf.read("lasagna_rag-0.0.9.dist-info/METADATA").decode()

        assert "Version: 0.0.9" in meta and "Version: 0.1.0" not in meta
        assert not any("0.1.0.dist-info" in n for n in names), (
            "the old dist-info survived; pip would see two distributions"
        )
        assert f"lasagna/{PROBE_MODULE}.py" in names, (
            "the probe module is what makes the upgrade-hygiene check "
            "non-vacuous; it must be in the synthetic old wheel"
        )

    def test_repacked_record_hashes_match_the_repacked_contents(self, tmp_path):
        """A stale RECORD makes pip reject the install for the wrong reason."""
        import base64
        import hashlib
        import zipfile

        from e2e.phases.p9_lifecycle import _repack

        out, why = _repack(self._minimal_wheel(tmp_path), "0.0.9", tmp_path / "out")
        assert out is not None, why

        with zipfile.ZipFile(out) as zf:
            record = zf.read("lasagna_rag-0.0.9.dist-info/RECORD").decode()
            entries = 0
            for line in record.strip().splitlines():
                path, digest, size = line.rsplit(",", 2)
                if not digest:  # RECORD itself carries no hash of itself
                    continue
                data = zf.read(path)
                want = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
                assert digest == f"sha256={want.decode()}", f"stale hash for {path}"
                assert int(size) == len(data), f"stale size for {path}"
                entries += 1
            assert entries >= 3, "RECORD lists suspiciously few files"

    def test_repack_reports_why_it_failed_instead_of_raising(self, tmp_path):
        """The phase turns a repack failure into a skip, so it needs a reason."""
        from e2e.phases.p9_lifecycle import _repack

        junk = tmp_path / "not_a_wheel-1.0-py3-none-any.whl"
        junk.write_bytes(b"definitely not a zip")
        out, why = _repack(junk, "0.0.9", tmp_path / "out")
        assert out is None
        assert why, "a failed repack must explain itself, not skip with an empty reason"


class TestDocumentManagerWaitForProcessing:
    """wait_for_processing() constructed a Document with the wrong field names.

    It passed `upload_date=` (the *backend row's* name; the public dataclass
    calls it `created_at`) and omitted the required `session_id`, so it raised
    TypeError the moment a document reached a terminal status — meaning the
    method had never once returned successfully. Found by the E2E quality phase,
    which was the first thing to await it against a real ingest.

    Pinned here because the SDK documentation recommends this call as the way
    to get a real chunk count after an async upload.
    """

    def test_document_accepts_the_fields_wait_for_processing_passes(self):
        import dataclasses

        from lasagna.sdk.types import Document, DocumentStatus

        fields = {f.name for f in dataclasses.fields(Document)}
        assert "created_at" in fields
        assert "upload_date" not in fields, (
            "the public dataclass uses created_at; a caller passing upload_date "
            "is using the backend row's name and will TypeError"
        )

        required = {
            f.name
            for f in dataclasses.fields(Document)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }
        assert required == {"document_id", "filename", "session_id", "status"}, (
            f"Document's required fields changed to {required}; every construction "
            f"site in document_manager.py must be re-checked"
        )

        # The exact shape wait_for_processing builds must construct cleanly.
        Document(
            document_id="d1",
            filename="a.txt",
            session_id="s1",
            status=DocumentStatus.READY,
            chunk_count=3,
            file_size=123,
            created_at="2026-01-01T00:00:00Z",
        )

    def test_wait_for_processing_does_not_use_the_backend_field_name(self):
        """Guard the specific regression rather than only the dataclass shape."""
        from lasagna.sdk.managers import document_manager
        from tests.test_event_loop_safety import executable_source

        # Comments here necessarily mention the very construct being banned,
        # so check the executable source only.
        src = executable_source(document_manager.DocumentManager._wait_for_processing)
        assert "upload_date=" not in src, (
            "wait_for_processing must not pass upload_date= to Document(); "
            "the public field is created_at"
        )
        assert "session_id=" in src, "Document requires session_id; omitting it raises TypeError"


class TestTheGateDocumentDescribesTheGateThatExists:
    """`e2e/README.md` advertised `p5` Backend matrix, `p6` Interfaces and `p7`
    Stress, each with a "Blocks release on" column promising a guarantee —
    conformance on any backend, an advertised interface off-contract, tenant
    isolation and unbounded memory.

    None of the three existed. `run.py` registers six phases, `e2e/phases/`
    holds six files, and `--phase p5` is rejected as unknown. A gate document
    advertising checks nobody runs is worse than a shorter one: a reader takes
    the guarantee at face value, and `.github/workflows/tests.yml` repeated the
    claim in a comment.
    """

    def _registered(self) -> set:
        import re
        from pathlib import Path

        import e2e.run

        source = Path(e2e.run.__file__).read_text(encoding="utf-8")
        return set(re.findall(r'"(p\d)"', source))

    def test_every_phase_in_the_readme_table_exists(self):
        import re
        from pathlib import Path

        import e2e

        readme = (Path(e2e.__file__).parent / "README.md").read_text(encoding="utf-8")
        # Table rows only: `| `p5` | Backend matrix | ...`
        advertised = set(re.findall(r"^\|\s*`(p\d)`\s*\|", readme, re.M))
        registered = self._registered()

        assert advertised, "the gate README no longer documents its phases"
        assert advertised <= registered, (
            f"e2e/README.md advertises phases that do not exist: "
            f"{sorted(advertised - registered)}. Registered: {sorted(registered)}"
        )

    def test_every_registered_phase_has_a_module(self):
        """The other direction: a phase that dispatches to nothing."""
        from pathlib import Path

        import e2e

        phase_dir = Path(e2e.__file__).parent / "phases"
        modules = {p.name.split("_")[0] for p in phase_dir.glob("p*_*.py")}
        assert self._registered() <= modules, (
            f"run.py registers phases with no module: {sorted(self._registered() - modules)}"
        )
