"""Dataset terms have to be discoverable, and the list has to stay true.

`NOTICE` tells users their obligations depend on each dataset's own license and
points them at a file listing them. That file did not exist — the legal notice
referenced a document that was never written.

Writing it by hand would have started drifting the moment a benchmark was
added, so it is generated from the adapters and this test fails when the
checked-in copy no longer matches. Same pattern as `test_model_licenses.py`,
for the same reason: a licence list is only useful if it is current.
"""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATASETS_MD = REPO / "src/lasagna/knowledge_evaluation_sdk/research/DATASETS.md"


class TestNoticePointsSomewhereReal:
    def test_the_referenced_file_exists(self):
        notice = (REPO / "NOTICE").read_text()
        assert "DATASETS.md" in notice, "NOTICE should point users at the dataset terms"
        assert DATASETS_MD.is_file(), (
            f"NOTICE references a dataset list at {DATASETS_MD.relative_to(REPO)}, "
            f"which does not exist — a legal notice pointing at a missing document"
        )

    def test_notice_names_the_path_that_exists(self):
        notice = (REPO / "NOTICE").read_text()
        rel = str(DATASETS_MD.relative_to(REPO))
        assert rel in notice, f"NOTICE should name {rel}"


class TestTheListIsCurrent:
    def test_every_public_benchmark_appears(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        text = DATASETS_MD.read_text()
        public = [
            n
            for n in benchmark_registry.names()
            if benchmark_registry.is_builtin(n) and not n.startswith("private_eval/")
        ]
        missing = [n for n in public if f"`{n}`" not in text]
        assert not missing, (
            f"{len(missing)} benchmark(s) missing from DATASETS.md: {missing[:5]}. "
            f"Regenerate with `lasagna benchmark licenses -o {DATASETS_MD}`"
        )

    def test_no_stale_entries(self):
        """A dataset removed from the code must not linger in the licence list,
        or users are told about terms for something they cannot run."""
        import re

        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        listed = set(re.findall(r"^\| `([^`]+)` \|", DATASETS_MD.read_text(), re.M))
        known = set(benchmark_registry.names())
        assert not (listed - known), f"DATASETS.md lists unknown benchmarks: {listed - known}"

    def test_declares_that_nothing_is_redistributed(self):
        text = DATASETS_MD.read_text()
        assert "redistributes no benchmark data" in text

    def test_live_capability_is_marked(self):
        """Which benchmarks can measure retrieval is the first thing a
        researcher needs, and it is not obvious from the name."""
        assert "Live" in DATASETS_MD.read_text()


class TestLicenseCoverage:
    def test_public_benchmarks_declare_a_license(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        undeclared = sorted(
            name
            for name, info in benchmark_registry.list_benchmarks().items()
            if not name.startswith("private_eval/") and not info.license
        )
        assert not undeclared, (
            f"benchmarks without a declared license: {undeclared}. Anyone "
            f"publishing results from these has no way to know their obligations."
        )

    def test_generator_is_reachable_from_the_cli(self):
        from lasagna.cli import benchmark_app

        names = {c.name for c in benchmark_app.registered_commands}
        assert "licenses" in names, (
            "the list must be regenerable by users, not only by whoever wrote it"
        )
