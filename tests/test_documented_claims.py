"""Claims the package makes about itself must stay true.

Documentation drifts silently. This package has already shipped one false
claim — a README line advertising a shared backend contract suite that did not
exist — and stale references to a file that had been moved. Neither broke a
test, because nothing tested prose.

These tests read the documentation and check it against the code. They are
deliberately about *verifiable* claims: counts, names, and behaviour a reader
would reasonably act on. Prose that cannot be mechanically checked is out of
scope; the point is that no number or name in the docs should be able to go
stale unnoticed.
"""

import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; the eval extra installs tomli there
    import tomli as tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Docs a newcomer is pointed at, and which therefore must not rot. Relative
#: links in each resolve against that file's own directory, which is only
#: accidentally the repo root.
ENTRY_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "GOVERNANCE.md",
    "AGENTS.md",
    "hub/README.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestVersionIsSingleSourced:
    """The version was declared in three files with nothing keeping them in step.

    `pyproject.toml`, `sdk/_version.py` and `CITATION.cff` each held a literal.
    Worse, `core/provenance.py` read the `_version.py` literal, so every results
    bundle recorded the string in that file rather than the version actually
    installed — a wheel built at one version with the file edited to another
    would have misreported provenance silently.

    `_version.py` now reads `importlib.metadata`, so pyproject is the source and
    the two cannot disagree. `CITATION.cff` is still hand-maintained (it is data,
    not code), so it is checked here.
    """

    def test_the_package_reports_the_installed_version(self):
        import importlib.metadata as md

        from lasagna.sdk._version import __version__ as pkg_version

        try:
            assert pkg_version == md.version("lasagna-rag")
        except md.PackageNotFoundError:  # pragma: no cover - source checkout
            assert pkg_version == "0.0.0+unknown"

    def test_pyproject_and_package_agree(self):
        from lasagna.sdk._version import __version__ as pkg_version

        declared = tomllib.loads(_read(PYPROJECT))["project"]["version"]
        assert declared == pkg_version, (
            f"pyproject.toml says {declared}, lasagna.sdk._version says "
            f"{pkg_version}. A release built from one would report the other."
        )

    def test_the_top_level_package_exposes_it(self):
        """`lasagna.__version__` raised AttributeError while
        `lasagna.sdk.__version__` worked — so the obvious way to report which
        version produced a result was the one that did not exist."""
        import lasagna

        assert isinstance(lasagna.__version__, str) and lasagna.__version__

    def test_the_citation_file_agrees(self):
        """A citation pinning a version the package never shipped is worse than
        no version at all — it looks authoritative."""
        yaml = pytest.importorskip("yaml")

        from lasagna.sdk._version import __version__ as pkg_version

        citation = yaml.safe_load(_read(REPO_ROOT / "CITATION.cff"))
        assert str(citation["version"]) == pkg_version, (
            f"CITATION.cff says {citation['version']}, the package is {pkg_version}"
        )

    def test_the_citation_can_produce_an_indexable_entry(self):
        """Without `date-released` the generated BibTeX has no year, and
        Scholar and Semantic Scholar will not index it."""
        yaml = pytest.importorskip("yaml")

        citation = yaml.safe_load(_read(REPO_ROOT / "CITATION.cff"))
        for field in ("cff-version", "message", "title", "authors", "date-released"):
            assert citation.get(field), f"CITATION.cff is missing {field!r}"

    def test_version_info_matches_version_string(self):
        from lasagna.sdk._version import __version__, __version_info__

        numeric = []
        for chunk in __version__.split(".")[:3]:
            digits = ""
            for ch in chunk:
                if not ch.isdigit():
                    break
                digits += ch
            numeric.append(int(digits) if digits else 0)
        assert __version_info__ == tuple(numeric)


class TestCountsThatDriftedOnce:
    """Each of these was wrong in a shipped document because nothing checked it.

    A numeral in prose is a claim about the code, and this project's pitch is
    that it checks its own numbers. "Seven backends" sat four lines above a list
    of six, and 200 lines above a sentence saying six.
    """

    def test_the_backend_count_is_real(self):
        from lasagna import vector_backends

        exported = [n for n in vector_backends.__all__ if n.endswith("Backend")]
        spelled = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        claims = re.findall(r"(\w+) (?:vector )?backends", _read(README))
        numeric = [spelled[c.lower()] for c in claims if c.lower() in spelled]
        assert numeric, "README no longer states a backend count"
        for claimed in numeric:
            assert claimed == len(exported), (
                f"README claims {claimed} backends; {len(exported)} are exported: "
                f"{sorted(exported)}"
            )

    def test_the_cold_start_range_is_the_same_in_every_document(self):
        """`tests/test_pypi_readme.py` enforced the range against README only,
        so `docs/guide.md` drifted to a window that 6 of 10 measured runs
        falsify — while citing the report it contradicts."""
        pattern = re.compile(r"between (\d+) and (\d+) minutes")
        ranges = {}
        for rel in ("README.md", "docs/guide.md"):
            text = _read(REPO_ROOT / rel)
            found = pattern.findall(text) or re.findall(r"(\d+)[–-](\d+) minutes", text)
            if found:
                ranges[rel] = found[0]
        assert len(ranges) >= 2, f"expected the range in both documents, got {ranges}"
        assert len(set(ranges.values())) == 1, (
            f"the documents disagree about cold-start time: {ranges}"
        )

    def test_no_document_hard_codes_a_test_count(self):
        """The badge said 2038 while 2337 were collected. A number nothing
        derives is a number that is wrong the next time anyone adds a test."""
        for rel in ("README.md", "docs/guide.md"):
            text = _read(REPO_ROOT / rel)
            assert not re.search(r"tests-\d+%20passing", text), (
                f"{rel} hard-codes a test count in a badge; nothing updates it"
            )

    def test_the_watsonx_env_var_matches_what_reads_it(self):
        """The CLI wrote `WATSONX_API_KEY`; the gate and the infrastructure docs
        use IBM's own spelling `WATSONX_APIKEY`. The CLI was setting a variable
        nothing reads."""
        cli = _read(REPO_ROOT / "src" / "lasagna" / "cli.py")
        assert "WATSONX_APIKEY" in cli
        assert '"WATSONX_API_KEY"' not in cli


class TestReadmeCountsAreReal:
    """Numbers in the README are checkable; check them."""

    def test_benchmark_dataset_count(self):
        # Tolerant of phrasing ("19 benchmarks", "19 benchmark datasets") but
        # strict about the number. The claim is what must stay true, not the
        # sentence it lives in.
        claimed = re.search(r"(\d+)\s+benchmark(?:s|\s+datasets)", _read(README))
        assert claimed, "README no longer states a benchmark count"

        # The registry, not a grep of `datasets.py`.
        #
        # That grep counted `"hf_dataset":` literals in the legacy catalogue —
        # a static subset that the BEIR adapters extend at import time — so it
        # agreed with a README claiming 19 while 31 benchmarks were actually
        # runnable. A count test that measures the wrong collection certifies
        # the wrong number.
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        internal = {"smoke"}
        public = [
            n
            for n in benchmark_registry.names()
            if n not in internal and not n.startswith("private_eval")
        ]
        assert int(claimed.group(1)) == len(public), (
            f"README claims {claimed.group(1)} benchmarks; the registry exposes "
            f"{len(public)} public ones (plus {len(benchmark_registry.names()) - len(public)} "
            f"internal harness fixtures)"
        )

    def test_query_time_stage_count_is_real(self):
        """The README's central claim is a count, so derive it from the code."""
        from lasagna.core.pipeline import QUERY_TIME_STAGES

        words = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
        claimed = re.search(r"(\w+) at query time", _read(README))
        assert claimed, "README no longer states how many stages act at query time"
        assert words.get(claimed.group(1).lower()) == len(QUERY_TIME_STAGES)

    def test_any_uninjected_stage_is_disclosed(self):
        """Silence about a half-wired stage is indistinguishable from it working.

        Currently empty, so this asserts nothing today — deliberately kept, so
        that adding a stage without wiring it fails here rather than shipping
        as a claim.
        """
        from lasagna.core.pipeline import INGEST_TIME_STAGES_NOT_YET_INJECTED

        text = _read(README)
        for stage in INGEST_TIME_STAGES_NOT_YET_INJECTED:
            assert f"`{stage}`" in text, (
                f"'{stage}' is registrable but not injected, and the README does not say so"
            )

    def test_the_stage_lists_cover_every_component_type(self):
        """A stage in no list would be undocumented in every direction."""
        from lasagna.core.pipeline import (
            ANSWER_TIME_STAGES,
            INGEST_TIME_STAGES,
            INGEST_TIME_STAGES_NOT_YET_INJECTED,
            QUERY_TIME_STAGES,
        )
        from lasagna.core.registry import COMPONENT_TYPES

        lists = [
            QUERY_TIME_STAGES,
            INGEST_TIME_STAGES,
            ANSWER_TIME_STAGES,
            INGEST_TIME_STAGES_NOT_YET_INJECTED,
        ]
        assert set().union(*(set(x) for x in lists)) == set(COMPONENT_TYPES)
        assert sum(len(x) for x in lists) == len(COMPONENT_TYPES), "a stage is in two lists"

    def test_every_component_type_named_in_readme_exists(self):
        from lasagna.core.registry import COMPONENT_TYPES

        text = _read(README)
        for ctype in COMPONENT_TYPES:
            assert f"`{ctype}`" in text, (
                f"component type '{ctype}' is pluggable but never named in the "
                f"README, so users cannot discover it"
            )


class TestDocumentedCommandsExist:
    def test_readme_cli_commands_are_real(self):
        """Both plain commands and sub-app groups.

        This previously only knew about top-level `@app.command()` names, so
        every `lasagna <group> <cmd>` in the docs was invisible to it — the
        check passed by not looking. Sub-apps are where most of the interesting
        surface lives (`benchmark`, `results`, `component`, `plugins`).
        """
        from lasagna.cli import app

        actual = {c.name or c.callback.__name__ for c in app.registered_commands}
        actual |= {g.name or g.typer_instance.info.name for g in app.registered_groups}

        documented = set(re.findall(r"^lasagna ([a-z][a-z-]*)", _read(README), re.M))
        missing = documented - actual
        assert not missing, (
            f"README tells users to run commands that do not exist: {sorted(missing)}. "
            f"Available: {sorted(actual)}"
        )

    def test_documented_subcommands_are_real(self):
        """`lasagna results bundle` must name a real subcommand of `results`.

        Scans every entry document, not just the README: the hub guide is the
        only place `results leaderboard` and `results schema` are documented, so
        renaming either would otherwise rot silently.
        """
        from lasagna.cli import app

        groups = {
            (g.name or g.typer_instance.info.name): g.typer_instance for g in app.registered_groups
        }
        pattern = re.compile(rf"lasagna ({'|'.join(groups)}) ([a-z][a-z-]*)")

        missing = []
        for rel in ENTRY_DOCS:
            doc = REPO_ROOT / rel
            if not doc.exists():
                continue
            for group_name, sub in set(pattern.findall(_read(doc))):
                available = {
                    c.name or c.callback.__name__ for c in groups[group_name].registered_commands
                }
                if sub not in available:
                    missing.append(f"{rel}: lasagna {group_name} {sub} (has: {sorted(available)})")
        assert not missing, f"docs reference subcommands that do not exist: {missing}"

    def test_the_readme_factorial_block_actually_builds_four_cells(self):
        """The README shows a YAML snippet and claims it produces a 2x2.

        A snippet is a claim about the parser. This one is load-bearing: the
        section it sits in exists because the previous way of expressing a
        two-arm comparison — a `null` in a dimension list — produced two
        identical arms and said nothing about it.
        """
        import yaml

        from lasagna.knowledge_evaluation_sdk.experiment import (
            FactorialSpec,
            PipelineSweepSpec,
            _generate_pipeline_configs,
        )

        blocks = re.findall(r"```yaml\n(.*?)```", _read(README), re.S)
        factorial_blocks = [b for b in blocks if "factorial:" in b]
        assert factorial_blocks, "README no longer documents the factorial design"

        for block in factorial_blocks:
            spec = (yaml.safe_load(block) or {}).get("pipeline") or {}
            configs = _generate_pipeline_configs(
                PipelineSweepSpec(
                    base_profile=spec.get("base_profile", "standard"),
                    factorial=FactorialSpec(**(spec.get("factorial") or {})),
                )
            )
            assert len(configs) == 4
            arms = {tuple(sorted(c["overrides"].items())) for c in configs}
            assert len(arms) == 4, f"README example produces duplicate arms: {arms}"

    def test_console_scripts_resolve(self):
        """Every entry point in pyproject must import and expose its target."""
        import importlib

        scripts = tomllib.loads(_read(PYPROJECT))["project"]["scripts"]
        for name, target in scripts.items():
            module, _, attr = target.partition(":")
            mod = importlib.import_module(module)
            assert hasattr(mod, attr), (
                f"console script '{name}' points at {target}, but {module} has "
                f"no attribute '{attr}' — installing the package would create a "
                f"command that crashes on first use"
            )


class TestProfilesMatchTheirDocumentation:
    """`settings.lasagna.toml` describes what each profile does. It must be true."""

    @pytest.fixture
    def specs(self):
        from lasagna.core.session_config._profiles import ProfileMixin

        return ProfileMixin._PROFILE_SPECS

    def test_speed_disables_what_it_claims_to(self, specs):
        """The settings file states Speed is "Pure Vector Search".

        A user picking it for latency, and silently getting reranking, would
        be paying for something they explicitly opted out of.
        """
        speed = specs["speed"]
        for toggle in (
            "enable_reranking",
            "enable_hybrid_search",
            "enable_mmr",
            "enable_hyde",
            "use_agentic_rag",
        ):
            assert speed.get(toggle) is False, (
                f"the speed profile is documented as pure vector search, but "
                f"{toggle} is {speed.get(toggle)!r}"
            )

    def test_profiles_are_actually_distinct(self, specs):
        """Two profiles with identical config are a menu that lies."""
        by_config = {}
        for name, cfg in specs.items():
            key = tuple(sorted((k, str(v)) for k, v in (cfg or {}).items()))
            by_config.setdefault(key, []).append(name)
        dupes = [names for names in by_config.values() if len(names) > 1]
        assert not dupes, f"these profiles are byte-identical: {dupes}"

    def test_embedding_model_and_dimension_agree(self, specs):
        """A model paired with the wrong dimension corrupts an index.

        Vectors are written at the model's true width into a collection
        declared at another, which fails at ingest or silently mis-indexes.
        """
        known = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "all-MiniLM-L6-v2": 384,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-large-v2": 1024,
            "intfloat/multilingual-e5-large": 1024,
        }
        for name, cfg in specs.items():
            model, dim = cfg.get("embedding_model"), cfg.get("embedding_dim")
            if model is None or dim is None:
                continue
            expected = known.get(model)
            if expected is None:
                continue  # unlisted model; model-licenses test covers cataloguing
            assert dim == expected, (
                f"profile '{name}' pairs {model} with embedding_dim={dim}, "
                f"but that model produces {expected}-dimensional vectors"
            )


class TestReferencedPathsExist:
    """A doc that points at a moved file is worse than one that says nothing."""

    def test_markdown_links_to_repo_files_resolve(self):
        broken = []
        for rel in ENTRY_DOCS:
            doc = REPO_ROOT / rel
            if not doc.exists():
                continue
            for target in re.findall(r"\]\(([^)#][^)]*)\)", _read(doc)):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                if not (doc.parent / target.split("#")[0]).exists():
                    broken.append(f"{rel} -> {target}")
        assert not broken, f"documentation links to missing paths: {broken}"

    def test_every_entry_doc_exists(self):
        """Otherwise the loop above skips a missing file and passes."""
        missing = [d for d in ENTRY_DOCS if not (REPO_ROOT / d).exists()]
        assert not missing, f"entry documents missing: {missing}"


class TestAgentInstructionsAreTrue:
    """A coding agent acts on AGENTS.md without checking it.

    Wrong instructions are worse here than in prose a human skims: the agent
    will follow a path that does not exist, or edit a directory that moved,
    and report the failure as a problem with the task.
    """

    AGENTS = REPO_ROOT / "AGENTS.md"

    def test_it_exists_and_claude_md_points_at_it(self):
        assert self.AGENTS.is_file()
        claude = REPO_ROOT / "CLAUDE.md"
        assert claude.exists(), "CLAUDE.md should resolve to the same instructions"
        assert claude.read_text() == self.AGENTS.read_text()

    def test_every_directory_in_the_layout_table_exists(self):
        missing = [
            path
            for path in re.findall(r"^\| `([^`]+)` \|", _read(self.AGENTS), re.M)
            if not (REPO_ROOT / path).exists()
        ]
        assert not missing, f"AGENTS.md describes paths that do not exist: {missing}"

    def test_the_symbols_it_names_are_importable(self):
        """It tells an agent to use these by name."""
        from lasagna.retrieval.search.ranker import adapt_reranker_for_pipeline  # noqa: F401
        from lasagna.utils.model_utils import get_default_model_name  # noqa: F401
        from lasagna.vector_backends.base import VectorStoreAdapter

        assert hasattr(VectorStoreAdapter, "iter_all")

    def test_the_test_files_it_cites_exist(self):
        cited = re.findall(r"`(tests/[\w./]+\.py)`", _read(self.AGENTS))
        assert cited, "AGENTS.md should point at the tests that enforce doc accuracy"
        missing = [p for p in cited if not (REPO_ROOT / p).exists()]
        assert not missing, f"AGENTS.md cites missing test files: {missing}"

    def test_the_mcp_config_is_valid_and_names_a_real_script(self):
        """`.mcp.json` is what makes cloning the repo enough to get the tools."""
        import json

        config = json.loads((REPO_ROOT / ".mcp.json").read_text())
        scripts = tomllib.loads(_read(PYPROJECT))["project"]["scripts"]
        for name, server in config["mcpServers"].items():
            assert server["command"] in scripts, (
                f".mcp.json server '{name}' runs '{server['command']}', "
                f"which is not a console script this package installs"
            )


class TestBadgesAreNotDecorative:
    """A badge with a stale number is a claim nobody re-checked."""

    def test_python_versions_badge_matches_requires_python(self):
        requires = tomllib.loads(_read(PYPROJECT))["project"]["requires-python"]
        text = _read(README)
        claimed = re.findall(r"3\.(\d+)", re.search(r"python-([^-]+)-blue", text).group(1))
        assert claimed, "python badge no longer lists versions"
        lo, hi = int(min(claimed)), int(max(claimed))
        assert f">=3.{lo}" in requires, f"badge starts at 3.{lo}, requires-python is {requires}"
        assert f"<3.{hi + 1}" in requires, f"badge ends at 3.{hi}, requires-python is {requires}"

    def test_license_badge_matches_declared_license(self):
        declared = tomllib.loads(_read(PYPROJECT))["project"]["license"]
        assert "Apache_2.0" in _read(README), f"badge does not match license {declared!r}"
        assert declared == "Apache-2.0"

    def test_the_readme_states_no_coverage_number(self):
        """No coverage figure for anyone to keep up to date.

        A badge and a limitations bullet both printed a percentage, and a
        number a reader cannot see re-measured goes stale. The README now says
        only that coverage is measured and floored in CI; the floor itself
        lives in pyproject.toml (``fail_under``), where CI enforces it.
        """
        text = _read(README)
        assert not re.search(r"badge/coverage-\d+", text), "a coverage badge is back"
        assert not re.search(r"[Cc]overage is \d+%", text), "a coverage percentage is back"
        assert tomllib.loads(_read(PYPROJECT))["tool"]["coverage"]["report"]["fail_under"] > 0
