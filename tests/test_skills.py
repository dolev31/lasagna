"""Skills tell an agent what to do. A wrong skill is worse than no skill.

An agent that reads `lasagna doctor --issue` in a skill will run it. If that
command does not exist, the agent does not fall back to reasoning — it reports
the failure, or invents something. So every command a skill names is checked
against the real CLI here, the same way documented README commands are.

The frontmatter matters for a different reason: `description` is what an agent
matches against to decide whether the skill applies at all. A vague or missing
one means the skill is never loaded, and silently does nothing.
"""

from __future__ import annotations

import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; the eval extra installs tomli there
    import tomli as tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / "src" / "lasagna" / "skills"


def _skill_dirs():
    return sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").is_file())


def _frontmatter(path: Path) -> dict:
    text = (path / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, f"{path.name}: no YAML frontmatter"
    out = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _body(path: Path) -> str:
    return (path / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cli_surface():
    """Every command and `group subcommand` pair the CLI actually exposes."""
    from lasagna.cli import app

    top = {c.name or c.callback.__name__ for c in app.registered_commands}
    pairs = set()
    for g in app.registered_groups:
        gname = g.name or g.typer_instance.info.name
        top.add(gname)
        for c in g.typer_instance.registered_commands:
            pairs.add((gname, c.name or c.callback.__name__))
    return top, pairs


class TestSkillsExist:
    def test_at_least_the_four_core_skills_ship(self):
        names = {p.name for p in _skill_dirs()}
        assert {"lasagna-rag", "lasagna-support", "lasagna-component", "lasagna-eval"} <= names

    def test_they_are_declared_as_package_data(self):
        """Otherwise `lasagna skills install` finds nothing after a pip install."""
        data = tomllib.loads((REPO / "pyproject.toml").read_text())
        patterns = data["tool"]["setuptools"]["package-data"]["lasagna"]
        assert any("skills" in p for p in patterns), (
            "skills are not in package-data, so they would be absent from the wheel"
        )


class TestFrontmatterIsUsable:
    @pytest.mark.parametrize("path", _skill_dirs(), ids=lambda p: p.name)
    def test_has_name_and_description(self, path):
        fm = _frontmatter(path)
        assert fm.get("name"), f"{path.name}: missing name"
        assert fm.get("description"), f"{path.name}: missing description"

    @pytest.mark.parametrize("path", _skill_dirs(), ids=lambda p: p.name)
    def test_name_matches_directory(self, path):
        assert _frontmatter(path)["name"] == path.name

    @pytest.mark.parametrize("path", _skill_dirs(), ids=lambda p: p.name)
    def test_description_says_when_to_use_it(self, path):
        """A description that does not describe a trigger never matches."""
        description = _frontmatter(path)["description"]
        assert len(description) > 60, f"{path.name}: description too vague to match on"
        assert "use when" in description.lower(), (
            f"{path.name}: description should state when the skill applies"
        )


class TestEveryCommandNamedIsReal:
    """The load-bearing check. An agent will run what the skill tells it to."""

    @pytest.mark.parametrize("path", _skill_dirs(), ids=lambda p: p.name)
    def test_lasagna_commands_exist(self, path, cli_surface):
        top, pairs = cli_surface
        text = _body(path)

        missing = []
        for match in re.finditer(r"lasagna ([a-z][a-z-]*)(?: ([a-z][a-z-]*))?", text):
            group, sub = match.group(1), match.group(2)
            if group not in top:
                missing.append(f"lasagna {group}")
                continue
            # A second word is only a subcommand if the first names a group.
            if sub and any(g == group for g, _ in pairs) and (group, sub) not in pairs:
                missing.append(f"lasagna {group} {sub}")
        assert not missing, (
            f"{path.name} tells an agent to run commands that do not exist: {sorted(set(missing))}"
        )

    @pytest.mark.parametrize("path", _skill_dirs(), ids=lambda p: p.name)
    def test_console_scripts_named_are_real(self, path):
        """Only invocations, not mentions.

        Skill names and the distribution itself are all `lasagna-something`, so
        a bare text search flags every one of them. What matters is a token in
        a position where something will actually be executed: the start of a
        line in a shell block, or an MCP `"command"` value.
        """
        declared = set(tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"])
        blocks = re.findall(r"```[a-z]*\n(.*?)```", _body(path), re.S)

        invoked = set()
        for block in blocks:
            invoked |= set(re.findall(r"^\s*(lasagna-[a-z-]+)\b", block, re.M))
            invoked |= set(re.findall(r'"command":\s*"(lasagna-[a-z-]+)"', block))

        missing = invoked - declared
        assert not missing, f"{path.name} invokes {sorted(missing)}, which are not console scripts"


class TestReferencedPathsResolve:
    @pytest.mark.parametrize("path", _skill_dirs(), ids=lambda p: p.name)
    def test_repo_paths_mentioned_exist(self, path):
        """Skills point at the plugin template and the licence docs; both must be there."""
        broken = [
            ref
            for ref in re.findall(r"`((?:examples|docs|src)/[\w./\-]+)`", _body(path))
            if not (REPO / ref).exists()
        ]
        assert not broken, f"{path.name} references missing paths: {broken}"


class TestCrossReferencesResolve:
    def test_named_skills_exist(self):
        """`lasagna-support` telling an agent to use a skill that does not ship."""
        names = {p.name for p in _skill_dirs()}
        for path in _skill_dirs():
            for ref in re.findall(r"`(lasagna-(?:rag|support|component|eval))` skill", _body(path)):
                assert ref in names, f"{path.name} points at missing skill '{ref}'"


MCP_SERVER = REPO / "src" / "lasagna" / "mcp" / "server.py"
MCP_README = REPO / "src" / "lasagna" / "mcp" / "README.md"


def _registered_mcp_tools() -> set:
    tools = set(re.findall(r"@mcp\.tool\(\)\s*(?:async\s+)?def (\w+)", MCP_SERVER.read_text()))
    assert tools, "no MCP tools found — the parse is wrong, not the docs"
    return tools


class TestMcpToolCatalogueIsAccurate:
    """The tool list is a capability claim, and it had drifted twice.

    The MCP README advertised 29 tools against 31 registered, omitting lifecycle
    policy control entirely — so an agent handed the README as its catalogue (the
    normal case before the server is running) would tell a user the capability
    does not exist. The skill undercounted differently, at 11.
    """

    def test_every_tool_the_skill_names_is_registered(self):
        listed = set(
            re.findall(
                r"`(\w+)`",
                _body(SKILLS / "lasagna-rag").split("Tools the agent gets:")[1].split("\n\n## ")[0],
            )
        )
        missing = listed - _registered_mcp_tools()
        assert not missing, f"skill names MCP tools that do not exist: {sorted(missing)}"

    def test_the_skill_names_every_registered_tool(self):
        """An undercount hides capability; the agent never tries what it cannot see."""
        listed = set(
            re.findall(
                r"`(\w+)`",
                _body(SKILLS / "lasagna-rag").split("Tools the agent gets:")[1].split("\n\n## ")[0],
            )
        )
        missing = _registered_mcp_tools() - listed
        assert not missing, f"skill omits registered MCP tools: {sorted(missing)}"

    def test_the_mcp_readme_documents_every_tool(self):
        text = MCP_README.read_text()
        missing = sorted(t for t in _registered_mcp_tools() if f"`{t}`" not in text)
        assert not missing, f"MCP README omits tools: {missing}"

    def test_the_advertised_count_is_the_real_count(self):
        """A stated number is the one thing that cannot be checked by reading."""
        actual = len(_registered_mcp_tools())
        for path in (MCP_README, SKILLS / "lasagna-rag" / "SKILL.md"):
            claimed = re.search(r"(\d+) tools", path.read_text())
            assert claimed, f"{path.name} no longer states a tool count"
            assert int(claimed.group(1)) == actual, (
                f"{path.name} claims {claimed.group(1)} MCP tools; {actual} are registered"
            )


class TestThePluginTemplateEndsSomewhereReal:
    """Step 4 is the step the template exists for, and it ran the wrong command.

    It called `lasagna benchmark run beir/scifact`, a single-arm run that
    produces one number from the stock pipeline — which reads exactly like a
    score for the reader's component and is not one. A silent wrong answer is
    worse than an error, and this one sat under the heading "Does it help? (a
    real benchmark, against a control)".
    """

    TEMPLATE = REPO / "examples" / "lasagna-plugin-template"

    def test_the_sweep_exists_and_parses(self):
        import yaml

        raw = yaml.safe_load((self.TEMPLATE / "sweep.yaml").read_text())
        assert raw["pipeline"]["rerankers"], "no arms declared"

    def test_it_has_a_control_and_the_component(self):
        """One arm is not a comparison."""
        import yaml

        arms = yaml.safe_load((self.TEMPLATE / "sweep.yaml").read_text())["pipeline"]["rerankers"]
        assert len(arms) >= 2, f"a sweep needs a control and the component: {arms}"
        registered = re.findall(
            r'"reranker\.(\w+)"', (self.TEMPLATE / "pyproject.toml").read_text()
        )
        assert registered, "template registers no reranker"
        assert any(name in arms for name in registered), (
            f"the sweep never runs the component this template registers "
            f"({registered}); arms are {arms}"
        )

    def test_it_loads_as_an_experiment_config(self):
        """A YAML that parses but does not build a config is worse than none."""
        import yaml

        from lasagna.knowledge_evaluation_sdk.experiment import (
            BenchmarkSpec,
            ExperimentConfig,
            PipelineSweepSpec,
        )

        raw = yaml.safe_load((self.TEMPLATE / "sweep.yaml").read_text())
        config = ExperimentConfig(
            name=raw["name"],
            description=raw.get("description", ""),
            benchmarks=[BenchmarkSpec(**b) for b in raw["benchmarks"]],
            pipeline=PipelineSweepSpec(**raw["pipeline"]),
            n_runs=raw["n_runs"],
            variance_source=raw["variance_source"],
            mode=raw["mode"],
            tracks=raw["tracks"],
        )
        # Retrieval-only on `speed`: the arms are deterministic, the sweep
        # itself says repeated runs were identical, and the significance test
        # pairs by query. tests/test_documented_claims.py holds every example
        # sweep to that rule.
        assert config.tracks == ["retrieval"] and config.n_runs == 1

    def test_the_readme_runs_the_sweep_not_a_single_arm(self):
        text = (self.TEMPLATE / "README.md").read_text()
        assert "benchmark experiment sweep.yaml" in text
        assert "benchmark run beir/scifact" not in text, (
            "step 4 is back to a single-arm run, which cannot compare anything"
        )
