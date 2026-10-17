"""The CLI says what is wrong with the setup, and what a result was computed under.

A bad environment setting (`RAG_CHUNK_SIZE=big`) raised ConfigError, whose
message names the setting and the fix, inside a 40-line traceback; `doctor`, the
command meant for that moment, crashed on it. `doctor` did not say which
session, profile or settings file was in force, and checked for the GitHub CLI.
`results bundle` on a file with no metrics version printed a traceback, and
`results validate` did not check the version at all. `status` said nothing
about a pre-0.1.0 collection no model reads. `init` cut its own title off and
promised "~2GB disk" whatever profile was chosen.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna import cli
from tests._paths import PACKAGE_DIR


def _python_m_lasagna(tmp_path, *args, **env):
    full_env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    full_env.update({"HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent), **env})
    return subprocess.run(
        [sys.executable, "-m", "lasagna", *args],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=tmp_path,
        timeout=300,
    )


# -- a bad setting ---------------------------------------------------------------


def test_a_bad_setting_is_one_message_and_exit_2(tmp_path):
    out = _python_m_lasagna(tmp_path, "status", RAG_CHUNK_SIZE="big")
    text = out.stdout + out.stderr
    assert out.returncode == 2, text
    assert "RAG_CHUNK_SIZE='big'" in text
    assert "Failed to get status" not in text and "Traceback" not in text


def test_main_turns_an_uncaught_config_error_into_the_message(monkeypatch, capsys):
    class ConfigError(ValueError):
        pass

    def app():
        try:
            raise ConfigError("RAG_X='y': x takes a number. Fix the value.")
        except ConfigError as exc:
            raise RuntimeError("wrapped") from exc

    monkeypatch.setattr(cli, "app", app)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert "RAG_X='y'" in capsys.readouterr().out


def test_other_errors_are_not_swallowed(monkeypatch):
    def app():
        raise KeyError("a bug")

    monkeypatch.setattr(cli, "app", app)
    with pytest.raises(KeyError):
        cli.main()


def test_both_ways_of_running_the_cli_go_through_main():
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib

    pyproject = PACKAGE_DIR.parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("no pyproject.toml beside the package (installed from a wheel)")
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    assert scripts["lasagna"] == "lasagna.cli:main"
    assert "main()" in (PACKAGE_DIR / "__main__.py").read_text()


# -- doctor ------------------------------------------------------------------------


def test_doctor_reports_a_bad_setting_as_a_failed_check(tmp_path):
    out = _python_m_lasagna(tmp_path, "doctor", "--json", RAG_CHUNK_SIZE="big")
    assert out.returncode == 1, out.stdout + out.stderr
    report = json.loads(out.stdout)
    (check,) = [c for c in report["checks"] if c["name"] == "config"]
    assert check["status"] == "error" and "RAG_CHUNK_SIZE='big'" in check["detail"]
    assert "Traceback" not in out.stderr


def test_doctor_says_which_setup_is_in_force(tmp_path):
    out = _python_m_lasagna(tmp_path, "doctor", "--json")
    report = json.loads(out.stdout)
    assert set(report["context"]) == {"session", "profile", "settings file", "data directory"}
    assert report["context"]["session"] == "default"
    assert "gh_cli" not in {c["name"] for c in report["checks"]}

    issue = _python_m_lasagna(tmp_path, "doctor", "--issue", "-s", "mine")
    assert "session     mine" in issue.stdout and "profile" in issue.stdout


# -- results bundle / validate ----------------------------------------------------------


def test_bundling_a_result_with_no_metrics_version_says_how_to_rerun(tmp_path):
    old = tmp_path / "results.json"
    old.write_text(json.dumps({"experiment_config": {"n_runs": 1}, "results": []}))
    result = CliRunner().invoke(cli.app, ["results", "bundle", str(old), "-o", str(tmp_path / "b")])
    said = " ".join(result.output.split())
    assert result.exit_code == 1, said
    assert "no metrics_version" in said and "lasagna benchmark experiment" in said
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_validate_refuses_a_bundle_from_other_metric_definitions(tmp_path):
    from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION, build_bundle

    rows = [
        {
            "benchmark_name": "smoke",
            "config_name": "c",
            "run_index": 0,
            "metrics": {"ndcg@10": 0.5},
            "split": "test",
            "resolved_mode": "live",
        }
    ]
    bundle = build_bundle(
        SimpleNamespace(_results=rows, experiment_config={"metrics_version": METRICS_VERSION})
    )
    bundle["metrics_version"] = "1.4"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(bundle))
    result = CliRunner().invoke(cli.app, ["results", "validate", str(path)])
    assert result.exit_code == 1
    assert "metrics_version:" in result.output and "1.4" in result.output


# -- status -------------------------------------------------------------------------


def test_status_names_a_collection_no_model_reads(monkeypatch):
    from lasagna.api import _collection_manager

    # status reads documents from the metadata database, and opens the vector
    # store only when a document could predate per-model collections.
    monkeypatch.setattr(cli, "_local_documents", lambda session_id: [])
    monkeypatch.setattr(cli, "_may_hold_legacy_documents", lambda: True)
    monkeypatch.setattr(
        "lasagna.api.rag_backend.get_rag_backend",
        lambda: SimpleNamespace(_vector_backend="the-store"),
    )
    seen = []
    monkeypatch.setattr(
        _collection_manager,
        "legacy_collections",
        lambda backend: seen.append(backend)
        or [{"name": "rag_documents_v2", "rows": 1234, "embedding_dim": 1024}],
    )
    result = CliRunner().invoke(cli.app, ["status"], terminal_width=200)
    said = " ".join(result.output.split())
    assert result.exit_code == 0, said
    assert seen == ["the-store"]
    assert "rag_documents_v2 holds 1,234 chunks at 1024 dimensions" in said
    # RAG_COLLECTION_NAME=rag_documents_v2 is the shipped default, which names
    # one collection per model, so it never reached the old collection.
    assert "set LASAGNA_LEGACY_COLLECTION_MODEL to that model's name" in said


# -- init -----------------------------------------------------------------------------


class _Answers:
    """questionary stand-in: each prompt returns the next scripted answer."""

    def __init__(self, selects, passwords):
        self.selects, self.passwords, self.prompts = list(selects), list(passwords), []

    def select(self, message, choices, default=None):
        self.prompts.append((message, [c.title for c in choices]))
        return SimpleNamespace(ask=lambda: self.selects.pop(0))

    def password(self, message):
        self.prompts.append((message, None))
        return SimpleNamespace(ask=lambda: self.passwords.pop(0))

    @staticmethod
    def Choice(title, value):  # noqa: N802 - questionary's own spelling
        return SimpleNamespace(title=title, value=value)


def test_init_states_each_profiles_download_and_writes_no_key(tmp_path, monkeypatch):
    # init used to ask for the key and save it to ./.env, which Lasagna then
    # never read; it now says which variable to export (see
    # test_init_writes_no_key_and_matches_status.py).
    answers = _Answers(["milvus", "openai", "speed"], [])
    monkeypatch.setitem(sys.modules, "questionary", answers)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "lasagna.core.session_config.get_knowledge_config_manager",
        lambda *a, **k: SimpleNamespace(set_active_profile=lambda *a, **k: None),
    )
    result = CliRunner().invoke(cli.app, ["init"], terminal_width=200)
    said = " ".join(result.output.split())
    assert result.exit_code == 0, said

    assert "Lasagna setup" in said and "Configure your knowledge base" in said
    profiles = dict(answers.prompts)["Quality profile?"]
    assert any("downloads 92 MB: all-MiniLM-L6-v2" in title for title in profiles)
    assert any("downloads 3.2 GB" in title for title in profiles)
    assert not any("2GB" in title for _, titles in answers.prompts for title in titles or [])

    assert "export OPENAI_API_KEY=<your key>" in said
    assert "OPENAI_API_KEY" not in (tmp_path / ".env").read_text()


# -- names ------------------------------------------------------------------------------


def test_the_command_is_called_lasagna():
    import typer

    assert typer.main.get_command(cli.app).name == "lasagna"


def test_profile_descriptions_say_what_runs():
    from lasagna.core.session_config import get_knowledge_config_manager

    described = {
        p["name"]: p.get("description", "") for p in get_knowledge_config_manager().list_profiles()
    }
    assert "all-MiniLM-L6-v2" in described["speed"]
    assert "CRAG" in described["max_quality"]
    assert not [n for n, d in described.items() if "SOTA" in d or "state-of-the-art" in d.lower()]
