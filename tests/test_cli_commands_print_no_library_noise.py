"""A one-shot command prints its answer, not its dependencies' chatter.

The first `lasagna upload` printed four gRPC fork() notices, a "Loading
weights" bar per model, and a warning that APScheduler was missing so "GC will
not run automatically", none of which the user could act on. gRPC, tqdm and
tokenizers read environment defaults that the `lasagna` and `lasagna-mcp`
commands now set (never `import lasagna`: a library must not change its host's
environment), and the scheduler warning is kept for `serve` and MCP, the
processes that would run the collector.
"""

from __future__ import annotations

import os
import subprocess
import sys

from tests._paths import PACKAGE_DIR

QUIET = {
    "GRPC_VERBOSITY": "ERROR",
    "GLOG_minloglevel": "2",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_DELAY": "2",
}


def _python(code: str, tmp_path, **env) -> subprocess.CompletedProcess:
    full_env = {k: v for k, v in os.environ.items() if k not in QUIET}
    full_env.update(
        {"HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent), "HF_HUB_OFFLINE": "1"}
    )
    full_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=full_env, timeout=300
    )


_REPORT_ENV = "".join(f"print({k!r}, os.environ.get({k!r}))\n" for k in QUIET)


def test_the_cli_sets_the_quiet_defaults_before_those_libraries_load(tmp_path):
    code = (
        "import os, sys\n"
        "import lasagna.cli\n"
        "early = [m for m in ('grpc', 'tqdm', 'tokenizers', 'transformers') if m in sys.modules]\n"
        "assert not early, f'imported before the defaults are set: {early}'\n"
        "sys.argv = ['lasagna', 'skills', 'list']\n"
        "try:\n"
        "    lasagna.cli.app()\n"
        "except SystemExit:\n"
        "    pass\n" + _REPORT_ENV
    )
    out = _python(code, tmp_path, GRPC_VERBOSITY="DEBUG")
    assert out.returncode == 0, out.stderr
    assert "GRPC_VERBOSITY DEBUG" in out.stdout, "a value the user set must win"
    for key, value in QUIET.items():
        if key != "GRPC_VERBOSITY":
            assert f"{key} {value}" in out.stdout, out.stdout


def test_the_mcp_server_sets_them_too(tmp_path):
    code = (
        "import os, sys\n"
        "sys.argv = ['lasagna-mcp', '--help']\n"
        "from lasagna.mcp.server import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit:\n"
        "    pass\n" + _REPORT_ENV
    )
    out = _python(code, tmp_path)
    assert out.returncode == 0, out.stderr
    for key, value in QUIET.items():
        assert f"{key} {value}" in out.stdout, out.stdout


def test_importing_the_library_changes_no_environment(tmp_path):
    out = _python("import os, lasagna, lasagna.sdk\n" + _REPORT_ENV, tmp_path)
    assert out.returncode == 0, out.stderr
    for key in QUIET:
        assert f"{key} None" in out.stdout, out.stdout


_SCHEDULER_WARNS = (
    "import sys\n"
    "sys.modules['apscheduler'] = None  # as on an install without the lifecycle extra\n"
    "from lasagna.lifecycle.scheduler import LifecycleScheduler\n"
    "LifecycleScheduler(gc=None, policy=None)._start_apscheduler()\n"
)


def test_a_one_shot_command_does_not_warn_about_background_gc(tmp_path):
    code = (
        "import sys\n"
        "sys.argv = ['lasagna', 'skills', 'list']\n"
        "from lasagna.cli import app\n"
        "try:\n"
        "    app()\n"
        "except SystemExit:\n"
        "    pass\n" + _SCHEDULER_WARNS
    )
    out = _python(code, tmp_path)
    assert out.returncode == 0, out.stderr
    assert "APScheduler" not in out.stderr, out.stderr


def test_the_server_mentions_it_only_when_asked_for_info(tmp_path):
    """The scheduler is optional (the `lifecycle` extra): a warning on every
    `lasagna serve` start was noise. `-v` still says how GC runs without it."""
    code = (
        "from lasagna.cli import _configure_cli_logging\n"
        "from loguru import logger\n"
        "logger.enable('lasagna')\n"
        "_configure_cli_logging()\n" + _SCHEDULER_WARNS
    )
    quiet = _python(code, tmp_path)
    assert quiet.returncode == 0, quiet.stderr
    assert "scheduler" not in quiet.stderr.lower(), quiet.stderr
    told = _python(code, tmp_path, LASAGNA_LOG_LEVEL="INFO")
    assert "lasagna-rag[lifecycle]" in told.stderr, told.stderr
