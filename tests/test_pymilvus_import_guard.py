"""Importing Lasagna's Milvus code does not read anyone's ``.env`` file.

``pymilvus/settings.py`` calls ``load_dotenv()`` on import, and python-dotenv
looks for a ``.env`` from the importing file's directory (or the working
directory, in ``python -c`` and notebooks) up to ``/``. With the virtualenv
inside a project, that found the project's ``.env``: an unrelated
``OPENROUTER_API_KEY`` appeared in ``os.environ`` and provider auto-detection
switched to OpenRouter. Every pymilvus import now goes through
``vector_backends/_pymilvus.py``, which switches the lookup off and restores
the environment.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import PACKAGE_DIR

pytest.importorskip("pymilvus")

SRC = PACKAGE_DIR.parent
LEAK = "LASAGNA_TEST_LEAK"
FAKE_KEY = "sk-or-lasagna-test-leak"


def _clean_env(home: Path, **extra: str) -> dict:
    """The parent environment minus any key the .env could also supply."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith(("_API_KEY", "_APIKEY")) and k not in (LEAK, "PYTHON_DOTENV_DISABLED")
    }
    env.update({"HOME": str(home), "PYTHONPATH": str(SRC), "HF_HUB_OFFLINE": "1", **extra})
    return env


def _run(args: list, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def project(tmp_path) -> Path:
    """A directory whose parent holds a .env, as a checkout inside a project does."""
    (tmp_path / ".env").write_text(f"{LEAK}=1\nOPENROUTER_API_KEY={FAKE_KEY}\n")
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "home").mkdir()
    return work


_REPORT = (
    "import os, sys\n"
    "print('pymilvus imported:', 'pymilvus' in sys.modules)\n"
    f"print('leaked:', os.environ.get({LEAK!r}), os.environ.get('OPENROUTER_API_KEY'))\n"
)

ROUTES = {
    "milvus backend": "import lasagna.vector_backends.milvus_backend\n",
    "milvus schema": (
        "from lasagna.core.schema import create_rag_document_schema\n"
        "create_rag_document_schema(8)\n"
    ),
}
# retrieval/search/hybrid.py imports the same way; the AST test below pins it
# without paying for its torch import here.


def test_the_setup_reproduces_the_leak(project):
    """A bare ``import pymilvus`` here does read the .env; the tests below
    would prove nothing otherwise."""
    out = _run(["-c", "import pymilvus\n" + _REPORT], project, _clean_env(project.parent / "home"))
    if f"leaked: 1 {FAKE_KEY}" not in out.stdout:
        pytest.skip("this pymilvus no longer loads .env on import; the guard is not needed")


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_dotenv_above_the_working_directory_is_not_loaded(project, route):
    out = _run(["-c", ROUTES[route] + _REPORT], project, _clean_env(project.parent / "home"))
    assert "pymilvus imported: True" in out.stdout, out.stdout
    assert "leaked: None None" in out.stdout, out.stdout


def test_a_dotenv_above_the_installed_package_is_not_loaded(tmp_path):
    """The reported case: the .env sits above site-packages, not the cwd.

    python-dotenv starts from the directory of ``pymilvus/settings.py`` when
    run from a script, so pymilvus is placed under a directory with a .env
    beside it (a symlink keeps that path; dotenv does not resolve it).
    """
    import pymilvus

    site = tmp_path / "venv" / "site-packages"
    site.mkdir(parents=True)
    (site / "pymilvus").symlink_to(Path(pymilvus.__file__).parent, target_is_directory=True)
    (tmp_path / "venv" / ".env").write_text(f"{LEAK}=1\nOPENROUTER_API_KEY={FAKE_KEY}\n")
    where = (
        "import sys\n"
        "assert sys.modules['pymilvus'].__file__.startswith(sys.argv[1]), 'wrong pymilvus'\n"
    )
    bare, guarded = tmp_path / "bare.py", tmp_path / "guarded.py"
    bare.write_text("import pymilvus\n" + where + _REPORT)
    guarded.write_text(ROUTES["milvus backend"] + where + _REPORT)
    home = tmp_path / "home"
    home.mkdir()
    env = _clean_env(home, PYTHONPATH=os.pathsep.join([str(site), str(SRC)]))

    if f"leaked: 1 {FAKE_KEY}" not in _run([str(bare), str(site)], tmp_path, env).stdout:
        pytest.skip("this pymilvus no longer loads .env on import; the guard is not needed")
    out = _run([str(guarded), str(site)], tmp_path, env)
    assert "leaked: None None" in out.stdout, out.stdout


def test_the_environment_and_dotenv_are_restored(project):
    """What the host had set survives, and its own load_dotenv still works."""
    code = (
        "import os, dotenv, dotenv.main\n"
        "from lasagna.vector_backends._pymilvus import import_pymilvus\n"
        "import_pymilvus()\n"
        "print('kept:', os.environ.get('LASAGNA_TEST_KEEP'))\n"
        "print('restored:', dotenv.load_dotenv is dotenv.main.load_dotenv,"
        " dotenv.load_dotenv.__name__)\n"
        f"print('before own load:', os.environ.get({LEAK!r}))\n"
        "dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))\n"
        f"print('after own load:', os.environ.get({LEAK!r}))\n"
    )
    env = _clean_env(project.parent / "home", LASAGNA_TEST_KEEP="yes")
    out = _run(["-c", code], project, env).stdout
    assert "kept: yes" in out
    assert "restored: True load_dotenv" in out
    assert "before own load: None" in out
    assert "after own load: 1" in out, "the host application's own load_dotenv was broken"


def test_a_cli_command_that_opens_the_store_does_not_load_the_dotenv(project):
    """End to end: `lasagna status` starts Milvus Lite, so it imports pymilvus.

    It opens the store only when a document could predate per-model
    collections, so the run records one with no collection first.
    """
    code = (
        "import os, sys\n"
        "from lasagna.core.database import get_rag_database\n"
        "db = get_rag_database()\n"
        "db.save_document({'document_id': 'doc_old', 'session_id': 'default', "
        "'filename': 'old.md', 'file_size': 1, 'file_type': 'md', 'status': 'ready', "
        "'chunks_count': 1}, '/old.md', collection_name=None)\n"
        "db.rebuild_quota_counters()\n"
        "sys.argv = ['lasagna', 'status']\n"
        "from lasagna.cli import app\n"
        "try:\n"
        "    app()\n"
        "except SystemExit as exc:\n"
        "    assert not exc.code, exc.code\n"
        "finally:\n"
        "    print('pymilvus imported:', 'pymilvus' in sys.modules)\n"
        f"    print('leaked:', os.environ.get({LEAK!r}), os.environ.get('OPENROUTER_API_KEY'))\n"
    )
    home = project.parent / "home"
    env = _clean_env(home, RAG_DB_DIR=str(home / "dbs"), RAG_UPLOAD_DIR=str(home / "up"))
    out = _run(["-c", code], project, env).stdout
    assert "pymilvus imported: True" in out, out
    assert "leaked: None None" in out, out


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(tree: ast.AST):
    """Import nodes outside ``if TYPE_CHECKING:`` blocks."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def test_every_pymilvus_import_goes_through_the_guard():
    guard = PACKAGE_DIR / "vector_backends" / "_pymilvus.py"
    offenders = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path == guard:
            continue
        for node in _runtime_imports(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
            )
            if any(n.split(".")[0] in ("pymilvus", "milvus_lite") for n in names):
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{node.lineno}")
    assert not offenders, (
        "import pymilvus through lasagna.vector_backends._pymilvus (import_pymilvus or "
        "from_pymilvus); a bare import loads the nearest .env: " + ", ".join(offenders)
    )
