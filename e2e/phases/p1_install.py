"""Phase 1 — Install: the package as a stranger receives it.

What this phase claims to prove
-------------------------------
1. The project builds a wheel, with the same command the release script uses,
   and the report says whether the tree it was built from matched ``HEAD``.
2. That wheel *contains the package*. Every ``.py`` module under ``src/lasagna``
   appears in the wheel, and every file matching a declared
   ``[tool.setuptools.package-data]`` pattern ships with it. ``import lasagna``
   alone cannot see this: the top-level ``__init__`` imports fine from a wheel
   that dropped ``lasagna/retrieval/`` entirely, and both console scripts still
   answer ``--help``, so without this check a packaging misconfiguration —
   a subpackage missing its ``__init__.py``, an ``exclude`` glob, a renamed
   directory — ships to every user with the gate reporting PASSED.
3. That wheel installs into an empty virtual environment on the interpreter
   running this gate, resolving its dependencies from the index, and
   ``import lasagna`` then works *inside that environment* — verified by a
   subprocess, not by the gate's own interpreter, which already has everything
   on its path and would happily pass a broken package.
4. Both ``[project.scripts]`` console entry points resolve and import — wiring
   that exists only in a wheel, so no source-tree test can cover it, and which
   p2 explicitly defers to this phase.
5. The light optional extras declared in ``pyproject.toml`` install on top of
   that environment, actually bring in the distributions they promise, and do
   not break ``import lasagna``.

What this phase does NOT prove
------------------------------
* **It builds the working tree, not a clean checkout.** That is deliberate —
  it is the artefact you are about to release — but it means an uncommitted or
  untracked file under ``src/`` is compiled into the wheel under test and would
  not exist in a CI build from ``HEAD``. A WARN check reports when the tree is
  dirty so the report says which of the two was measured.
* **It does not test PyPI resolution.** ``lasagna-rag`` is not published yet,
  so there is nothing to resolve; this installs the locally built wheel by
  path. The day the package ships, ``pip install lasagna-rag`` becomes a
  different and additionally necessary check — name availability, index
  metadata, yanked releases and sdist fallback are all untested here.
* **It tests one interpreter: the one running the gate.** Other Pythons are
  probed and reported, never installed into — a matrix install pulls torch
  (~1.5 GB) per version and belongs in CI, not on a laptop. Those rows are
  *skipped*, which the report renders as unvalidated, not as passing.
* **The pip cache is warm.** Install wall-times are measured with whatever the
  developer's ``~/.cache/pip`` already holds, so they are a floor. A cold
  machine downloading torch over a hotel connection sees a much larger number.
  The metric carries that caveat in its note.
* **The heavy extras are not installed at all** — ``all``, ``local``,
  ``docling``, ``flag-embedding`` are multi-GB and minutes each. They are
  skipped with a reason. Whether they resolve is genuinely unknown to this
  gate.
* **Importing is not running.** ``import lasagna`` and ``--help`` prove the
  wheel's modules and dependencies are present and importable; they prove
  nothing about behaviour. Phases p2 and p4 own that.
* One platform, one architecture. The wheel is ``py3-none-any``, so there is no
  per-platform wheel to test, but the *dependencies* resolved here are macOS
  arm64 builds. A Linux resolution can differ.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..contract import PhaseResult, Severity, timed

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_ROOT = REPO_ROOT / "src"
SRC_PKG = SRC_ROOT / "lasagna"

# Check names are constants because `_skip_rest` has to name the very same
# assertions to record them as unrun. Duplicated string literals drift, and the
# drift is silent: the skip stops matching, the row vanishes from the report,
# and absence reads as "fine" — the exact failure Rule 2 exists to prevent.
C_TREE = "wheel is built from a committed tree"
C_BUILD = "wheel builds from the working tree"
C_SIZE = "wheel is under {:.0f} MB"
C_MODULES = "wheel ships every module in the source tree"
C_PKGDATA = "wheel ships every declared package-data file"
C_EXTRAS_META = "wheel declares every pyproject extra"
C_CORE_INSTALL = "core install succeeds in a clean venv"
C_IMPORT = "`import lasagna` works in the clean venv"
C_SCRIPT = "console script `{} --help` runs"
C_EXTRAS = "optional extras install"
C_PIPCHECK = "dependency set is consistent after all light extras"

# Printed by the import probe so the path can be recovered from output that
# also carries warnings on stderr. Taking `splitlines()[-1]` instead makes a
# CRITICAL check hostage to any DeprecationWarning the package happens to emit.
IMPORT_MARK = "LASAGNA_AT="
IMPORT_PROBE = f"import lasagna; print({IMPORT_MARK!r} + (lasagna.__file__ or '?'))"

# Dumps the venv's installed distribution names, PEP 503-normalised, so an
# extra can be checked for having actually brought something in.
DIST_MARK = "DISTS="
DIST_PROBE = (
    "import importlib.metadata as m, re; "
    f"print({DIST_MARK!r} + ' '.join(sorted("
    "  re.sub(r'[-_.]+', '-', (d.metadata['Name'] or '')).lower()"
    "  for d in m.distributions())))"
)

# Extras deliberately not installed: each is multi-GB and minutes long, and
# `all` is the union of them. Skipped, never silently passed.
HEAVY_EXTRAS: Dict[str, str] = {
    "all": "union of every extra, including the heavy ones below (multi-GB, >10 min)",
    "local": "transformers + accelerate + ollama bindings (multi-GB model stack)",
    "docling": "docling pulls its own layout/OCR model stack (multi-GB, minutes)",
    "flag-embedding": "FlagEmbedding pulls BGE-M3 weights and its own torch stack",
}

# Probed for reporting only; never installed into. See module docstring.
OTHER_PYTHON_SERIES = ("3.10", "3.11", "3.13")

# Timeouts. Generous, because the failure mode they exist to prevent is the
# gate hanging forever, not the gate being brisk.
BUILD_TIMEOUT_S = 900.0
VENV_TIMEOUT_S = 300.0
CORE_INSTALL_TIMEOUT_S = 3600.0
EXTRA_INSTALL_TIMEOUT_S = 1800.0
IMPORT_TIMEOUT_S = 180.0
PIP_CHECK_TIMEOUT_S = 300.0

# Budgets. All WARN: a loaded laptop or a slow link must not block a release.
# The wheel-size budget is the exception in spirit — it is not about speed, it
# is a tripwire for accidentally packaging test fixtures or a model file — but
# it stays WARN because a legitimate large asset should not block a release
# either, only get noticed.
WHEEL_SIZE_BUDGET_MB = 8.0
CORE_INSTALL_BUDGET_S = 900.0
FOOTPRINT_BUDGET_MB = 3000.0

# Paths used only inside `repro` strings, so a reader can paste the command
# without first reconstructing this phase's throwaway temp directory. Both
# directories are cleared first: without that, a second run leaves two wheels
# in the outdir, `{R_WHEELS}/*.whl` expands to both, and the repro stops
# reproducing anything.
R_WHEELS = "/tmp/lasagna-p1-wheels"
R_VENV = "/tmp/lasagna-p1-venv"
R_BUILD = (
    f"rm -rf {R_WHEELS} && cd {REPO_ROOT} && "
    f"uvx --from build pyproject-build --wheel --outdir {R_WHEELS} ."
)
R_MKVENV = f"{R_BUILD} && rm -rf {R_VENV} && {sys.executable} -m venv {R_VENV}"
R_INSTALL = f"{R_MKVENV} && {R_VENV}/bin/python -m pip install {R_WHEELS}/*.whl"


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

# Set once, at the top of `_body`, from the phase's own temp directory.
# Module-level rather than threaded through every call because every child in
# this phase wants the same answer, and a call site that forgets is a leak.
_SANDBOX_ENV: Dict[str, str] = {}
_CHILD_CWD: Optional[Path] = None


def _child_env() -> Dict[str, str]:
    """Environment for build/pip children.

    Inherits the real environment — pip needs PATH, proxy and CA settings to
    reach an index at all, and "no network" is not the defect this phase looks
    for. What is stripped is anything that would make the child install
    somewhere other than the throwaway venv, or import from the developer's
    source tree instead of from site-packages.

    ``HOME`` is deliberately *kept*: the pip cache lives under it, and blanking
    it would turn a 90-second install into a torch download. The store this
    package actually writes to is redirected by its own documented env vars
    instead, so `lasagna --help` cannot create or touch the developer's real
    ``~/.rag_knowledge``.
    """
    env = dict(os.environ)
    for leak in (
        "VIRTUAL_ENV",  # would make `pip` target the gate's own venv
        "PYTHONPATH",  # would let `import lasagna` succeed from source
        "PYTHONHOME",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_REQUIRE_VIRTUALENV",
        "CONDA_PREFIX",
    ):
        env.pop(leak, None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(_SANDBOX_ENV)
    return env


def _run(
    args: List[str],
    timeout: float,
    cwd: Optional[Path] = None,
) -> Tuple[Optional[int], str, float]:
    """Run a child process. Returns (returncode or None on timeout, output, elapsed).

    Never raises: a missing executable and a non-zero exit both come back as a
    return code the caller turns into a failed check.

    Defaults to the phase's temp directory rather than inheriting the gate's
    cwd. For ``python -c`` that matters: ``sys.path[0]`` is the working
    directory, so a child launched from the repo root can import the source
    tree it is supposed to be testing the *installed* copy of. It also keeps
    the repo's gitignored ``.env`` out of the console scripts under test.
    """
    where = cwd or _CHILD_CWD
    with timed() as t:
        try:
            cp = subprocess.run(
                args,
                cwd=str(where) if where else None,
                env=_child_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(out, bytes):  # pragma: no cover - platform dependent
                out = out.decode("utf-8", "replace")
            return None, f"TIMEOUT after {timeout:.0f}s\n{_tail(out)}", t.elapsed
        except OSError as exc:
            return 127, f"{type(exc).__name__}: {exc}", t.elapsed
    return cp.returncode, (cp.stdout or "") + (cp.stderr or ""), t.elapsed


def _tail(text: str, limit: int = 1200) -> str:
    """Last `limit` characters — enough to see the actual error, not a log dump."""
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _dir_mb(path: Path) -> float:
    """Recursive apparent size in MB. 0.0 for a missing directory."""
    if not path.exists():
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return round(total / 1e6, 1)


def _venv_python(venv: Path) -> Path:
    return (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )


def _site_packages(venv: Path) -> Path:
    hits = sorted(venv.glob("lib/**/site-packages")) or sorted(venv.glob("Lib/site-packages"))
    return hits[0] if hits else venv


def _normalise_extra(name: str) -> str:
    """PEP 685 extra normalisation, so pyproject and wheel metadata compare."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------------------
# pyproject / wheel metadata
# ---------------------------------------------------------------------------


def _load_pyproject() -> Tuple[Optional[dict], str]:
    """Parsed pyproject.toml, or (None, reason)."""
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None, (
                "no TOML reader on this interpreter (needs Python 3.11+ or `tomli`), "
                "so pyproject.toml could not be read"
            )
    try:
        return tomllib.loads(PYPROJECT.read_text(encoding="utf-8")), ""
    except (OSError, ValueError) as exc:
        return None, f"could not parse {PYPROJECT}: {type(exc).__name__}: {exc}"


def _read_extras() -> Tuple[Optional[Dict[str, List[str]]], str]:
    """`{extra: [requirement, ...]}` from [project.optional-dependencies].

    Read from pyproject rather than hard-coded so an extra added tomorrow is
    tested tomorrow, instead of quietly never being tested. The requirement
    lists come back too, so "the extra installed" can mean something stronger
    than "pip exited 0".
    """
    data, reason = _load_pyproject()
    if data is None:
        return None, reason
    extras = data.get("project", {}).get("optional-dependencies", {})
    if not isinstance(extras, dict) or not extras:
        return None, "pyproject.toml declares no [project.optional-dependencies]"
    return {str(k): list(v or []) for k, v in extras.items()}, ""


def _package_data_files() -> Tuple[Optional[List[str]], str]:
    """Every file the [tool.setuptools.package-data] globs match, as wheel arcnames.

    Expanded from the source tree rather than guessed, so a pattern added
    tomorrow is enforced tomorrow. Patterns that match nothing (a bundle CI
    builds and a laptop does not) contribute nothing and cannot fail.
    """
    data, reason = _load_pyproject()
    if data is None:
        return None, reason
    table = data.get("tool", {}).get("setuptools", {}).get("package-data", {})
    if not isinstance(table, dict) or not table:
        return None, "pyproject.toml declares no [tool.setuptools.package-data]"
    found: List[str] = []
    for pkg, patterns in table.items():
        base = SRC_ROOT if pkg == "*" else SRC_ROOT / str(pkg).replace(".", "/")
        for pattern in patterns or []:
            try:
                matches = base.glob(str(pattern))
            except (ValueError, OSError):  # a malformed pattern is not a wheel defect
                continue
            for f in matches:
                if f.is_file() and "__pycache__" not in f.parts:
                    found.append(f.relative_to(SRC_ROOT).as_posix())
    return sorted(set(found)), ""


def _source_modules() -> List[str]:
    """Every `.py` under src/lasagna, as the arcname the wheel should carry."""
    if not SRC_PKG.is_dir():
        return []
    return sorted(
        p.relative_to(SRC_ROOT).as_posix()
        for p in SRC_PKG.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _wheel_names(wheel: Path) -> Optional[List[str]]:
    """Every path inside the wheel. None if it is not a readable zip."""
    try:
        with zipfile.ZipFile(wheel) as zf:
            return zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return None


def _wheel_extras(wheel: Path) -> Optional[List[str]]:
    """`Provides-Extra` names declared inside the built wheel. None if unreadable."""
    try:
        with zipfile.ZipFile(wheel) as zf:
            meta = next((n for n in zf.namelist() if n.endswith(".dist-info/METADATA")), None)
            if meta is None:
                return None
            text = zf.read(meta).decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    return sorted(
        _normalise_extra(line.split(":", 1)[1].strip())
        for line in text.splitlines()
        if line.lower().startswith("provides-extra:")
    )


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _required_dists(requirements: List[str]) -> List[str]:
    """Distribution names an extra must bring in, normalised for comparison.

    Requirements carrying an environment marker are dropped: they are allowed
    to be absent on this interpreter, and asserting them would fail honestly
    installed extras. So is the self-referential `lasagna-rag[...]` of `all`.
    """
    out = []
    for req in requirements:
        text = str(req)
        if ";" in text:
            continue
        m = _REQ_NAME.match(text)
        if not m:
            continue
        name = _normalise_extra(m.group(1))
        if name == "lasagna-rag":
            continue
        out.append(name)
    return sorted(set(out))


def _marked(out: str, mark: str) -> str:
    """Value printed by a probe, found by its marker rather than by position.

    Child output here is stdout and stderr concatenated, so "the last line" is
    whatever warning the package emitted last, not the answer.
    """
    for line in reversed((out or "").splitlines()):
        if line.startswith(mark):
            return line[len(mark) :].strip()
    return ""


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


def run() -> PhaseResult:
    global _SANDBOX_ENV, _CHILD_CWD
    p = PhaseResult(key="p1", title="Install")
    workdir: Optional[Path] = None
    try:
        workdir = Path(tempfile.mkdtemp(prefix="lasagna-e2e-p1-"))
        _body(p, workdir)
    except Exception as exc:  # never raise out of run()
        p.check(
            "phase p1 ran to completion",
            False,
            detail=f"unhandled {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
            repro="python -m e2e.run --phase p1",
        )
        # Rule 2 applies to a crash as much as to a controlled stop. Without
        # this the remaining assertions are simply absent from the report, and
        # absence reads as "fine" to anyone skimming it.
        _skip_rest(p, f"the phase crashed ({type(exc).__name__}) before this ran")
    finally:
        # Venvs here are 1.5-2 GB. Leaving them behind fills a laptop in a
        # handful of runs.
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
        # The sandbox pointed into a directory that no longer exists; a later
        # child must not inherit a dangling store path.
        _SANDBOX_ENV = {}
        _CHILD_CWD = None
    return p


def _body(p: PhaseResult, workdir: Path) -> None:
    global _SANDBOX_ENV, _CHILD_CWD

    wheels = workdir / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    venv = workdir / "venv"

    # Children run from here, not from the gate's cwd: `python -c` puts the
    # working directory on `sys.path[0]`, and the repo root is the one place
    # that must never be importable while testing an installed copy.
    _CHILD_CWD = workdir

    # The package writes to ~/.rag_knowledge/{dbs,uploads} by default, and
    # SimpleConfigProvider mkdirs both the moment providers auto-configure. A
    # release gate must not create or write anything in the developer's real
    # store, so point its documented env vars at the throwaway directory that
    # gets deleted in `run()`'s finally.
    store = workdir / "store"
    (store / "dbs").mkdir(parents=True, exist_ok=True)
    (store / "uploads").mkdir(parents=True, exist_ok=True)
    _SANDBOX_ENV = {
        "RAG_DB_DIR": str(store / "dbs"),
        "RAG_UPLOAD_DIR": str(store / "uploads"),
        "RAG_SESSION_BACKEND": "memory",
    }

    # -- 0. build frontend ---------------------------------------------------
    # `uvx --from build` is what the release script uses. A local `python -m
    # build` is an acceptable stand-in; neither present means the phase cannot
    # run at all, which is a skip, not a failure.
    if shutil.which("uvx"):
        build_cmd = [
            "uvx",
            "--from",
            "build",
            "pyproject-build",
            "--wheel",
            "--outdir",
            str(wheels),
            ".",
        ]
    elif _run([sys.executable, "-c", "import build"], 60)[0] == 0:
        build_cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheels), "."]
    else:
        p.skipped_reason = (
            "no build frontend available: `uvx` is not on PATH and the `build` "
            "module is not importable by this interpreter"
        )
        # Rule 2 again. Without these rows the report shows p1 with no
        # assertions at all, which tells a reader nothing about what went
        # unverified. The phase still reports SKIPPED, so the run's verdict is
        # unchanged — but `unvalidated` now itemises what was skipped.
        _skip_rest(p, p.skipped_reason)
        return

    # -- 1. what is about to be built ----------------------------------------
    # The wheel comes from the working tree, which is the artefact under
    # review — but if that tree is dirty, the wheel validated here is not the
    # wheel CI builds from HEAD, and an untracked module compiled into it would
    # be missing for every user. WARN: releasing from a dirty tree is a choice,
    # making it silently is not.
    rc, out, _ = _run(
        ["git", "status", "--porcelain", "--", "src", "pyproject.toml"], 60, REPO_ROOT
    )
    if rc != 0:
        p.skip(
            C_TREE,
            f"`git status` unavailable here (exit={rc}), so the tree's state is unknown",
            severity=Severity.WARN,
        )
    else:
        dirty = [ln for ln in out.splitlines() if ln.strip()]
        untracked = [ln for ln in dirty if ln.startswith("??")]
        p.check(
            C_TREE,
            not dirty,
            severity=Severity.WARN,
            detail=(
                f"{len(dirty)} uncommitted path(s) under src/ or pyproject.toml "
                f"({len(untracked)} untracked) — this wheel is not the wheel HEAD "
                f"builds, and untracked files in it exist for nobody else:\n"
                + "\n".join(dirty[:10])
                + ("\n..." if len(dirty) > 10 else "")
            )
            if dirty
            else "working tree matches HEAD for src/ and pyproject.toml",
            repro=f"cd {REPO_ROOT} && git status --porcelain -- src pyproject.toml",
        )

    # -- 2. build the wheel --------------------------------------------------
    rc, out, build_s = _run(build_cmd, BUILD_TIMEOUT_S, cwd=REPO_ROOT)
    p.measure("wheel_build_s", round(build_s, 1), "s", note=" ".join(build_cmd[:3]))
    built = sorted(wheels.glob("*.whl"))
    if not p.check(
        C_BUILD,
        rc == 0 and bool(built),
        detail=f"exit={rc}\n{_tail(out)}"
        if rc != 0
        else ("build succeeded but produced no .whl" if not built else f"built {built[0].name}"),
        repro=R_BUILD,
        duration_s=build_s,
    ):
        # Everything downstream needs the wheel. Report it as unrun, not failed.
        _skip_rest(p, "no wheel was produced, so nothing could be installed")
        return

    wheel = built[0]
    wheel_mb = round(wheel.stat().st_size / 1e6, 2)
    p.measure(
        "wheel_size_mb",
        wheel_mb,
        "MB",
        note="pure-python wheel; a jump usually means packaged fixtures or data",
    )
    p.check(
        C_SIZE.format(WHEEL_SIZE_BUDGET_MB),
        wheel_mb <= WHEEL_SIZE_BUDGET_MB,
        severity=Severity.WARN,
        detail=f"{wheel.name} is {wheel_mb} MB",
        repro=f"{R_BUILD} && ls -l {R_WHEELS}/*.whl",
    )

    # -- 3. does the wheel actually contain the package? ---------------------
    # `import lasagna` cannot answer this. The top-level __init__ imports
    # cleanly from a wheel that dropped every subpackage, and both console
    # scripts still answer --help, because the CLI imports lasagna.sdk lazily
    # inside its commands. So the classic src-layout packaging failure — a
    # subpackage without __init__.py, silently excluded by
    # setuptools.packages.find — ships to users with this phase green. Compare
    # the source tree to the wheel's own manifest instead.
    names = _wheel_names(wheel)
    src_modules = _source_modules()
    if names is None:
        p.skip(C_MODULES, "the built wheel is not a readable zip archive")
    elif not src_modules:
        # A completeness check with nothing to compare against always passes.
        # That is the failure mode this phase exists to remove, so say so.
        p.skip(C_MODULES, f"no Python modules found under {SRC_PKG} — has the layout moved?")
    else:
        in_wheel = set(names)
        missing_mods = [m for m in src_modules if m not in in_wheel]
        p.check(
            C_MODULES,
            not missing_mods,
            detail=(
                f"{len(missing_mods)} of {len(src_modules)} modules are in the source "
                f"tree but not in the wheel — every user gets ModuleNotFoundError for "
                f"them:\n" + "\n".join(missing_mods[:15])
            )
            if missing_mods
            else f"all {len(src_modules)} source modules present",
            repro=(
                f"{R_BUILD} && diff <(cd {SRC_ROOT} && find lasagna -name '*.py' | sort) "
                f"<(unzip -Z1 {R_WHEELS}/*.whl | grep '\\.py$' | sort)"
            ),
        )

    # Non-Python payload is declared explicitly, so it can be checked exactly:
    # every file the package-data globs match must be in the wheel. WARN, not
    # CRITICAL — a missing settings default or deploy manifest degrades the
    # package rather than making it unimportable.
    pkg_data, pkgdata_reason = _package_data_files()
    if names is None:
        p.skip(C_PKGDATA, "the built wheel is not a readable zip archive", severity=Severity.WARN)
    elif pkg_data is None:
        p.skip(C_PKGDATA, pkgdata_reason, severity=Severity.WARN)
    elif not pkg_data:
        p.skip(
            C_PKGDATA,
            "the declared package-data patterns match no files in this checkout",
            severity=Severity.WARN,
        )
    else:
        in_wheel = set(names)
        missing_data = [f for f in pkg_data if f not in in_wheel]
        p.check(
            C_PKGDATA,
            not missing_data,
            severity=Severity.WARN,
            detail=(
                f"{len(missing_data)} of {len(pkg_data)} declared data files did not "
                f"ship (setuptools' package-data globs skip dot-files, among other "
                f"reasons):\n" + "\n".join(missing_data[:15])
            )
            if missing_data
            else f"all {len(pkg_data)} declared data files present",
            repro=(
                f"{R_BUILD} && unzip -Z1 {R_WHEELS}/*.whl | grep -F "
                f"'{missing_data[0] if missing_data else 'lasagna/'}' || echo 'NOT IN WHEEL'"
            ),
        )

    # Extras a user can ask for are the ones in the *wheel*, not the ones in
    # pyproject. If those two disagree, `pip install lasagna-rag[eval]` installs
    # nothing and only warns. WARN rather than CRITICAL: it degrades an optional
    # feature, it does not break the core install.
    declared, extras_reason = _read_extras()
    meta_extras = _wheel_extras(wheel)
    if declared is None:
        p.skip(C_EXTRAS_META, extras_reason, severity=Severity.WARN)
    elif meta_extras is None:
        p.skip(
            C_EXTRAS_META,
            "could not read METADATA out of the built wheel",
            severity=Severity.WARN,
        )
    else:
        missing = sorted({_normalise_extra(e) for e in declared} - set(meta_extras))
        p.check(
            C_EXTRAS_META,
            not missing,
            severity=Severity.WARN,
            detail=f"missing from wheel METADATA: {missing}"
            if missing
            else f"{len(meta_extras)} extras declared",
            repro=f"{R_BUILD} && unzip -p {R_WHEELS}/*.whl '*/METADATA' | grep Provides-Extra",
        )

    # -- 4. other interpreters: probed, never installed into -----------------
    # Recorded here, before anything that can bail out, so the report always
    # carries the row saying these were not validated.
    for series in OTHER_PYTHON_SERIES:
        found = shutil.which(f"python{series}")
        p.skip(
            f"install on python{series}",
            f"python{series} is not installed on this machine"
            if not found
            else f"found at {found}, but this gate installs on the running interpreter "
            f"({sys.version.split()[0]}) only — a matrix install pulls torch per "
            f"version; run the matrix in CI",
            severity=Severity.WARN,
        )

    # -- 5. throwaway venv on the current interpreter ------------------------
    rc, out, venv_s = _run([sys.executable, "-m", "venv", str(venv)], VENV_TIMEOUT_S)
    p.measure("venv_create_s", round(venv_s, 1), "s")
    vpy = _venv_python(venv)
    if not p.check(
        f"clean venv created on the running interpreter (python {sys.version.split()[0]})",
        rc == 0 and vpy.exists(),
        detail=f"exit={rc}\n{_tail(out)}",
        repro=f"{sys.executable} -m venv {R_VENV}",
        duration_s=venv_s,
    ):
        _skip_rest(p, "no virtual environment, so nothing could be installed into one")
        return

    # -- 6. core install -----------------------------------------------------
    # Rule 1: measure first, assert after — a four-minute install that then
    # fails its import check should still report the four minutes.
    rc, out, install_s = _run(
        [str(vpy), "-m", "pip", "install", str(wheel)], CORE_INSTALL_TIMEOUT_S
    )
    p.measure(
        "core_install_s",
        round(install_s, 1),
        "s",
        note="local wheel, warm pip cache; a cold cache downloads torch and is far slower",
    )
    core_ok = p.check(
        C_CORE_INSTALL,
        rc == 0,
        detail=f"exit={rc}\n{_tail(out)}",
        repro=R_INSTALL,
        duration_s=install_s,
    )
    p.check(
        f"core install completes within {CORE_INSTALL_BUDGET_S:.0f}s",
        install_s <= CORE_INSTALL_BUDGET_S,
        severity=Severity.WARN,
        detail=f"took {install_s:.0f}s (warm pip cache)",
        repro=R_INSTALL,
    )

    footprint_mb = _dir_mb(_site_packages(venv))
    p.measure(
        "installed_footprint_mb",
        footprint_mb,
        "MB",
        note="site-packages after the core install, before any extra",
    )
    p.check(
        f"installed footprint under {FOOTPRINT_BUDGET_MB:.0f} MB",
        0 < footprint_mb <= FOOTPRINT_BUDGET_MB,
        severity=Severity.WARN,
        detail=f"{footprint_mb} MB of site-packages (torch and CUDA-free wheels dominate)",
        repro=f"{R_INSTALL} && du -sh {R_VENV}/lib/*/site-packages",
    )

    if not core_ok:
        _skip_rest(p, "the core install failed, so the resulting venv proves nothing")
        return

    # -- 7. import inside that venv -----------------------------------------
    # Run as a subprocess of the *venv's* interpreter. Importing in this
    # process would pass on the gate's own site-packages and prove nothing.
    rc, out, import_s = _run([str(vpy), "-c", IMPORT_PROBE], IMPORT_TIMEOUT_S)
    p.measure(
        "installed_import_s",
        round(import_s, 2),
        "s",
        note="cold-bytecode import of the installed package; p2 owns the import budget",
    )
    # Guard against a false pass from an inherited path: the module must live
    # inside the throwaway venv, not in the repo or the gate's own venv. The
    # path is recovered by marker, not by "last line of output", so a warning
    # on stderr cannot turn a healthy package into a CRITICAL failure.
    imported_from = _marked(out, IMPORT_MARK)
    from_venv = bool(imported_from) and imported_from.startswith(str(venv))
    p.check(
        C_IMPORT,
        rc == 0 and from_venv,
        detail=(
            f"exit={rc}\n{_tail(out)}"
            if rc != 0
            else f"imported from {imported_from or '<probe printed nothing>'}"
            + ("" if from_venv else " — NOT inside the test venv")
        ),
        repro=f"{R_INSTALL} && {R_VENV}/bin/python -c 'import lasagna; print(lasagna.__file__)'",
        duration_s=import_s,
    )

    # Console scripts are wheel-only wiring: nothing in the source tree
    # exercises `[project.scripts]`, and p2 explicitly skips it, naming p1 as
    # the phase that catches it. Executing --help proves the entry point
    # resolves and its target module imports on a core install; it proves
    # nothing about what the commands actually do.
    #
    # CRITICAL, not WARN, and deliberately so: an advertised command that
    # cannot import is a correctness defect, identical for every user, and the
    # "a busy laptop must not block a release" argument for WARN does not apply
    # to a deterministic failure. If a project decides a console script may
    # depend on an extra, this is the one word to change.
    for script in ("lasagna", "lasagna-mcp"):
        exe = venv / ("Scripts" if os.name == "nt" else "bin") / script
        if not exe.exists():
            p.check(
                C_SCRIPT.format(script),
                False,
                detail=f"{exe} does not exist — [project.scripts] entry point not installed",
                repro=f"{R_INSTALL} && ls {R_VENV}/bin/",
            )
            continue
        rc, out, s = _run([str(exe), "--help"], IMPORT_TIMEOUT_S)
        p.check(
            C_SCRIPT.format(script),
            rc == 0,
            detail=f"exit={rc}\n{_tail(out)}" if rc != 0 else "entry point resolves",
            repro=f"{R_INSTALL} && {R_VENV}/bin/{script} --help",
            duration_s=s,
        )

    # -- 8. optional extras --------------------------------------------------
    if declared is None:
        p.skip(C_EXTRAS, extras_reason, severity=Severity.WARN)
        p.skip(C_PIPCHECK, extras_reason, severity=Severity.WARN)
        return

    _install_extras(p, declared, vpy, wheel, venv)


def _install_extras(
    p: PhaseResult, declared: Dict[str, List[str]], vpy: Path, wheel: Path, venv: Path
) -> None:
    """Install the light extras cumulatively into the venv already under test.

    Cumulative rather than one-venv-per-extra: it is both an order of magnitude
    faster and the harder case, since it is the combination a user asking for
    `lasagna-rag[eval,vector-backends,storage-all]` actually gets. The trade is
    that a conflict reported at the end may be between two extras rather than
    inside one — `pip check` names the packages, so that stays diagnosable.
    """
    names = sorted(declared)
    light = [e for e in names if e not in HEAVY_EXTRAS]
    for extra in names:
        if extra in HEAVY_EXTRAS:
            p.skip(
                f"extra `{extra}` installs",
                f"deliberately not installed: {HEAVY_EXTRAS[extra]}",
                severity=Severity.WARN,
            )

    total_s = 0.0
    for i, extra in enumerate(light):
        target = f"{wheel}[{extra}]"
        # The repro installs every light extra *up to and including* this one,
        # because that is the venv the check ran against. Installing the extra
        # alone would be a different environment, and for a conflict introduced
        # by an earlier extra it would not reproduce at all.
        #
        # The wheel name is spelled out and the argument quoted on purpose: an
        # unquoted `foo.whl[eval]` is a shell glob character class, so the
        # obvious repro command silently installs the wrong thing.
        so_far = ",".join(light[: i + 1])
        repro = (
            f'{R_INSTALL} && {R_VENV}/bin/python -m pip install "{R_WHEELS}/{wheel.name}[{so_far}]"'
        )
        rc, out, s = _run([str(vpy), "-m", "pip", "install", target], EXTRA_INSTALL_TIMEOUT_S)
        total_s += s
        # Rule 1 again: the timing is recorded whether or not the install stuck.
        p.measure(f"install_{extra}_s", round(s, 1), "s", note="cumulative venv, warm cache")

        # Exit code 0 is not evidence the extra installed anything. `pip install
        # wheel[typo]` prints "does not provide the extra" and exits 0, so a
        # renamed or mis-normalised extra passes this check while
        # `pip install lasagna-rag[typo]` gives the user nothing. Assert that
        # the distributions the extra promises are actually in the venv.
        want = _required_dists(declared.get(extra, []))
        drc, dout, _ = _run([str(vpy), "-c", DIST_PROBE], IMPORT_TIMEOUT_S)
        have = set(_marked(dout, DIST_MARK).split())
        absent = [d for d in want if d not in have] if (drc == 0 and have) else []
        unknown = drc != 0 or not have

        ok = p.check(
            f"extra `{extra}` installs",
            rc == 0 and not absent,
            severity=Severity.WARN,  # an optional extra must not block a release
            detail=(
                f"exit={rc}\n{_tail(out)}"
                if rc != 0
                else f"pip exited 0 but {len(absent)} of {len(want)} promised "
                f"distributions are not in the venv — the extra resolved to nothing: "
                f"{absent}"
                if absent
                else f"installed in {s:.0f}s"
                + (
                    " (could not enumerate distributions to confirm)"
                    if unknown
                    else f"; {len(want)} promised distributions present"
                )
            ),
            repro=repro,
            duration_s=s,
        )
        if not ok:
            # Import is unchanged by an install that did not happen; asserting
            # it here would manufacture a pass.
            p.skip(
                f"`import lasagna` survives extra `{extra}`",
                f"`{extra}` did not install, so there was nothing to re-verify",
                severity=Severity.WARN,
            )
            continue

        # CRITICAL, unlike the install above. An extra failing to resolve
        # degrades an optional feature; an extra that leaves `import lasagna`
        # broken means `pip install lasagna-rag[eval]` produces a package the
        # user cannot import at all, which is the core promise, not an optional
        # one. The venv-path guard is the same as for the core import: without
        # it this asserts only that *some* lasagna is importable.
        rc, out, _ = _run([str(vpy), "-c", IMPORT_PROBE], IMPORT_TIMEOUT_S)
        where = _marked(out, IMPORT_MARK)
        in_venv = bool(where) and where.startswith(str(venv))
        p.check(
            f"`import lasagna` survives extra `{extra}`",
            rc == 0 and in_venv,
            detail=(
                f"exit={rc}\n{_tail(out)}\n(cumulative venv: an earlier extra may share the blame)"
                if rc != 0
                else f"import still clean, from {where or '<probe printed nothing>'}"
                + ("" if in_venv else " — NOT inside the test venv")
            ),
            repro=f"{repro} && {R_VENV}/bin/python -c 'import lasagna'",
        )

    p.measure("extras_install_total_s", round(total_s, 1), "s", note=f"{len(light)} light extras")
    p.measure(
        "footprint_with_extras_mb",
        _dir_mb(_site_packages(venv)),
        "MB",
        note="site-packages after all light extras",
        lower_is_better=True,
    )

    # `pip install` exits 0 while printing a resolver conflict, so the exit
    # codes above do not cover version incompatibility. WARN, and note the
    # cumulative caveat: a conflict between two extras a user would never
    # install together is worth seeing, but is not a broken package.
    #
    # The repro installs the light extras before running `pip check`. It has to:
    # `pip check` on a core-only venv is clean, so the shorter command produces
    # the opposite of the reported result and reads as a false alarm.
    rc, out, s = _run([str(vpy), "-m", "pip", "check"], PIP_CHECK_TIMEOUT_S)
    p.check(
        C_PIPCHECK,
        rc == 0,
        severity=Severity.WARN,
        detail=(
            f"pip check exit={rc}\n{_tail(out)}\n"
            "(cumulative install of every light extra at once — a conflict here "
            "may be between two extras, not inside one)"
        )
        if rc != 0
        else "pip check clean",
        repro=(
            f"{R_INSTALL} && {R_VENV}/bin/python -m pip install "
            f'"{R_WHEELS}/{wheel.name}[{",".join(light)}]" && '
            f"{R_VENV}/bin/python -m pip check"
        ),
        duration_s=s,
    )


def _skip_rest(p: PhaseResult, reason: str) -> None:
    """Record every assertion this phase owns but did not reach as unrun. Rule 2.

    A phase that stops early — or crashes — must not leave its remaining
    assertions absent from the report, because absence reads as "fine" to
    anyone skimming it. Called from the bail-out paths *and* from `run()`'s
    exception handler, so a crash halfway through still produces a full row set.
    """
    # `optional extras install` is a placeholder for the per-extra rows, so it
    # is only worth emitting when none of those exist.
    ran_extras = any(c.name.startswith("extra `") for c in p.checks)
    for name in (
        C_TREE,
        C_BUILD,
        C_SIZE.format(WHEEL_SIZE_BUDGET_MB),
        C_MODULES,
        C_PKGDATA,
        C_EXTRAS_META,
        C_CORE_INSTALL,
        C_IMPORT,
        C_SCRIPT.format("lasagna"),
        C_SCRIPT.format("lasagna-mcp"),
        C_EXTRAS,
        C_PIPCHECK,
    ):
        if name == C_EXTRAS and ran_extras:
            continue
        if any(c.name == name for c in p.checks):
            continue
        p.skip(name, reason)
