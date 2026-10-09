"""Phase 9 — Lifecycle: upgrade over a previous release, then uninstall.

Every other phase asks whether a *fresh* install works. This one asks what
happens on the second day: the user already has Lasagna installed, a new
version ships, and later they remove it. Both transitions touch state the
fresh-install path never exercises, and both fail in ways that look like
success — a stale module that keeps importing after it was deleted from the
package, or an uninstall that leaves a half-populated ``site-packages`` behind.

What this phase claims to prove
-------------------------------
1. Installing a newer version **over** an existing one leaves exactly one
   distribution registered, reporting the new version.
2. The upgrade **replaces** the package directory rather than merging into it.
   Verified positively: a probe module is injected into the older wheel, its
   presence is asserted *before* the upgrade (so the check cannot pass
   vacuously), and its absence is asserted after.
3. The package still imports and both console scripts still resolve after the
   upgrade — an upgrade that silently breaks the entry points is the failure a
   user hits first.
4. ``pip uninstall`` removes the package, its ``dist-info`` and its console
   scripts, leaving nothing of the distribution behind in ``site-packages``.
5. After uninstall, ``import lasagna`` fails. An import that still succeeds
   means a shadow copy is on the path.
6. Uninstall does not touch user data under ``$HOME``. Checked against an
   isolated ``HOME``, never the developer's real one.
7. The package can be installed again cleanly after being removed.

What this phase does NOT prove
------------------------------
* **The "previous release" is synthetic.** ``lasagna-rag`` has never been
  published, so there is no real earlier version to upgrade from. The older
  wheel here is the freshly built one, repacked with a lower version number
  and one extra module. That is a faithful test of *pip's replacement of an
  installed distribution* — which is what upgrade hygiene means — but it is
  not a test of migrating real data or configuration written by an older
  release, because no older release exists to have written any.
* **No dependency churn.** Both wheels declare identical requirements, so this
  never exercises the case where an upgrade needs to move a shared dependency
  and conflicts with something else in the environment.
* **Uninstall is pip's, not ours.** The package ships no custom uninstall hook,
  so checks 4-6 are largely a regression guard: they will start earning their
  keep the day someone adds a post-install script or a ``.pth`` file.
* One interpreter, one platform, warm pip cache — the same caveats as p1.

Severity note
-------------
The original plan classified this phase as *warn*. Its checks are CRITICAL
instead, deliberately.

WARN exists for thresholds a loaded laptop can miss without the package being
broken — a p95 latency target, an install wall-time. Nothing in this phase is
like that. Every check here is a deterministic property of what pip did with
two wheels: the module is importable or it is not, the script is on PATH or it
is not. The same input gives the same answer on any machine, so the argument
that justifies WARN elsewhere has nothing to bite on, and ``e2e/README.md``
states the governing rule directly: *correctness is never WARN*.

A stale module still importable after an upgrade means a user is running code
that was deleted from the release. A console script left behind by an uninstall
is a command on their PATH that cannot work. Neither should ship.

The one check that stays WARN is the upgrade-hygiene assertion when its probe
could not be established — see the ``skip`` beside it. That is an unrun check,
not a passing one.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

from ..contract import PhaseResult, Severity
from .p1_install import (
    BUILD_TIMEOUT_S,
    CORE_INSTALL_TIMEOUT_S,
    IMPORT_TIMEOUT_S,
    REPO_ROOT,
    VENV_TIMEOUT_S,
    _dir_mb,
    _run,
    _site_packages,
    _tail,
    _venv_python,
)

# The synthetic "already installed" release. Must sort below the real version.
OLD_VERSION = "0.0.9"

# Injected into the old wheel only. Named so a stray copy is unmistakable if it
# ever turns up in a real environment.
PROBE_MODULE = "_p9_removed_in_next_version"
PROBE_SOURCE = (
    "# Injected by the Lasagna E2E lifecycle phase into a synthetic older\n"
    "# release. It must not survive an upgrade. If you are reading this inside\n"
    "# a real installation, an upgrade merged into the package directory\n"
    "# instead of replacing it.\n"
    "MARKER = 'p9-stale'\n"
)

UNINSTALL_TIMEOUT_S = 600.0

R_VENV = "/tmp/lasagna-p9-venv"
R_WHEELS = "/tmp/lasagna-p9-wheels"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_in(args: List[str], timeout: float, home: Path) -> Tuple[Optional[int], str, float]:
    """Run a child that *uses* the package, with an isolated HOME.

    Anything the package writes on first use — ``~/.rag_knowledge`` and its
    SQLite database — must land in a throwaway directory. A gate that polluted
    the developer's real home while checking that uninstall is tidy would be a
    poor joke.

    Implemented by patching ``os.environ`` around the call rather than passing
    an environment down, because :func:`_run` builds its own child environment
    from ``os.environ`` (deliberately: pip needs the real PATH, proxy and CA
    settings). Restoring in ``finally`` keeps that contained; the gate is
    single-threaded, so no other call can observe the swap.

    ``PIP_CACHE_DIR`` is not redirected: only runtime children come through
    here, and pointing pip at an empty cache would re-download torch.
    """
    saved = {k: os.environ.get(k) for k in ("HOME", "USERPROFILE")}
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    try:
        return _run(args, timeout)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _project_version() -> Optional[str]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    try:
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    v = data.get("project", {}).get("version")
    return str(v) if v else None


def _entries(site: Path) -> Set[str]:
    """Top-level names in site-packages, for before/after leftover comparison."""
    try:
        return {p.name for p in site.iterdir()}
    except OSError:
        return set()


def _repack(wheel: Path, new_version: str, out_dir: Path) -> Tuple[Optional[Path], str]:
    """Rebuild `wheel` under a different version, with the probe module added.

    RECORD is regenerated rather than patched: pip validates the hashes it
    lists, so a stale RECORD would fail the install for a reason that has
    nothing to do with what this phase is testing.
    """
    stem = wheel.name[: -len(".whl")]
    parts = stem.split("-")
    if len(parts) < 3:
        return None, f"unexpected wheel filename: {wheel.name}"
    dist, old_version, tags = parts[0], parts[1], parts[2:]

    stage = out_dir / f"stage-{new_version}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(stage)
    except (OSError, zipfile.BadZipFile) as exc:
        return None, f"could not unpack {wheel.name}: {type(exc).__name__}: {exc}"

    old_di = stage / f"{dist}-{old_version}.dist-info"
    if not old_di.is_dir():
        return None, f"{old_di.name} not found inside the wheel"
    new_di = stage / f"{dist}-{new_version}.dist-info"
    old_di.rename(new_di)

    meta = new_di / "METADATA"
    try:
        text = meta.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not read METADATA: {exc}"
    text, n = re.subn(r"(?m)^Version: .*$", f"Version: {new_version}", text, count=1)
    if n != 1:
        return None, "METADATA has no Version: line to rewrite"
    meta.write_text(text, encoding="utf-8")

    pkg = stage / "lasagna"
    if not pkg.is_dir():
        return None, "the wheel contains no `lasagna/` package directory"
    (pkg / f"{PROBE_MODULE}.py").write_text(PROBE_SOURCE, encoding="utf-8")

    record = new_di / "RECORD"
    record_rel = record.relative_to(stage).as_posix()
    lines: List[str] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(stage).as_posix()
        if rel == record_rel:
            continue
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        lines.append(f"{rel},sha256={digest},{len(data)}")
    lines.append(f"{record_rel},,")
    record.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = out_dir / f"{dist}-{new_version}-{'-'.join(tags)}.whl"
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())
    except OSError as exc:
        return None, f"could not write the repacked wheel: {exc}"
    shutil.rmtree(stage, ignore_errors=True)
    return out, ""


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


def run() -> PhaseResult:
    p = PhaseResult(key="p9", title="Lifecycle (upgrade, uninstall)")
    workdir: Optional[Path] = None
    try:
        workdir = Path(tempfile.mkdtemp(prefix="lasagna-e2e-p9-"))
        _body(p, workdir)
    except Exception as exc:
        p.check(
            "phase p9 ran to completion",
            False,
            detail=f"unhandled {type(exc).__name__}: {exc}",
            repro="python -m e2e.run --phase p9",
        )
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
    return p


def _body(p: PhaseResult, workdir: Path) -> None:
    wheels = workdir / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    venv = workdir / "venv"
    home = workdir / "home"
    home.mkdir(parents=True, exist_ok=True)

    version = _project_version()
    if not version:
        p.skipped_reason = "could not read project.version from pyproject.toml"
        return

    # -- 0. build the wheel under test ---------------------------------------
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
            "no build frontend available: `uvx` is not on PATH and `build` is not "
            "importable by this interpreter"
        )
        return

    rc, out, build_s = _run(build_cmd, BUILD_TIMEOUT_S, cwd=REPO_ROOT)
    built = sorted(wheels.glob(f"*-{version}-*.whl"))
    if not p.check(
        "wheel builds (the version being upgraded to)",
        rc == 0 and bool(built),
        detail=f"exit={rc}\n{_tail(out)}" if rc != 0 else "no matching .whl produced",
        repro=f"cd {REPO_ROOT} && uvx --from build pyproject-build --wheel --outdir {R_WHEELS} .",
        duration_s=build_s,
    ):
        _skip_rest(p, "no wheel was produced, so no lifecycle transition could be attempted")
        return
    new_wheel = built[0]

    # -- 1. synthesise the release being upgraded from -----------------------
    old_wheel, why = _repack(new_wheel, OLD_VERSION, wheels)
    if old_wheel is None:
        p.skipped_reason = f"could not synthesise a previous release: {why}"
        return

    # -- 2. venv ------------------------------------------------------------
    rc, out, _ = _run([sys.executable, "-m", "venv", str(venv)], VENV_TIMEOUT_S)
    vpy = _venv_python(venv)
    if not p.check(
        "clean venv created",
        rc == 0 and vpy.exists(),
        detail=f"exit={rc}\n{_tail(out)}",
        repro=f"{sys.executable} -m venv {R_VENV}",
    ):
        _skip_rest(p, "no virtual environment, so nothing could be installed into one")
        return

    site = _site_packages(venv)
    baseline = _entries(site)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    baseline_bin = _entries(bindir)

    # -- 3. install the older release ---------------------------------------
    rc, out, old_s = _run(
        [str(vpy), "-m", "pip", "install", str(old_wheel)], CORE_INSTALL_TIMEOUT_S
    )
    p.measure(
        "previous_version_install_s",
        round(old_s, 1),
        "s",
        note=f"synthetic {OLD_VERSION}, warm pip cache",
    )
    if not p.check(
        f"synthetic previous release {OLD_VERSION} installs",
        rc == 0,
        detail=f"exit={rc}\n{_tail(out)}",
        repro=f"{R_VENV}/bin/python -m pip install {R_WHEELS}/{old_wheel.name}",
        duration_s=old_s,
    ):
        _skip_rest(p, "the previous release did not install, so there was nothing to upgrade")
        return

    # The probe must be present *before* the upgrade, or its later absence
    # proves nothing. This is the check that stops check 5 being vacuous.
    rc, out, _ = _run_in(
        [str(vpy), "-c", f"import lasagna.{PROBE_MODULE} as m; print(m.MARKER)"],
        IMPORT_TIMEOUT_S,
        home,
    )
    probe_live = p.check(
        "upgrade probe is present in the installed previous release",
        rc == 0 and "p9-stale" in out,
        detail=(
            f"exit={rc}\n{_tail(out)}"
            if rc != 0
            else "probe importable, so its removal is a meaningful assertion"
        ),
        repro=f"{R_VENV}/bin/python -c 'import lasagna.{PROBE_MODULE}'",
    )

    # -- 4. upgrade ----------------------------------------------------------
    rc, out, up_s = _run(
        [str(vpy), "-m", "pip", "install", "--upgrade", str(new_wheel)],
        CORE_INSTALL_TIMEOUT_S,
    )
    p.measure("upgrade_s", round(up_s, 1), "s", note=f"{OLD_VERSION} -> {version}, warm cache")
    upgraded = p.check(
        f"upgrade {OLD_VERSION} -> {version} succeeds",
        rc == 0,
        detail=f"exit={rc}\n{_tail(out)}",
        repro=f"{R_VENV}/bin/python -m pip install --upgrade {R_WHEELS}/{new_wheel.name}",
        duration_s=up_s,
    )

    if not upgraded:
        _skip_rest(p, "the upgrade failed, so the resulting environment proves nothing")
        return

    rc, out, _ = _run_in(
        [str(vpy), "-c", "import importlib.metadata as m; print(m.version('lasagna-rag'))"],
        IMPORT_TIMEOUT_S,
        home,
    )
    reported = out.strip().splitlines()[-1] if rc == 0 and out.strip() else ""
    p.check(
        "the upgraded environment reports the new version",
        reported == version,
        detail=f"importlib.metadata reports {reported or '<nothing>'}, expected {version}",
        repro=f"{R_VENV}/bin/python -c \"import importlib.metadata as m; print(m.version('lasagna-rag'))\"",
    )

    dist_infos = sorted(site.glob("lasagna_rag-*.dist-info"))
    p.check(
        "exactly one distribution is registered after the upgrade",
        len(dist_infos) == 1,
        detail=f"found {[d.name for d in dist_infos]}",
        repro=f"ls -d {R_VENV}/lib/*/site-packages/lasagna_rag-*.dist-info",
    )

    rc, out, _ = _run_in(
        [str(vpy), "-c", "import lasagna; print(lasagna.__file__)"], IMPORT_TIMEOUT_S, home
    )
    p.check(
        "`import lasagna` still works after the upgrade",
        rc == 0,
        detail=f"exit={rc}\n{_tail(out)}" if rc != 0 else out.strip().splitlines()[-1],
        repro=f"{R_VENV}/bin/python -c 'import lasagna'",
    )

    for script in ("lasagna", "lasagna-mcp"):
        exe = bindir / script
        if not exe.exists():
            p.check(
                f"console script `{script}` survives the upgrade",
                False,
                detail=f"{exe} does not exist after the upgrade",
                repro=f"ls {R_VENV}/bin/",
            )
            continue
        rc, out, _ = _run_in([str(exe), "--help"], IMPORT_TIMEOUT_S, home)
        p.check(
            f"console script `{script}` survives the upgrade",
            rc == 0,
            detail=f"exit={rc}\n{_tail(out)}" if rc != 0 else "entry point still resolves",
            repro=f"{R_VENV}/bin/{script} --help",
        )

    # -- 5. the upgrade replaced the package, it did not merge into it -------
    if not probe_live:
        p.skip(
            "the upgrade removes a module deleted in the new version",
            "the probe was not importable before the upgrade, so its absence "
            "afterwards would prove nothing",
            severity=Severity.WARN,
        )
    else:
        rc, out, _ = _run_in(
            [str(vpy), "-c", f"import lasagna.{PROBE_MODULE}"], IMPORT_TIMEOUT_S, home
        )
        p.check(
            "the upgrade removes a module deleted in the new version",
            rc != 0,
            detail=(
                "stale module still importable after the upgrade — the package "
                "directory was merged into rather than replaced"
                if rc == 0
                else "stale module is gone, as it must be"
            ),
            repro=f"{R_VENV}/bin/python -c 'import lasagna.{PROBE_MODULE}'",
        )

    p.measure("footprint_before_uninstall_mb", _dir_mb(site), "MB")

    # -- 6. user data, then uninstall ---------------------------------------
    # Written under the isolated HOME so the real ~/.rag_knowledge is never in
    # scope. Content matters: an uninstall that recreates an empty directory
    # would pass an existence check while having destroyed the data.
    data_dir = home / ".rag_knowledge"
    data_dir.mkdir(parents=True, exist_ok=True)
    canary = data_dir / "p9-user-data.txt"
    canary.write_text("user documents live here", encoding="utf-8")

    rc, out, un_s = _run(
        [str(vpy), "-m", "pip", "uninstall", "-y", "lasagna-rag"], UNINSTALL_TIMEOUT_S
    )
    p.measure("uninstall_s", round(un_s, 1), "s")
    uninstalled = p.check(
        "`pip uninstall lasagna-rag` succeeds",
        rc == 0,
        detail=f"exit={rc}\n{_tail(out)}",
        repro=f"{R_VENV}/bin/python -m pip uninstall -y lasagna-rag",
        duration_s=un_s,
    )

    p.check(
        "uninstall leaves user data under $HOME alone",
        canary.is_file() and canary.read_text(encoding="utf-8").startswith("user documents"),
        detail=(
            "canary file intact"
            if canary.is_file()
            else f"{canary} was removed or emptied by the uninstall"
        ),
        repro=f"HOME=/tmp/lasagna-p9-home {R_VENV}/bin/python -m pip uninstall -y lasagna-rag",
    )

    if not uninstalled:
        _skip_rest(p, "the uninstall failed, so the resulting environment proves nothing")
        return

    rc, out, _ = _run_in([str(vpy), "-c", "import lasagna"], IMPORT_TIMEOUT_S, home)
    p.check(
        "`import lasagna` fails after uninstall",
        rc != 0,
        detail=(
            "the package still imports after being uninstalled — a shadow copy "
            f"is on the path: {_tail(out, 300)}"
            if rc == 0
            else "import fails, as it must"
        ),
        repro=f"{R_VENV}/bin/python -c 'import lasagna'",
    )

    leftovers = sorted(
        n
        for n in _entries(site) - baseline
        if n.startswith("lasagna") or n.startswith("lasagna_rag")
    )
    p.measure(
        "leftover_files",
        float(len(leftovers)),
        "entries",
        note="distribution entries still in site-packages after uninstall",
    )
    p.check(
        "uninstall leaves nothing of the distribution in site-packages",
        not leftovers,
        detail=f"left behind: {leftovers}" if leftovers else "site-packages is back to baseline",
        repro=f"ls {R_VENV}/lib/*/site-packages | grep -i lasagna",
    )

    stale_scripts = sorted(n for n in _entries(bindir) - baseline_bin if n.startswith("lasagna"))
    p.check(
        "uninstall removes the console scripts",
        not stale_scripts,
        detail=f"still present: {stale_scripts}" if stale_scripts else "both scripts removed",
        repro=f"ls {R_VENV}/bin | grep lasagna",
    )

    # -- 7. reinstall on the cleaned environment -----------------------------
    rc, out, re_s = _run([str(vpy), "-m", "pip", "install", str(new_wheel)], CORE_INSTALL_TIMEOUT_S)
    p.measure("reinstall_s", round(re_s, 1), "s", note="after a full uninstall")
    if p.check(
        "the package reinstalls cleanly after being removed",
        rc == 0,
        detail=f"exit={rc}\n{_tail(out)}",
        repro=f"{R_VENV}/bin/python -m pip install {R_WHEELS}/{new_wheel.name}",
        duration_s=re_s,
    ):
        rc, out, _ = _run_in([str(vpy), "-c", "import lasagna"], IMPORT_TIMEOUT_S, home)
        p.check(
            "`import lasagna` works again after reinstall",
            rc == 0,
            detail=f"exit={rc}\n{_tail(out)}" if rc != 0 else "import restored",
            repro=f"{R_VENV}/bin/python -c 'import lasagna'",
        )
    else:
        p.skip(
            "`import lasagna` works again after reinstall",
            "the reinstall failed, so there was nothing to import",
        )


def _skip_rest(p: PhaseResult, reason: str) -> None:
    """Record everything downstream of a hard stop as unrun, never as absent."""
    for name in (
        f"synthetic previous release {OLD_VERSION} installs",
        "upgrade probe is present in the installed previous release",
        "the upgraded environment reports the new version",
        "exactly one distribution is registered after the upgrade",
        "`import lasagna` still works after the upgrade",
        "console script `lasagna` survives the upgrade",
        "console script `lasagna-mcp` survives the upgrade",
        "the upgrade removes a module deleted in the new version",
        "`pip uninstall lasagna-rag` succeeds",
        "uninstall leaves user data under $HOME alone",
        "`import lasagna` fails after uninstall",
        "uninstall leaves nothing of the distribution in site-packages",
        "uninstall removes the console scripts",
        "the package reinstalls cleanly after being removed",
        "`import lasagna` works again after reinstall",
    ):
        if any(c.name == name for c in p.checks):
            continue
        p.skip(name, reason)
