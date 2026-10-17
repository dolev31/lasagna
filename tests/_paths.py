"""Shared filesystem anchors for the test suite.

Tests live at the repository root (``tests/``) while the code lives under
``src/lasagna/``, so a test must never derive package paths by walking up
from its own ``__file__`` -- that couples every test to the current layout
and breaks the moment the suite is run against an installed wheel.

``find_spec`` resolves the package directory wherever it actually is, and
does so *without* executing ``lasagna/__init__.py``, which matters because
several tests deliberately load modules by file path to avoid pulling in
the heavy import chain.
"""

import importlib.util
import subprocess
from pathlib import Path

_spec = importlib.util.find_spec("lasagna")
if _spec is None or not _spec.submodule_search_locations:  # pragma: no cover
    raise RuntimeError(
        "The 'lasagna' package is not importable. Install it first: pip install -e ."
    )

#: Directory containing the installed/checked-out ``lasagna`` package.
PACKAGE_DIR = Path(_spec.submodule_search_locations[0])

#: Deployment manifests (docker-compose, Dockerfile, k8s overlays).
DEPLOY_DIR = PACKAGE_DIR / "deploy"

#: Configuration UI, including its Vite/React frontend sources.
CONFIGURATOR_DIR = PACKAGE_DIR / "configurator"

#: Folders that hold build output or installed dependencies, never shipped text.
_NOT_SHIPPED = {"build", "dist", "node_modules", ".venv"}


def tracked(paths, root):
    """The files among ``paths`` that git tracks under ``root``.

    A plain glob also read git-ignored files, such as the READMEs of the
    configurator's ``node_modules``, and failed on text that ships nowhere.
    Outside a git checkout (an installed wheel), keeps the files outside build
    output and dependency folders.
    """
    paths = [Path(p) for p in paths]
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        return sorted(p for p in paths if not _NOT_SHIPPED & set(p.parts))
    names = {(Path(root) / name).resolve() for name in out.split("\0") if name}
    return sorted(p for p in paths if p.resolve() in names)


__all__ = ["PACKAGE_DIR", "DEPLOY_DIR", "CONFIGURATOR_DIR", "tracked"]
