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

__all__ = ["PACKAGE_DIR", "DEPLOY_DIR", "CONFIGURATOR_DIR"]
