"""Every ``mock.patch("a.b.c")`` target must name something that exists.

A patch target that does not resolve is the quietest way a test can stop
testing. With ``create=True`` the patch invents the attribute, succeeds, and
the test goes on asserting things about a mock the code under test never
touches — so it passes, and keeps passing through exactly the bug it was
written to catch. Without ``create=True`` it raises, but only on the code path
that reaches it, which for a service-gated or rarely-run test may be never.

This found four such tests when it was first run as a one-off:

* ``test_small_queue_no_deadlock`` patched a module-level alias the ingest
  function never reads (it re-imports the config locally), so the small batch
  sizes it sets never applied and the backpressure path in its name was never
  exercised.
* ``test_soft_delete_does_not_free_quota`` patched a name its module has never
  had, then asserted the invented mock was not called — true no matter what
  the code did.
* Two more ``create=True`` patches on names that do not exist, each beside a
  correct patch of the real target.

Keeping it as a test rather than a script is the point: the failure mode
reappears silently whenever an import moves.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import List, Tuple

import pytest

TESTS_DIR = Path(__file__).resolve().parent
ROOT_PACKAGE = "lasagna"


def _patch_targets() -> List[Tuple[str, str, int]]:
    """Every string target passed to a ``patch(...)`` call under ``tests/``.

    Returns (target, file, lineno). Only dotted strings rooted at the package
    under test are collected: patching ``os.environ`` or a third-party name is
    not this test's business.
    """
    found: List[Tuple[str, str, int]] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):  # pragma: no cover - unreadable file
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name != "patch":
                continue
            arg = node.args[0]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            target = arg.value
            if target.startswith(f"{ROOT_PACKAGE}.") and "." in target:
                rel = path.relative_to(TESTS_DIR.parent)
                found.append((target, str(rel), node.lineno))
    return found


def _resolve(target: str) -> Tuple[bool, str]:
    """Try to reach ``target`` by importing the longest importable prefix.

    Returns (resolved, detail). A module that cannot be imported at all — an
    optional dependency, say — is reported as unresolvable-for-import-reasons
    so the caller can skip rather than fail it.
    """
    parts = target.split(".")
    obj = None
    rest: List[str] = []
    last_error = ""
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            rest = parts[i:]
            break
        except Exception as exc:  # noqa: BLE001 - any import failure is a skip
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    if obj is None:
        return False, f"no importable prefix ({last_error})"

    for attr in rest:
        if not hasattr(obj, attr):
            missing = _optional_dependency_behind(obj, attr) if inspect.ismodule(obj) else ""
            if missing:
                return False, f"optional dependency {missing!r} is not installed"
            return False, f"{'.'.join(parts[: len(parts) - len(rest)])} has no attribute {attr!r}"
        obj = getattr(obj, attr)
    return True, "ok"


def _optional_dependency_behind(module, attr: str) -> str:
    """The missing package, if ``module`` binds ``attr`` only behind a failed optional import.

    Backends guard their client libraries with ``try: from pkg import x as y``
    / ``except ImportError``, so ``y`` exists only when ``pkg`` is installed.
    A patch of ``module.y`` is then correct, and the test that makes it skips
    itself without ``pkg``; failing it here would report a missing extra as a
    broken test. Names that are absent for any other reason still fail.
    """
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        guards_import = any(
            handler.type is not None
            and any(
                isinstance(n, ast.Name) and n.id in {"ImportError", "ModuleNotFoundError"}
                for n in ast.walk(handler.type)
            )
            for handler in node.handlers
        )
        if not guards_import:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
                bound = {alias.asname or alias.name for alias in stmt.names}
                package = stmt.module.split(".")[0]
            elif isinstance(stmt, ast.Import):
                bound = {alias.asname or alias.name.split(".")[0] for alias in stmt.names}
                package = stmt.names[0].name.split(".")[0]
            else:
                continue
            if attr in bound and importlib.util.find_spec(package) is None:
                return package
    return ""


TARGETS = _patch_targets()


def test_the_scan_found_targets_to_check():
    """Guard against this whole file silently becoming a no-op.

    If the AST walk stops matching — a refactor to a different mocking helper,
    say — every parametrised case below would vanish and the suite would look
    greener, not redder.
    """
    assert len(TARGETS) >= 20, (
        f"only {len(TARGETS)} patch targets found; the scan is probably broken"
    )


@pytest.mark.parametrize(
    "target,source,lineno",
    TARGETS,
    ids=[f"{t}" for t, _s, _l in TARGETS],
)
def test_patch_target_resolves(target, source, lineno):
    resolved, detail = _resolve(target)
    if not resolved and detail.startswith("no importable prefix"):
        pytest.skip(f"{target}: module not importable here ({detail})")
    if not resolved and detail.startswith("optional dependency"):
        pytest.skip(f"{target}: {detail}")
    assert resolved, (
        f"{source}:{lineno} patches {target!r}, but {detail}.\n"
        f"The patch therefore controls nothing the code under test reads. With "
        f"create=True it will still succeed, and the test will pass while "
        f"verifying nothing. Point it at the name the code actually resolves — "
        f"note that a function-local `from x import y` is resolved through `x`, "
        f"not through the module doing the importing."
    )
