"""Import pymilvus without reading anyone's ``.env`` file.

``pymilvus/settings.py`` calls ``dotenv.load_dotenv()`` when it is imported.
Given no path, python-dotenv searches for a ``.env`` from the directory of the
calling file (or the working directory, in a notebook, ``python -c`` or under a
debugger) up to ``/``. A virtualenv inside a project therefore found
that project's ``.env``, and every key in it, such as an unrelated
``OPENROUTER_API_KEY``, appeared in this process's environment, where LLM
provider auto-detection picked it up.

Every pymilvus import in the package goes through this module.
``tests/test_pymilvus_import_guard.py`` fails if one does not.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from types import ModuleType
from typing import Any, Dict, Tuple

_lock = threading.Lock()


def _skip_dotenv(*args: Any, **kwargs: Any) -> bool:
    """Stands in for ``dotenv.load_dotenv`` while pymilvus imports."""
    return False


def _restore_environ(before: Dict[str, str]) -> None:
    for key in set(os.environ) - set(before):
        del os.environ[key]
    for key, value in before.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def import_pymilvus() -> ModuleType:
    """Return the ``pymilvus`` module, imported without loading a ``.env``.

    ``dotenv.load_dotenv`` is a no-op for the duration of the import, and
    ``os.environ`` is put back exactly as it was afterwards, in case a pymilvus
    release reads the file some other way. Both are restored even when the
    import fails. Once pymilvus is imported this returns it at once.

    Raises:
        ImportError: pymilvus is not installed or cannot be imported.
    """
    module = sys.modules.get("pymilvus")
    if module is not None:
        return module
    with _lock:
        module = sys.modules.get("pymilvus")
        if module is not None:
            return module
        try:
            import dotenv
            import dotenv.main
        except ImportError:
            # pymilvus depends on python-dotenv, so its own import says what
            # is missing.
            return importlib.import_module("pymilvus")
        # gRPC starts with pymilvus; a fork after that floods stderr. See there.
        from lasagna._quiet import tqdm_without_process_lock

        tqdm_without_process_lock()
        saved = (dotenv.load_dotenv, dotenv.main.load_dotenv)
        before = dict(os.environ)
        dotenv.load_dotenv = dotenv.main.load_dotenv = _skip_dotenv
        try:
            return importlib.import_module("pymilvus")
        finally:
            dotenv.load_dotenv, dotenv.main.load_dotenv = saved
            _restore_environ(before)


def from_pymilvus(*names: str) -> Tuple[Any, ...]:
    """``from pymilvus import a, b`` as ``a, b = from_pymilvus("a", "b")``.

    Raises ImportError for a missing name, as the ``from`` statement does, so
    callers that fall back on ImportError keep working.
    """
    module = import_pymilvus()
    values = []
    for name in names:
        try:
            values.append(getattr(module, name))
        except AttributeError:
            raise ImportError(f"cannot import name {name!r} from 'pymilvus'") from None
    return tuple(values)
