"""Asking for a built-in backend whose library is missing names the extra.

``get_backend("qdrant")`` without qdrant-client said "Unknown backend
'qdrant'. Available: milvus", as if Qdrant did not exist, and named nothing to
install. Same for the S3 and fsspec storage backends.
"""

from __future__ import annotations

import pytest


def _vector_registry_without(name):
    from lasagna.vector_backends.registry import BackendRegistry

    registry = BackendRegistry()
    registry._backends.pop(name, None)
    return registry


@pytest.mark.parametrize(
    "name,extra",
    [("qdrant", "vector-backends"), ("pgvector", "pgvector"), ("opensearch", "opensearch")],
)
def test_a_vector_backend_without_its_client_names_the_extra(name, extra):
    with pytest.raises(ValueError, match=rf"lasagna-rag\[{extra}\]"):
        _vector_registry_without(name).create(name)


def test_an_unknown_vector_backend_names_none():
    with pytest.raises(ValueError) as err:
        _vector_registry_without("nope").create("nope")
    assert "lasagna-rag[" not in str(err.value)


@pytest.mark.parametrize("name,extra", [("s3", "storage-s3"), ("fsspec", "storage-all")])
def test_a_storage_backend_without_its_library_names_the_extra(name, extra):
    from lasagna.storage.registry import StorageRegistry

    registry = StorageRegistry()
    registry._backends.pop(name, None)
    with pytest.raises(ValueError, match=rf"lasagna-rag\[{extra}\]"):
        registry.create(name)


def test_every_named_extra_exists():
    import sys
    from pathlib import Path

    from lasagna.storage.registry import _EXTRA_FOR_BACKEND as storage_extras
    from lasagna.vector_backends.registry import _EXTRA_FOR_BACKEND as vector_extras

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        import tomli as tomllib
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    extras = set(tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"])
    assert set(vector_extras.values()) | set(storage_extras.values()) <= extras
