"""Tests for pluggable storage backend system.

Tests cover:
- LocalStorageBackend: store, retrieve, delete, exists, ensure_local
- StorageRegistry: register, create, singleton caching, unregister
- Module-level convenience functions: get_storage_backend, register_storage_backend
- Custom backend registration (bring your own)
- Path traversal prevention (security)
"""

import io
import os
import tempfile
import shutil

import pytest

from lasagna.storage.base import (
    StorageBackend,
    StorageCapabilities,
    StorageFileInfo,
    StorageError,
    StorageNotFoundError,
)
from lasagna.storage.local_backend import LocalStorageBackend
from lasagna.storage.registry import (
    StorageRegistry,
    get_storage_backend,
    get_available_storage_backends,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for testing."""
    d = tempfile.mkdtemp(prefix="rag_storage_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def local_storage(tmp_dir):
    """Create a LocalStorageBackend with a temp directory."""
    return LocalStorageBackend(base_dir=tmp_dir)


@pytest.fixture
def sample_file_content():
    """Sample file content for testing."""
    return b"Hello, this is a test document for RAG storage backend testing."


@pytest.fixture
def sample_file(sample_file_content):
    """Create a BytesIO stream with sample content."""
    return io.BytesIO(sample_file_content)


# =============================================================================
# LocalStorageBackend Tests
# =============================================================================


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    def test_backend_name(self, local_storage):
        assert local_storage.backend_name == "local"

    def test_capabilities(self, local_storage):
        caps = local_storage.capabilities
        assert isinstance(caps, StorageCapabilities)
        assert caps.supports_streaming is True
        assert caps.supports_presigned_urls is False

    def test_store_and_retrieve(self, local_storage, sample_file, sample_file_content):
        info = local_storage.store(sample_file, "doc_test123", "report.pdf")

        assert isinstance(info, StorageFileInfo)
        assert info.size == len(sample_file_content)
        assert "doc_test123" in info.uri
        assert "report.pdf" in info.uri

        # Retrieve
        stream = local_storage.retrieve(info.uri)
        data = stream.read()
        stream.close()
        assert data == sample_file_content

    def test_store_creates_directory(self, local_storage, sample_file, tmp_dir):
        _info = local_storage.store(sample_file, "doc_newdir", "test.txt")
        assert os.path.exists(os.path.join(tmp_dir, "doc_newdir", "test.txt"))

    def test_exists(self, local_storage, sample_file):
        info = local_storage.store(sample_file, "doc_exists", "check.txt")
        assert local_storage.exists(info.uri) is True
        assert local_storage.exists("/nonexistent/path.txt") is False
        assert local_storage.exists("") is False

    def test_get_size(self, local_storage, sample_file, sample_file_content):
        info = local_storage.store(sample_file, "doc_size", "sized.txt")
        assert local_storage.get_size(info.uri) == len(sample_file_content)

    def test_get_size_not_found(self, local_storage):
        with pytest.raises(StorageNotFoundError):
            local_storage.get_size("/nonexistent/file.txt")

    def test_delete(self, local_storage, sample_file, tmp_dir):
        info = local_storage.store(sample_file, "doc_delete", "deleteme.txt")
        assert local_storage.exists(info.uri) is True

        result = local_storage.delete(info.uri)
        assert result is True

        # Directory should be removed too
        assert not os.path.exists(os.path.join(tmp_dir, "doc_delete"))

    def test_delete_nonexistent(self, local_storage):
        assert local_storage.delete("/nonexistent/path.txt") is False
        assert local_storage.delete("") is False

    def test_get_local_path(self, local_storage, sample_file):
        info = local_storage.store(sample_file, "doc_local", "local.txt")
        path = local_storage.get_local_path(info.uri)
        assert path == info.uri
        assert os.path.exists(path)

    def test_get_local_path_nonexistent(self, local_storage):
        assert local_storage.get_local_path("/nonexistent") is None

    def test_ensure_local(self, local_storage, sample_file):
        info = local_storage.store(sample_file, "doc_ensure", "ensure.txt")
        path = local_storage.ensure_local(info.uri)
        assert path == info.uri  # Local storage returns same path (zero-copy)

    def test_ensure_local_not_found(self, local_storage):
        with pytest.raises(StorageNotFoundError):
            local_storage.ensure_local("/nonexistent/file.txt")

    def test_path_traversal_prevention(self, local_storage):
        """Ensure path traversal attacks are blocked."""
        evil_file = io.BytesIO(b"malicious content")
        with pytest.raises(StorageError):
            local_storage.store(evil_file, "doc_evil", "../../../etc/passwd")

    def test_path_traversal_via_document_id(self, local_storage, tmp_dir):
        """Ensure traversal via document_id is blocked without side effects."""
        evil_file = io.BytesIO(b"malicious content")
        with pytest.raises(StorageError):
            local_storage.store(evil_file, "../escape_dir", "file.txt")

        # Verify no directory was created outside base
        escape_path = os.path.join(tmp_dir, "..", "escape_dir")
        assert not os.path.exists(escape_path)

    @pytest.mark.parametrize(
        "doc_id",
        [
            "../../etc",
            "../sibling",
        ],
    )
    def test_path_traversal_document_id_variants(self, local_storage, tmp_dir, doc_id):
        """Traversal patterns that escape base_dir are blocked without side effects."""
        evil_file = io.BytesIO(b"test")
        with pytest.raises(StorageError):
            local_storage.store(evil_file, doc_id, "file.txt")

        # Verify no directory was created outside base
        escaped = os.path.realpath(os.path.join(tmp_dir, doc_id))
        assert not os.path.exists(escaped)

    def test_store_with_content_type(self, local_storage, sample_file):
        info = local_storage.store(sample_file, "doc_ct", "doc.pdf", content_type="application/pdf")
        assert info.content_type == "application/pdf"


# =============================================================================
# is_safe_document_id Tests
# =============================================================================


def _get_is_safe_document_id():
    """Import is_safe_document_id directly from the module file.

    Uses importlib to load ``_validation.py`` without triggering the
    ``api/__init__.py`` package init, which pulls heavy runtime deps
    (dynaconf, embedding managers, etc.).  This keeps the validation
    tests hermetic and runnable in minimal environments.
    """
    import importlib.util
    import pathlib

    # find_spec locates the package without executing its __init__, keeping
    # this helper free of the heavy import chain described above.
    mod_path = (
        pathlib.Path(importlib.util.find_spec("lasagna").submodule_search_locations[0])
        / "api"
        / "_validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lasagna.api._validation_standalone", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.is_safe_document_id


class TestIsSafeDocumentId:
    """Tests for is_safe_document_id validation.

    Uses a direct module import to avoid triggering the heavy
    ``api/__init__.py`` → ``core/config.py`` → dynaconf chain.
    """

    _is_safe = staticmethod(_get_is_safe_document_id())

    @pytest.mark.parametrize(
        "doc_id",
        [
            "doc_abc123def456789a",  # auto-generated
            "TREC-2020-doc-42",  # benchmark
            "clueweb09-en0000-00",  # IR benchmark
            "custom_benchmark_id",  # simple custom
            "my-doc.v2",  # dots without traversal
            "a",  # minimal valid
        ],
    )
    def test_accepts_valid_custom_ids(self, doc_id):
        assert self._is_safe(doc_id) is True

    @pytest.mark.parametrize(
        "doc_id,reason",
        [
            ("../escape", "parent traversal"),
            ("foo/../bar", "embedded traversal"),
            ("foo/bar", "forward slash"),
            ("foo\\bar", "backslash"),
            ("foo\x00bar", "null byte"),
            ("", "empty string"),
            ("x" * 513, "exceeds max length"),
        ],
    )
    def test_rejects_unsafe_ids(self, doc_id, reason):
        assert self._is_safe(doc_id) is False, f"Should reject: {reason}"

    def test_rejects_none(self):
        assert self._is_safe(None) is False

    def test_rejects_non_string(self):
        assert self._is_safe(123) is False


# =============================================================================
# Registry Tests
# =============================================================================


class TestStorageRegistry:
    """Tests for StorageRegistry."""

    def test_singleton(self):
        r1 = StorageRegistry()
        r2 = StorageRegistry()
        assert r1 is r2

    def test_local_registered_by_default(self):
        registry = StorageRegistry()
        assert registry.has_backend("local")
        assert "local" in registry.list_backends()

    def test_create_local(self, tmp_dir):
        registry = StorageRegistry()
        backend = registry.create("local", singleton=False, base_dir=tmp_dir)
        assert isinstance(backend, LocalStorageBackend)
        assert backend.backend_name == "local"

    def test_create_unknown_raises(self):
        registry = StorageRegistry()
        with pytest.raises(ValueError, match="Unknown storage backend"):
            registry.create("nonexistent_backend")

    def test_register_custom_backend(self, tmp_dir):
        """Test bringing your own storage backend."""

        class MockStorage(StorageBackend):
            @property
            def backend_name(self):
                return "mock"

            @property
            def capabilities(self):
                return StorageCapabilities()

            def store(self, source, document_id, filename, **kw):
                return StorageFileInfo(uri=f"mock://{document_id}/{filename}", size=0)

            def retrieve(self, uri):
                return io.BytesIO(b"mock data")

            def delete(self, uri):
                return True

            def exists(self, uri):
                return True

            def get_size(self, uri):
                return 0

        registry = StorageRegistry()
        registry.register("mock_test", MockStorage, override=True)
        assert registry.has_backend("mock_test")

        backend = registry.create("mock_test", singleton=False)
        assert backend.backend_name == "mock"

        info = backend.store(io.BytesIO(b"test"), "doc1", "file.txt")
        assert info.uri == "mock://doc1/file.txt"

        # Cleanup
        registry.unregister("mock_test")

    def test_register_type_check(self):
        registry = StorageRegistry()
        with pytest.raises(TypeError, match="must be a subclass"):
            registry.register("bad", str)  # type: ignore

    def test_register_duplicate_raises(self):
        registry = StorageRegistry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register("local", LocalStorageBackend)

    def test_register_duplicate_with_override(self):
        registry = StorageRegistry()
        registry.register("local", LocalStorageBackend, override=True)
        assert registry.has_backend("local")

    def test_unregister(self):
        registry = StorageRegistry()
        # Register a temp backend
        registry.register("temp_unreg", LocalStorageBackend, override=True)
        assert registry.has_backend("temp_unreg")
        result = registry.unregister("temp_unreg")
        assert result is True
        assert not registry.has_backend("temp_unreg")

    def test_unregister_nonexistent(self):
        registry = StorageRegistry()
        assert registry.unregister("doesnt_exist") is False

    def test_singleton_caching(self, tmp_dir):
        registry = StorageRegistry()
        b1 = registry.create("local", singleton=True, base_dir=tmp_dir)
        b2 = registry.create("local", singleton=True, base_dir=tmp_dir)
        assert b1 is b2

    def test_non_singleton(self, tmp_dir):
        registry = StorageRegistry()
        b1 = registry.create("local", singleton=False, base_dir=tmp_dir)
        b2 = registry.create("local", singleton=False, base_dir=tmp_dir)
        assert b1 is not b2

    def test_clear_instances(self, tmp_dir):
        registry = StorageRegistry()
        b1 = registry.create("local", singleton=True, base_dir=tmp_dir)
        registry.clear_instances()
        b2 = registry.create("local", singleton=True, base_dir=tmp_dir)
        assert b1 is not b2


# =============================================================================
# Module-Level Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_available_backends(self):
        backends = get_available_storage_backends()
        assert "local" in backends

    def test_get_storage_backend_default(self):
        backend = get_storage_backend(singleton=False)
        assert isinstance(backend, StorageBackend)
        assert backend.backend_name == "local"

    def test_get_storage_backend_by_name(self, tmp_dir):
        backend = get_storage_backend("local", singleton=False, base_dir=tmp_dir)
        assert backend.backend_name == "local"


# =============================================================================
# Base Class ensure_local() Tests (cloud simulation)
# =============================================================================


class TestEnsureLocal:
    """Test the ensure_local() bridge for cloud backends."""

    def test_base_ensure_local_downloads_to_temp(self):
        """Test that base ensure_local downloads to a temp file."""

        class FakeCloudStorage(StorageBackend):
            @property
            def backend_name(self):
                return "fake_cloud"

            @property
            def capabilities(self):
                return StorageCapabilities()

            def store(self, source, document_id, filename, **kw):
                return StorageFileInfo(uri=f"cloud://{document_id}/{filename}", size=0)

            def retrieve(self, uri):
                return io.BytesIO(b"cloud file content")

            def delete(self, uri):
                return True

            def exists(self, uri):
                return True

            def get_size(self, uri):
                return 18

        storage = FakeCloudStorage()

        # get_local_path returns None (default) → ensure_local downloads to temp
        assert storage.get_local_path("cloud://doc1/file.pdf") is None

        local_path = storage.ensure_local("cloud://doc1/file.pdf")
        try:
            assert os.path.exists(local_path)
            assert local_path.endswith(".pdf")
            with open(local_path, "rb") as f:
                assert f.read() == b"cloud file content"
        finally:
            os.unlink(local_path)


# =============================================================================
# Base Class _validate_store_inputs() Tests
# =============================================================================


class TestValidateStoreInputs:
    """Test StorageBackend._validate_store_inputs (base class guard)."""

    # -- document_id --
    @pytest.mark.parametrize(
        "doc_id",
        [
            "../escape",
            "foo/../bar",
            "foo/bar",
            "foo\\bar",
            "foo\x00bar",
            "",
            "x" * 513,
        ],
    )
    def test_rejects_unsafe_document_ids(self, doc_id):
        with pytest.raises(StorageError):
            StorageBackend._validate_store_inputs(doc_id, "safe.txt")

    def test_rejects_none_document_id(self):
        with pytest.raises(StorageError):
            StorageBackend._validate_store_inputs(None, "safe.txt")

    # -- filename --
    @pytest.mark.parametrize(
        "fname",
        [
            "../../../etc/passwd",
            "foo/bar.txt",
            "foo\\bar.txt",
            "foo\x00bar.txt",
            "",
        ],
    )
    def test_rejects_unsafe_filenames(self, fname):
        with pytest.raises(StorageError):
            StorageBackend._validate_store_inputs("safe_doc_id", fname)

    def test_rejects_none_filename(self):
        with pytest.raises(StorageError):
            StorageBackend._validate_store_inputs("safe_doc_id", None)

    # -- valid inputs --
    @pytest.mark.parametrize(
        "doc_id,fname",
        [
            ("doc_abc123def456789a", "report.pdf"),
            ("TREC-2020-doc-42", "file.txt"),
            ("custom_id", "my-doc.v2.pdf"),
            ("a" * 512, "x"),
        ],
    )
    def test_accepts_safe_inputs(self, doc_id, fname):
        StorageBackend._validate_store_inputs(doc_id, fname)  # no raise


# =============================================================================
# Backend-Invocation Tests (verify store() rejects before I/O)
# =============================================================================


class TestBackendStoreRejectsBeforeIO:
    """Verify each backend's store() rejects invalid inputs before any I/O."""

    def test_local_rejects_traversal_document_id(self, local_storage):
        with pytest.raises(StorageError):
            local_storage.store(io.BytesIO(b"x"), "../escape", "f.txt")

    def test_local_rejects_traversal_filename(self, local_storage):
        with pytest.raises(StorageError):
            local_storage.store(io.BytesIO(b"x"), "safe_id", "../etc/passwd")

    def test_s3_rejects_traversal_document_id(self):
        """S3StorageBackend.store() rejects before upload_fileobj."""
        try:
            from lasagna.storage.s3_backend import S3StorageBackend
        except ImportError:
            pytest.skip("boto3 not installed")
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        backend = S3StorageBackend.__new__(S3StorageBackend)
        backend._client = mock_client
        backend._bucket = "test"
        backend._prefix = "pfx"
        with pytest.raises(StorageError):
            backend.store(io.BytesIO(b"x"), "../escape", "f.txt")
        mock_client.upload_fileobj.assert_not_called()

    def test_s3_rejects_traversal_filename(self):
        """S3StorageBackend.store() rejects before upload_fileobj."""
        try:
            from lasagna.storage.s3_backend import S3StorageBackend
        except ImportError:
            pytest.skip("boto3 not installed")
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        backend = S3StorageBackend.__new__(S3StorageBackend)
        backend._client = mock_client
        backend._bucket = "test"
        backend._prefix = "pfx"
        with pytest.raises(StorageError):
            backend.store(io.BytesIO(b"x"), "safe_id", "../etc/passwd")
        mock_client.upload_fileobj.assert_not_called()

    def test_fsspec_rejects_traversal_document_id(self):
        """FsspecStorageBackend.store() rejects before fs.open."""
        try:
            from lasagna.storage.fsspec_backend import (
                FsspecStorageBackend,
            )
        except ImportError:
            pytest.skip("fsspec not installed")
        from unittest.mock import MagicMock

        mock_fs = MagicMock()
        backend = FsspecStorageBackend.__new__(FsspecStorageBackend)
        backend._fs = mock_fs
        backend._protocol = "memory"
        backend._base_path = "test"
        with pytest.raises(StorageError):
            backend.store(io.BytesIO(b"x"), "../escape", "f.txt")
        mock_fs.open.assert_not_called()
        mock_fs.mkdirs.assert_not_called()

    def test_fsspec_rejects_traversal_filename(self):
        """FsspecStorageBackend.store() rejects before fs.open."""
        try:
            from lasagna.storage.fsspec_backend import (
                FsspecStorageBackend,
            )
        except ImportError:
            pytest.skip("fsspec not installed")
        from unittest.mock import MagicMock

        mock_fs = MagicMock()
        backend = FsspecStorageBackend.__new__(FsspecStorageBackend)
        backend._fs = mock_fs
        backend._protocol = "memory"
        backend._base_path = "test"
        with pytest.raises(StorageError):
            backend.store(io.BytesIO(b"x"), "safe_id", "../etc/passwd")
        mock_fs.open.assert_not_called()
        mock_fs.mkdirs.assert_not_called()


# =============================================================================
# Validation Parity Tests (API layer and storage layer agree)
# =============================================================================


class TestValidationParity:
    """Verify API-level and storage-level validation agree on all vectors."""

    _api_is_safe = staticmethod(_get_is_safe_document_id())

    @pytest.mark.parametrize(
        "doc_id",
        [
            # Valid
            "doc_abc123def456789a",
            "TREC-2020-doc-42",
            "custom_id",
            "my-doc.v2",
            "a",
            # Invalid
            "../escape",
            "foo/../bar",
            "foo/bar",
            "foo\\bar",
            "foo\x00bar",
            "",
            "x" * 513,
        ],
    )
    def test_api_and_storage_agree(self, doc_id):
        api_ok = self._api_is_safe(doc_id)
        try:
            StorageBackend._validate_store_inputs(doc_id, "safe.txt")
            storage_ok = True
        except StorageError:
            storage_ok = False
        assert api_ok == storage_ok, (
            f"Parity violation for {doc_id!r}: api={api_ok}, storage={storage_ok}"
        )
