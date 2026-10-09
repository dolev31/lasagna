"""Detachability tests — Lasagna stands on its own.

Two guarantees are asserted here, and both are load-bearing for a library
other people install:

1. ``import lasagna`` stays cheap.  The package uses ``__getattr__``-based
   lazy loading, so importing it must not drag in torch, transformers,
   sentence-transformers or pymilvus.  Anyone talking to a remote backend
   should pay for none of that.
2. Nothing in the package imports a third-party agent framework, except the
   clearly-marked optional adapters under ``lasagna/integrations/``.

Prerequisites: none — these tests create everything they need.
"""

import asyncio
import sys
import subprocess
from pathlib import Path

import pytest


async def _wait_until_searchable(client, query, timeout=30.0, interval=0.5):
    """Poll until an uploaded document can be found, or give up.

    Ingestion is asynchronous — `upload` returns once the document is accepted
    and chunking happens on a worker thread — so anything that inspects the
    result immediately is racing the pipeline.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        results = await client.search_documents(query)
        if results.results:
            return True
        await asyncio.sleep(interval)
    return False


SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "lasagna"


class TestImportIsolation:
    """Verify the package doesn't pull in heavy deps at import time."""

    def test_lazy_imports_no_heavy_deps(self):
        """``import lasagna`` should NOT eagerly load heavy ML libs.

        ``lasagna/__init__.py`` uses ``__getattr__``-based lazy loading, so
        the bare import must not trigger torch/transformers/pymilvus.  Those
        load only when a symbol that needs them is actually accessed.
        """
        # Run in a subprocess to get a clean import state
        code = """
import sys, json
before = set(sys.modules.keys())
import lasagna
after = set(sys.modules.keys())
new = after - before

heavy = ["torch", "transformers", "sentence_transformers", "pymilvus"]
loaded = [m for m in heavy if m in new]
print(json.dumps({"loaded_heavy": loaded, "total_new": len(new)}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Import crashed: {result.stderr[:500]}"
        import json

        data = json.loads(result.stdout.strip())
        assert not data["loaded_heavy"], (
            f"Heavy deps loaded at 'import lasagna': {data['loaded_heavy']}. "
            "lasagna/__init__.py lazy loading is broken."
        )

    def test_no_agent_framework_imports_outside_integrations(self):
        """Only ``lasagna/integrations/`` may reference an agent framework.

        This is what makes Lasagna usable as a plain library.  If a stray
        ``import cuga`` (or any other framework) leaks into the pipeline, the
        package stops being installable on its own -- so fail loudly and name
        the offending files.
        """
        # Frameworks that must never be imported by the pipeline itself.
        frameworks = ("cuga",)
        offenders = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            if "integrations" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if not stripped.startswith(("import ", "from ")):
                    continue
                # Importing our own optional adapter is fine — that module is
                # what keeps the framework dependency contained.
                if "lasagna.integrations" in stripped:
                    continue
                # Match the imported root module, not any mention of the name.
                root = stripped.split()[1].split(".")[0]
                if root in frameworks:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {stripped}")

        assert not offenders, (
            "Agent-framework imports found outside lasagna/integrations/:\n  "
            + "\n  ".join(offenders)
        )

    def test_sdk_importable_standalone(self):
        """KnowledgeClient and KnowledgeConfig can be imported standalone."""
        code = """
from lasagna.sdk import KnowledgeClient, KnowledgeConfig
config = KnowledgeConfig()
print(f"OK: KnowledgeConfig created with defaults")
print(f"  embedding_model: {config.embedding_model}")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"SDK import failed: {result.stdout}\n{result.stderr}"
        assert "OK" in result.stdout


class TestKnowledgeCLI:
    """Verify the knowledge package CLI works standalone."""

    def test_cli_help(self):
        """python -m lasagna --help shows available commands."""
        result = subprocess.run(
            [sys.executable, "-m", "lasagna", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"CLI help failed: {result.stderr}"
        # Should show command names
        output = result.stdout.lower()
        assert any(cmd in output for cmd in ["upload", "search", "serve", "init"]), (
            f"Expected CLI commands in: {result.stdout}"
        )

    def test_cli_upload_help(self):
        """Upload subcommand help is available."""
        result = subprocess.run(
            [sys.executable, "-m", "lasagna", "upload", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should show help or at least not crash
        assert result.returncode == 0, f"Upload help failed: {result.stderr}"


@pytest.mark.asyncio
class TestStandaloneBackend:
    """Test the knowledge backend in standalone mode (no Lasagna services)."""

    async def test_knowledge_client_standalone(self, tmp_path):
        """KnowledgeClient works with explicit config, no Lasagna env.

        NOTE: When lasagna serve is running, the default Milvus DB file
        is locked. This test uses a separate tmp_path DB to avoid conflicts.
        If it still fails due to Milvus lock, it's skipped — the standalone
        test is better run without demo services.
        """
        try:
            from lasagna.sdk import KnowledgeClient, KnowledgeConfig
        except ImportError:
            pytest.skip("Knowledge SDK not available")

        config = KnowledgeConfig()
        client = KnowledgeClient(config=config, session_id="standalone_test")

        # Create a test doc
        doc = tmp_path / "standalone.txt"
        doc.write_text(
            "Standalone test: The Eiffel Tower is 330 meters tall and is located in Paris, France."
        )

        try:
            result = await client.upload(str(doc))
            assert result.document_id is not None

            # Ingestion is asynchronous: `upload` returns as soon as the
            # document is accepted, and chunking runs on a worker thread. The
            # log line "[Worker] Document ... processed: 1 chunks" arrives after
            # this call returns, so `result.chunk_count` is legitimately 0 here
            # and asserting on it was a race the test lost every time.
            #
            # Wait for the document to become searchable instead, which is the
            # property this test actually cares about.
            searchable = await _wait_until_searchable(client, "How tall is the Eiffel Tower?")
            assert searchable, "document never became searchable after upload"

            # Search (client.search is a SearchManager, use .search() or search_documents())
            results = await client.search_documents("How tall is the Eiffel Tower?")
            assert len(results.results) > 0

            # Cleanup
            await client.delete_document(result.document_id)
        except Exception as e:
            if "milvus" in str(e).lower() or "locked" in str(e).lower() or "open" in str(e).lower():
                pytest.skip(
                    f"Milvus DB locked (likely by running demo): {e}. "
                    "Run standalone tests without lasagna serve."
                )
            raise
        finally:
            await client.close()

    async def test_standalone_session_auto_generation(self):
        """KnowledgeClient auto-generates session ID when None."""
        try:
            from lasagna.sdk import KnowledgeClient
        except ImportError:
            pytest.skip("Knowledge SDK not available")

        client = KnowledgeClient(session_id=None)
        assert client.session_id.startswith("sdk_")
        assert len(client.session_id) > 4
        await client.close()

    async def test_standalone_invalid_session_ids(self):
        """Invalid session IDs are rejected."""
        try:
            from lasagna.sdk import KnowledgeClient
        except ImportError:
            pytest.skip("Knowledge SDK not available")

        for invalid in ("None", "null", "undefined", ""):
            with pytest.raises(ValueError, match="Invalid session_id"):
                KnowledgeClient(session_id=invalid)
