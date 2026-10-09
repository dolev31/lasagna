"""Agentic RAG Integration Tests with PgVector Backend.

Tests the full agentic RAG pipeline:
  1. Connect to pgvector
  2. Index sample documents
  3. Run agentic retrieval with a real LLM
  4. Validate citations, metrics, and result structure

Prerequisites:
    - PostgreSQL with pgvector running on localhost:5432
    - LLM API key configured in .env (OPENAI_API_KEY)

When prerequisites are missing the tests are SKIPPED with a loud ALERT
log so CI dashboards surface the gap without failing the build.
Run with:
    # pragma: allowlist nextline secret
    PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag \
    pytest tests/live/test_agentic_rag_integration.py -v

This module lived in ``tests/`` alongside the hermetic suite and evaluated
``_HAS_LLM = _llm_available()`` at IMPORT time, which calls
``llm.invoke("Say OK")``. So ``pytest tests/ --collect-only`` — collection, not
even execution — issued a billable request, and the module's own ``.env``
loading walked up to the repo root to find credentials for it.

The probe was also broken in a way that hid itself: ``provider.get_chat_model``
takes ``model_settings`` as a required positional, so the primary path always
raised ``TypeError``, ``except Exception: pass`` swallowed it, and the gate
silently degraded to "is OPENAI_API_KEY set". Fixed here, but the module belongs
behind ``LASAGNA_RUN_LIVE_TESTS`` regardless.
"""

import logging
import os
import uuid

import pytest

# Load .env BEFORE any imports that trigger provider auto-configuration.
# The knowledge-providers module runs _auto_configure_from_env() once, and if
# OPENAI_API_KEY is not yet in os.environ at that point the LLM provider stays
# None forever.  Calling load_dotenv() early ensures API keys are available.
try:
    from pathlib import Path
    from dotenv import load_dotenv

    # Walk up from this file to find the project root .env
    _env_file = Path(__file__).resolve()
    for _parent in _env_file.parents:
        _candidate = _parent / ".env"
        if _candidate.exists():
            load_dotenv(_candidate)
            break
    else:
        load_dotenv()  # fallback: CWD
except ImportError:
    pass

# ---------------------------------------------------------------------------
# ALERT logger — always visible, even when tests are skipped
# ---------------------------------------------------------------------------
_alert_logger = logging.getLogger("rag.test.alert")
_alert_logger.setLevel(logging.WARNING)
if not _alert_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("\n%(levelname)s | %(name)s | %(message)s\n"))
    _alert_logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DEFAULT_PGVECTOR_DSN = "postgresql://rag:rag@localhost:5432/rag"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------


def _pgvector_available() -> bool:
    """Check if pgvector backend can connect."""
    try:
        from lasagna.vector_backends.pgvector_backend import (
            PgVectorBackend,
            PGVECTOR_AVAILABLE,
        )

        if not PGVECTOR_AVAILABLE:
            return False
        dsn = os.environ.get("PGVECTOR_DSN", DEFAULT_PGVECTOR_DSN)
        backend = PgVectorBackend(dsn=dsn, pool_min=1, pool_max=2)
        backend.connect()
        backend.disconnect()
        return True
    except Exception:
        return False


def _llm_available() -> bool:
    """Check if an LLM is reachable via the knowledge provider stack.

    Uses get_llm_provider() — the same path AgenticRAG.create() uses — so
    the guard matches what the tests actually exercise.  Falls back to a
    direct ChatOpenAI ping when the provider stack is not configured.

    Note: .env is already loaded at module level (lines 31-44).
    """
    try:
        # Primary check: the provider stack that AgenticRAG uses
        try:
            from lasagna.providers import get_llm_provider

            provider = get_llm_provider()
            llm = provider.get_chat_model({}, temperature=0, max_tokens=10)
            resp = llm.invoke("Say OK")
            return bool(resp.content)
        except Exception:
            pass

        # Fallback: direct OpenAI check
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return False

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
            max_tokens=10,
        )
        resp = llm.invoke("Say OK")
        return bool(resp.content)
    except Exception:
        return False


# Evaluate once at import time
_HAS_PGVECTOR = _pgvector_available()
_HAS_LLM = _llm_available()

if not _HAS_PGVECTOR:
    _alert_logger.warning(
        "ALERT: pgvector backend NOT available — "
        "agentic RAG integration tests SKIPPED. "
        "Start pgvector: cd deploy && VECTOR_BACKEND=pgvector make up"
    )

if not _HAS_LLM:
    _alert_logger.warning(
        "ALERT: LLM API NOT reachable — "
        "agentic RAG integration tests SKIPPED. "
        "Ensure OPENAI_API_KEY is set in .env and the endpoint is accessible."
    )

_SKIP_REASON = []
if not _HAS_PGVECTOR:
    _SKIP_REASON.append("pgvector not available")
if not _HAS_LLM:
    _SKIP_REASON.append("LLM not available")

skip_agentic = pytest.mark.skipif(
    bool(_SKIP_REASON),
    reason=" + ".join(_SKIP_REASON) if _SKIP_REASON else "",
)


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "chunk_id": "quantum_001",
        "document_id": "doc_aa00000000000001",
        "session_id": "test_agentic",
        "content": (
            "Quantum entanglement is a phenomenon in quantum mechanics where "
            "two particles become interconnected and the quantum state of one "
            "particle instantly influences the state of the other, regardless "
            "of the distance between them. Albert Einstein famously referred to "
            "this as 'spooky action at a distance'."
        ),
        "embedding": None,  # filled at runtime
        "metadata": {"page_number": 1, "chunk_index": 0, "source": "quantum_physics.pdf"},
    },
    {
        "chunk_id": "quantum_002",
        "document_id": "doc_aa00000000000001",
        "session_id": "test_agentic",
        "content": (
            "Entanglement is fundamental to quantum computing and quantum "
            "cryptography. When particles are entangled, measuring the spin "
            "of one particle immediately determines the spin of its partner. "
            "This property enables quantum teleportation and quantum key "
            "distribution protocols like BB84 and E91."
        ),
        "embedding": None,
        "metadata": {"page_number": 2, "chunk_index": 1, "source": "quantum_physics.pdf"},
    },
    {
        "chunk_id": "ml_001",
        "document_id": "doc_bb00000000000002",
        "session_id": "test_agentic",
        "content": (
            "Supervised learning algorithms learn from labeled training data. "
            "Common approaches include decision trees, support vector machines, "
            "and neural networks. The goal is to learn a mapping from inputs "
            "to outputs based on example input-output pairs."
        ),
        "embedding": None,
        "metadata": {"page_number": 1, "chunk_index": 0, "source": "ml_overview.pdf"},
    },
]

DOCUMENT_METADATA = [
    {
        "document_id": "doc_aa00000000000001",
        "session_id": "test_agentic",
        "filename": "quantum_physics.pdf",
        "file_path": "/tmp/test/quantum_physics.pdf",
        "file_type": ".pdf",
        "file_size": 1024,
    },
    {
        "document_id": "doc_bb00000000000002",
        "session_id": "test_agentic",
        "filename": "ml_overview.pdf",
        "file_path": "/tmp/test/ml_overview.pdf",
        "file_type": ".pdf",
        "file_size": 512,
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_embeddings():
    """Pre-computed embeddings for SAMPLE_DOCS (computed once per module).

    Both ``pgvector_backend`` and ``rag_backend_live`` need the same
    embeddings for the same documents, so we compute them once here.
    """
    from lasagna.infrastructure.embedding.manager import get_embedding_manager

    manager = get_embedding_manager()
    texts = [doc["content"] for doc in SAMPLE_DOCS]
    return manager.encode_documents_batch(texts, show_progress=False)


@pytest.fixture(scope="module")
def pgvector_backend(sample_embeddings):
    """Create a connected pgvector backend with a test collection."""
    from lasagna.vector_backends.pgvector_backend import PgVectorBackend
    from lasagna.vector_backends.base import (
        CollectionSchema,
        FieldSchema,
    )

    dsn = os.environ.get("PGVECTOR_DSN", DEFAULT_PGVECTOR_DSN)
    backend = PgVectorBackend(dsn=dsn, pool_min=1, pool_max=2)
    backend.connect()

    collection = f"test_agentic_{uuid.uuid4().hex[:8]}"

    # Derive dimension from pre-computed embeddings (single source of truth)
    embeddings = sample_embeddings
    dim = embeddings[0].shape[0]

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=128),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=dim),
            FieldSchema(name="content", dtype="VARCHAR", max_length=8000),
            FieldSchema(name="metadata", dtype="JSON"),
        ]
    )
    backend.create_collection(collection, schema)

    records = []
    for doc, emb in zip(SAMPLE_DOCS, embeddings):
        rec = dict(doc)
        rec["embedding"] = emb.tolist()
        records.append(rec)

    backend.insert(collection, records)

    yield backend, collection

    # Cleanup
    try:
        backend.drop_collection(collection)
    except Exception:
        pass
    backend.disconnect()


@pytest.fixture(scope="module")
def pgvector_env():
    """Configure env vars and rag_config for pgvector backend.

    Sets RAG_VECTOR_BACKEND and PGVECTOR_DSN in the environment, then
    uses the public ``set_override()`` API to notify the rag_config
    singleton (which was created before these env vars existed).

    Restores everything on teardown.
    """
    from lasagna.core.config import rag_config

    # Save original env vars
    _saved_env = {}
    for key in ("RAG_VECTOR_BACKEND", "PGVECTOR_DSN"):
        _saved_env[key] = os.environ.get(key)

    dsn = os.environ.get("PGVECTOR_DSN", DEFAULT_PGVECTOR_DSN)

    # Set env vars + public override
    os.environ["RAG_VECTOR_BACKEND"] = "pgvector"
    os.environ["PGVECTOR_DSN"] = dsn
    rag_config.set_override("vector_backend", "pgvector")

    yield

    # Restore
    rag_config.clear_overrides()
    for key, val in _saved_env.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


@pytest.fixture(scope="module")
def rag_backend_instance(pgvector_env):
    """RAGBackend singleton initialized with pgvector.

    Resets the singleton, creates a fresh RAGBackend with the pgvector
    backend, and sets the session-level vector_backend override so that
    per-session backend resolution also returns pgvector.
    """
    from lasagna.api.rag_backend import RAGBackend, reset_instance

    reset_instance()
    backend = RAGBackend()

    # Set the session-level vector_backend override so that
    # resolve_backend("test_agentic") returns the pgvector backend.
    # Uses the public set_vector_backend() API which writes to
    # session_overrides (highest priority in get_resolved_config()).
    try:
        from lasagna.core.session_config import (
            get_knowledge_config_manager,
        )

        config_mgr = get_knowledge_config_manager()
        config_mgr.set_vector_backend("pgvector", session_id="test_agentic")
    except Exception as exc:
        logging.getLogger(__name__).warning(f"Could not set session vector_backend: {exc}")

    yield backend

    # Clear session overrides before resetting the singleton
    try:
        from lasagna.core.session_config import (
            get_knowledge_config_manager,
        )

        config_mgr = get_knowledge_config_manager()
        config_mgr.reset_config("active", "test_agentic")
    except Exception:
        pass
    reset_instance()


@pytest.fixture(scope="module")
def rag_backend_live(rag_backend_instance, sample_embeddings):
    """Seed test documents into pgvector and SQLite, clean up on teardown.

    Inserts SAMPLE_DOCS into the pgvector collection and DOCUMENT_METADATA
    into the SQLite rag_documents table so the full search pipeline
    (vector search → metadata lookup → citation extraction) works.
    """
    from lasagna.core.config import rag_config
    from lasagna.vector_backends.filters import FilterExpression

    backend = rag_backend_instance
    collection_name = rag_config.collection_name
    vector_backend = backend._vector_backend

    # --- Seed pgvector ---
    embeddings = sample_embeddings

    records = []
    for doc, emb in zip(SAMPLE_DOCS, embeddings):
        records.append(
            {
                "chunk_id": doc["chunk_id"],
                "document_id": doc["document_id"],
                "session_id": doc["session_id"],
                "content": doc["content"],
                "embedding": emb.tolist(),
                "metadata": doc["metadata"],
            }
        )
    vector_backend.insert(collection_name, records)

    # --- Seed SQLite ---
    from lasagna.core.database import get_rag_database

    rag_db = get_rag_database()
    with rag_db.connection() as conn:
        cursor = conn.cursor()
        for doc_meta in DOCUMENT_METADATA:
            cursor.execute(
                "INSERT OR REPLACE INTO rag_documents "
                "(document_id, session_id, filename, file_path, "
                "file_type, file_size, status, chunks_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_meta["document_id"],
                    doc_meta["session_id"],
                    doc_meta["filename"],
                    doc_meta["file_path"],
                    doc_meta["file_type"],
                    doc_meta["file_size"],
                    "ready",
                    len([d for d in SAMPLE_DOCS if d["document_id"] == doc_meta["document_id"]]),
                ),
            )
        conn.commit()

    yield backend

    # --- Cleanup pgvector ---
    try:
        for doc_meta in DOCUMENT_METADATA:
            vector_backend.delete(
                collection_name,
                FilterExpression.eq("document_id", doc_meta["document_id"]),
            )
    except Exception:
        logging.getLogger(__name__).warning("Cleanup: pgvector delete failed", exc_info=True)

    # --- Cleanup SQLite ---
    try:
        rag_db = get_rag_database()
        with rag_db.connection() as conn:
            cursor = conn.cursor()
            for doc_meta in DOCUMENT_METADATA:
                cursor.execute(
                    "DELETE FROM rag_documents WHERE document_id = ?",
                    (doc_meta["document_id"],),
                )
            conn.commit()
    except Exception:
        logging.getLogger(__name__).warning("Cleanup: SQLite delete failed", exc_info=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_agentic
@pytest.mark.integration
class TestAgenticRAGWithPgVector:
    """End-to-end agentic RAG tests against a real pgvector + LLM stack."""

    def test_basic_vector_search(self, pgvector_backend):
        """Sanity check: vector search returns results before testing agentic."""
        backend, collection = pgvector_backend

        from lasagna.infrastructure.embedding.manager import get_embedding_manager

        embedding_manager = get_embedding_manager()

        query_emb = embedding_manager.encode_query("quantum entanglement")
        results = backend.search(
            collection_name=collection,
            query_embedding=query_emb.tolist(),
            limit=3,
        )

        assert len(results) > 0
        # Top result should be about quantum
        assert "quantum" in results[0].entity["content"].lower()

    def test_hybrid_search(self, pgvector_backend):
        """Hybrid search (dense + sparse) returns results."""
        backend, collection = pgvector_backend

        from lasagna.infrastructure.embedding.manager import get_embedding_manager

        embedding_manager = get_embedding_manager()

        query_emb = embedding_manager.encode_query("quantum computing protocols")
        results = backend.hybrid_search(
            collection_name=collection,
            query_text="quantum computing protocols",
            query_embedding=query_emb.tolist(),
            limit=3,
        )

        assert len(results) > 0

    def test_filtered_search(self, pgvector_backend):
        """Filtered search by session_id works with pgvector."""
        backend, collection = pgvector_backend

        from lasagna.vector_backends.filters import FilterExpression
        from lasagna.infrastructure.embedding.manager import get_embedding_manager

        embedding_manager = get_embedding_manager()

        query_emb = embedding_manager.encode_query("machine learning")
        results = backend.search(
            collection_name=collection,
            query_embedding=query_emb.tolist(),
            filter_expr=FilterExpression.eq("session_id", "test_agentic"),
            limit=3,
        )

        assert len(results) > 0
        for r in results:
            assert r.entity["session_id"] == "test_agentic"

    def test_pgvector_registered_in_backend_registry(self):
        """pgvector is discoverable via get_available_backends()."""
        from lasagna.vector_backends.registry import get_available_backends

        backends = get_available_backends()
        assert "pgvector" in backends, f"pgvector not registered! Available: {backends}"

    def test_pgvector_get_backend_factory(self):
        """get_backend('pgvector') returns a PgVectorBackend instance."""
        from lasagna.vector_backends.registry import get_backend
        from lasagna.vector_backends.pgvector_backend import PgVectorBackend

        backend = get_backend("pgvector", singleton=False)
        assert isinstance(backend, PgVectorBackend)


@skip_agentic
@pytest.mark.integration
class TestAgenticRAGPipeline:
    """Test the agentic RAG pipeline components that require LLM."""

    def test_llm_provider_creates_model(self):
        """LLM provider can create a chat model from .env config."""
        # Reset providers so auto-configuration picks up env vars set by load_dotenv()
        from lasagna.providers import reset_providers, get_llm_provider

        reset_providers()
        provider = get_llm_provider()

        # Use MODEL_NAME from env to avoid model access restrictions
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        llm = provider.get_chat_model({"model_name": model_name})

        # Verify it can produce a response
        from langchain_core.messages import HumanMessage

        resp = llm.invoke([HumanMessage(content="Say hello")])
        assert resp.content and len(resp.content) > 0

    def test_agentic_rag_constructor(self):
        """AgenticRAG constructor produces a configured instance."""
        from lasagna.agentic.agentic_rag import AgenticRAG
        from langchain_openai import ChatOpenAI

        # Build LLM directly to avoid provider model name mismatches
        llm = ChatOpenAI(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
            max_tokens=10,
        )
        agent = AgenticRAG(llm=llm, max_iterations=2, enable_final_search=False)
        assert agent is not None
        assert agent.max_iterations == 2
        assert agent.graph is not None  # LangGraph agent built

    def test_citation_manager_basic(self):
        """CitationManager can create citations from search results."""
        from lasagna.agentic.citation_manager import CitationManager

        manager = CitationManager()

        mock_results = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "document_name": "test.pdf",
                "content": "Quantum entanglement connects particles across distances.",
                "score": 0.9,
                "chunk_index": 0,
                "metadata": {"page_number": 1},
            },
        ]

        citations = manager.create_citations(
            search_results=mock_results,
            query="quantum entanglement",
            session_id="test_session",
        )

        assert len(citations) == 1
        assert citations[0].document_name == "test.pdf"
        assert citations[0].score == 0.9


# ---------------------------------------------------------------------------
# Full agentic retrieval tests
# ---------------------------------------------------------------------------


@skip_agentic
@pytest.mark.integration
class TestAgenticRAGFullRetrieval:
    """Full agentic retrieval test — real RAGBackend + pgvector + LLM.

    Exercises the complete multi-hop agentic pipeline with NO mocks:
      Real LLM → Real AgenticRAG agent → Real RAGBackend/SearchOrchestrator
      → Real embedding model → Real pgvector search → Real reranking
      → Real SQLite metadata lookup → Real citation extraction
    """

    @pytest.mark.asyncio
    async def test_full_agentic_retrieve(self, rag_backend_live):
        """Full end-to-end: AgenticRAG.retrieve() with real LLM + pgvector."""
        from lasagna.agentic.agentic_rag import AgenticRAG
        from langchain_openai import ChatOpenAI

        # Build LLM directly from env vars (same as _llm_available check)
        # to avoid provider system model name mismatches.
        llm = ChatOpenAI(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=400,
        )
        agent = AgenticRAG(llm=llm, max_iterations=2, enable_final_search=True)
        result = await agent.retrieve(
            question="What is quantum entanglement and how does it relate to quantum computing?",
            session_id="test_agentic",
            top_k=3,
        )

        # 1. Formatted context is non-empty and structured
        assert result.formatted_context, "Should produce formatted context"
        assert "<knowledge_base>" in result.formatted_context

        # 2. Chunks contain quantum-related content
        assert len(result.chunks) > 0, "Should retrieve chunks"
        chunk_text = " ".join(c["content"] for c in result.chunks).lower()
        assert "quantum" in chunk_text, "Should find quantum content"

        # 3. Citations have correct structure and reference real documents
        assert len(result.citations) > 0, "Should have citations"
        for citation in result.citations:
            assert "index" in citation
            assert "document_name" in citation
            assert "chunk_id" in citation
            assert "score" in citation
            assert citation["document_name"] in ("quantum_physics.pdf", "ml_overview.pdf"), (
                f"Unexpected document: {citation['document_name']}"
            )

        # 4. Agent produced a summary
        assert result.agent_summary, "Agent should synthesize findings"

        # 5. Metrics are populated
        assert result.metrics is not None
        assert result.metrics.hop_count >= 1
        assert result.metrics.total_latency_ms > 0
        assert result.search_count >= 1

        # 6. Hops tracked
        assert len(result.hops) >= 1
        for hop in result.hops:
            assert hop.query_used, "Each hop should have a query"
            assert hop.search_latency_ms > 0
