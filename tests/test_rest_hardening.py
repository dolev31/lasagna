"""The REST API refuses, reports and records what it used to let through silently.

What went wrong before, one class per fix:

* With ``RAG_REQUIRE_AUTH`` on, anyone who could reach ``POST /sessions`` minted
  tokens, and rate limits counted per session, so each new token was a fresh
  allowance.
* A route's 413 ran after Starlette had spooled the whole multipart body to
  disk, and a batch upload had no file-count or total-size limit.
* The Redis URL, password included, was logged at startup.
* A networked MCP server with auth off read any server-side path it was given.
* Log lines carried no request id; there was no JSON log format.
* The latency histogram had no status label, and an unmatched URL became its
  own label value; nothing counted LLM calls, ingests or fallbacks.
* Readiness did not ping anything, so a dead Redis or vector store read as ready,
  and a hung one hung the probe.
* HyDE, GraphRAG and hybrid fallbacks were invisible to the caller; a failed
  focused-document lookup searched the whole session; a failed native hybrid
  search returned no results; a vector-store error listed a document's chunks
  as an empty 200.
* The OpenAPI description said "Production-ready", and every search logged
  "SOTA search: ... hybrid" at INFO, also on profiles with hybrid search off.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from starlette.requests import Request

from lasagna.utils import degraded as _degraded
from lasagna.api import _search_orchestrator as orch_mod
from lasagna.api.rest import limits, session_token
from lasagna.api.rest.app import _rate_limit_key, create_app
from lasagna.api.rest.dependencies import get_backend
from lasagna.core.config import rag_config
from lasagna.utils.monitoring import METRICS

KEY = "k" * 40
OTHER_KEY = "o" * 40


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)


@pytest.fixture
def auth_off(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)


def _client(backend=None) -> TestClient:
    app = create_app()
    if backend is not None:
        app.dependency_overrides[get_backend] = lambda: backend
    return TestClient(app)


def _request(headers) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/documents",
            "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": ("203.0.113.7", 5000),
        }
    )


@pytest.fixture
def metrics(monkeypatch):
    """Every METRICS call made during the test, as (kind, name, tags)."""
    calls = []

    class _Recorder:
        def counter(self, name, value=1, tags=None):
            calls.append(("counter", name, dict(tags or {})))

        def histogram(self, name, value, tags=None):
            calls.append(("histogram", name, dict(tags or {})))

    monkeypatch.setattr(METRICS, "_backend", _Recorder())
    return calls


@pytest.fixture
def logs():
    records = []
    sink = logger.add(lambda m: records.append(m.record), level="DEBUG")
    yield records
    logger.remove(sink)


# ---------------------------------------------------------------------------
# Auth: minting a session needs an API key, and limits follow the key
# ---------------------------------------------------------------------------


class TestMintingASessionNeedsAKey:
    def test_without_a_key_it_is_refused_with_the_fix(self, auth_on, monkeypatch):
        monkeypatch.setenv("RAG_API_KEYS", KEY)
        r = _client().post("/api/v1/sessions", json={})
        assert r.status_code == 401
        assert "X-API-Key" in r.json()["detail"] and "RAG_API_KEYS" in r.json()["detail"]

    def test_a_wrong_key_is_refused(self, auth_on, monkeypatch):
        monkeypatch.setenv("RAG_API_KEYS", KEY)
        r = _client().post("/api/v1/sessions", json={}, headers={"X-API-Key": OTHER_KEY})
        assert r.status_code == 401

    @pytest.mark.parametrize("configured", ["", "change_me", "short-key"])
    def test_with_no_usable_key_configured_nothing_is_issued(
        self, auth_on, monkeypatch, configured
    ):
        monkeypatch.setenv("RAG_API_KEYS", configured)
        r = _client().post("/api/v1/sessions", json={}, headers={"X-API-Key": configured or KEY})
        assert r.status_code == 503
        assert "RAG_API_KEYS" in r.json()["detail"]

    def test_a_listed_key_mints_tokens_counted_against_the_key(self, auth_on, monkeypatch):
        monkeypatch.setenv("RAG_API_KEYS", f"{OTHER_KEY}, {KEY}")
        client = _client()
        tokens = [
            client.post("/api/v1/sessions", json={}, headers={"X-API-Key": KEY}).json()["token"]
            for _ in range(2)
        ]
        from lasagna.api.rest.auth_mode import principal_for_api_key

        principal = principal_for_api_key(KEY)
        assert principal and KEY not in principal
        keys = {_rate_limit_key(_request({"X-Session-Token": t})) for t in tokens}
        # Two sessions, one bucket: minting more sessions buys no more requests.
        assert keys == {f"prn:{principal}"}
        claims = session_token.decode_session_claims(tokens[0])
        assert claims["prn"] == principal

    def test_with_auth_off_nothing_changes(self, auth_off, monkeypatch):
        monkeypatch.delenv("RAG_API_KEYS", raising=False)
        assert _client().post("/api/v1/sessions", json={}).status_code == 201


# ---------------------------------------------------------------------------
# Bodies are limited before they are read
# ---------------------------------------------------------------------------


def _run_asgi(app, scope, body_chunks=None):
    sent = []
    received = []

    async def receive():
        if body_chunks is None:
            raise AssertionError("the body was read")
        chunk = body_chunks.pop(0) if body_chunks else b""
        received.append(chunk)
        return {"type": "http.request", "body": chunk, "more_body": bool(body_chunks)}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent, received


def _scope(path, headers=()):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": list(headers),
        "query_string": b"",
    }


class TestBodiesAreLimitedBeforeTheyAreRead:
    def test_a_declared_length_over_the_limit_is_refused_unread(self):
        async def inner(scope, receive, send):
            raise AssertionError("the route ran")

        too_big = rag_config.max_file_size + limits.MULTIPART_OVERHEAD + 1
        scope = _scope("/api/v1/documents", [(b"content-length", str(too_big).encode())])
        sent, _ = _run_asgi(limits.BodySizeLimitMiddleware(inner), scope, body_chunks=None)
        assert sent[0]["status"] == 413
        assert b"RAG_MAX_FILE_SIZE" in sent[1]["body"]

    def test_a_chunked_body_is_cut_off_at_the_limit(self):
        """No Content-Length: counted as it arrives, refused once over the limit."""
        seen = []

        async def inner(scope, receive, send):
            while True:
                message = await receive()
                seen.append(len(message["body"]))
                if not message["more_body"]:
                    break

        chunks = [b"x" * (256 * 1024)] * 8  # 2 MiB of JSON body, limit 1 MiB
        sent, received = _run_asgi(
            limits.BodySizeLimitMiddleware(inner), _scope("/api/v1/search"), list(chunks)
        )
        assert sent[0]["status"] == 413
        assert len(received) < len(chunks), "the whole body was read before refusing"

    def test_the_route_sees_413_not_its_own_parse_error(self, auth_off):
        def body():
            for _ in range(8):
                yield b"x" * (256 * 1024)

        r = _client(SimpleNamespace()).post(
            "/api/v1/search", content=body(), headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 413
        assert "1 MiB" in r.json()["detail"]

    def test_a_batch_with_too_many_files_is_refused(self, auth_off, monkeypatch):
        monkeypatch.setenv("RAG_MAX_BATCH_FILES", "2")
        files = [("files", (f"f{i}.txt", b"hello", "text/plain")) for i in range(3)]
        r = _client(SimpleNamespace()).post("/api/v1/documents/batch", files=files)
        assert r.status_code == 400
        assert "RAG_MAX_BATCH_FILES" in r.json()["detail"]

    def test_a_batch_over_its_total_size_is_refused(self, auth_off, monkeypatch):
        monkeypatch.setenv("RAG_MAX_BATCH_BYTES", "1000")
        files = [("files", (f"f{i}.txt", b"x" * 600_000, "text/plain")) for i in range(2)]
        r = _client(SimpleNamespace()).post("/api/v1/documents/batch", files=files)
        assert r.status_code == 413
        assert "RAG_MAX_BATCH_BYTES" in r.json()["detail"]

    @pytest.mark.parametrize(
        "name, value",
        [
            ("RAG_MAX_BATCH_FILES", "many"),
            ("RAG_MAX_BATCH_BYTES", "0"),
            ("RAG_SEARCH_TIMEOUT", "-1"),
        ],
    )
    def test_an_invalid_limit_stops_the_app_naming_it(self, monkeypatch, name, value):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            create_app()


# ---------------------------------------------------------------------------
# Credentials in URLs are not logged
# ---------------------------------------------------------------------------


class TestCredentialsAreNotLogged:
    @pytest.mark.parametrize(
        "url, shown",
        [
            ("redis://redis:6379", "redis://redis:6379"),
            ("redis://:hunter2@cache:6379/0", "redis://:***@cache:6379/0"),
            ("postgresql://rag:hunter2@db:5432/rag", "postgresql://rag:***@db:5432/rag"),
            ("redis://cache:6379/?password=hunter2&db=1", "redis://cache:6379/?password=***&db=1"),
        ],
    )
    def test_redact_url(self, url, shown):
        from lasagna.core.provenance import redact_url

        assert redact_url(url) == shown

    @pytest.fixture
    def fake_redis(self, monkeypatch):
        import sys

        client = SimpleNamespace(ping=lambda: True, register_script=lambda script: object())
        monkeypatch.setitem(
            sys.modules,
            "redis",
            SimpleNamespace(Redis=SimpleNamespace(from_url=lambda url, **kw: client)),
        )

    def test_the_rate_limiter_does_not_log_the_password(self, fake_redis, logs):
        from lasagna.core.rate_limiter import RedisRateLimiter

        RedisRateLimiter(redis_url="redis://:hunter2@cache:6379")
        text = " ".join(r["message"] for r in logs)
        assert "cache:6379" in text and "hunter2" not in text

    def test_the_quota_manager_does_not_log_the_password(self, fake_redis, logs):
        from lasagna.core.quota import RedisQuotaManager

        RedisQuotaManager(enabled=True, redis_url="redis://:hunter2@cache:6379")
        text = " ".join(r["message"] for r in logs)
        assert "cache:6379" in text and "hunter2" not in text

    def test_the_startup_summary_does_not_log_the_password(self, auth_off, monkeypatch, logs):
        import lasagna.api.rag_backend as rag_backend
        import lasagna.core.config as config_module

        monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: SimpleNamespace())
        monkeypatch.setattr(rag_backend, "close_backend", lambda: None)
        summary = SimpleNamespace(
            database_backend="sqlite",
            vector_backend="milvus",
            cache_backend="redis",
            redis_url="redis://:hunter2@cache:6379",
            embedding_model="m",
            embedding_dim=8,
        )
        monkeypatch.setattr(config_module, "rag_config", summary, raising=False)
        with TestClient(create_app()):
            pass
        text = " ".join(r["message"] for r in logs)
        assert "cache:6379" in text and "hunter2" not in text


# ---------------------------------------------------------------------------
# MCP: server-side paths are confined on every networked transport
# ---------------------------------------------------------------------------


class TestNetworkedMcpConfinesUploads:
    @pytest.fixture
    def sse_without_auth(self, monkeypatch):
        from lasagna.mcp import transport_guard

        monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
        previous = transport_guard.get_transport()
        transport_guard.set_transport("sse")
        yield transport_guard
        transport_guard.set_transport(previous)

    def test_no_root_means_no_server_side_path(self, sse_without_auth, monkeypatch, tmp_path):
        monkeypatch.delenv("RAG_MCP_UPLOAD_ROOT", raising=False)
        secret = tmp_path / "credentials.json"
        secret.write_text("{}")
        with pytest.raises(sse_without_auth.MCPAuthError, match="RAG_MCP_UPLOAD_ROOT"):
            sse_without_auth.guard_server_path(str(secret))

    def test_only_paths_inside_the_root_are_read(self, sse_without_auth, monkeypatch, tmp_path):
        root = tmp_path / "inbox"
        root.mkdir()
        (root / "a.md").write_text("ok")
        (tmp_path / "b.json").write_text("{}")
        monkeypatch.setenv("RAG_MCP_UPLOAD_ROOT", str(root))
        assert sse_without_auth.guard_server_path(str(root / "a.md")) == str(
            (root / "a.md").resolve()
        )
        with pytest.raises(sse_without_auth.MCPAuthError):
            sse_without_auth.guard_server_path(str(tmp_path / "b.json"))


# ---------------------------------------------------------------------------
# Request ids reach every log line; JSON logs
# ---------------------------------------------------------------------------


def _app_with_probe():
    app = create_app()

    @app.get("/probe")
    async def probe():
        logger.info("probe ran")
        return {"ok": True}

    return app


class TestRequestIdsReachTheLogs:
    def test_a_log_line_inside_a_request_carries_its_id(self, auth_off, logs):
        r = TestClient(_app_with_probe()).get("/probe", headers={"X-Request-ID": "trace-42"})
        assert r.headers["X-Request-ID"] == "trace-42"
        probe = [rec for rec in logs if rec["message"] == "probe ran"]
        assert probe and probe[0]["extra"].get("request_id") == "trace-42"

    def test_an_id_that_is_not_an_id_is_replaced(self, auth_off, logs):
        r = TestClient(_app_with_probe()).get("/probe", headers={"X-Request-ID": "a b <script>"})
        assert r.headers["X-Request-ID"] != "a b <script>"
        assert [rec["extra"]["request_id"] for rec in logs if rec["message"] == "probe ran"] == [
            r.headers["X-Request-ID"]
        ]

    def test_json_format_replaces_the_handlers_with_a_serialized_one(self, monkeypatch):
        import sys

        calls = []
        monkeypatch.setattr(
            limits,
            "logger",
            SimpleNamespace(
                remove=lambda: calls.append(("remove",)),
                add=lambda sink, **kw: calls.append(("add", sink, kw)),
            ),
        )
        monkeypatch.setenv("LASAGNA_LOG_FORMAT", "json")
        assert limits.apply_log_format("warning") is True
        assert calls[0] == ("remove",)
        _, sink, kwargs = calls[1]
        assert sink is sys.stderr
        assert kwargs["serialize"] is True and kwargs["level"] == "WARNING"
        assert kwargs["diagnose"] is False

    def test_a_json_line_logged_during_a_request_carries_its_id(self, auth_off):
        lines = []
        sink = logger.add(lines.append, serialize=True, level="INFO")
        try:
            TestClient(_app_with_probe()).get("/probe", headers={"X-Request-ID": "r-7"})
        finally:
            logger.remove(sink)
        records = [json.loads(line)["record"] for line in lines]
        probe = [r for r in records if r["message"] == "probe ran"]
        assert probe and probe[0]["extra"].get("request_id") == "r-7"

    def test_text_is_the_default_and_changes_nothing(self, monkeypatch):
        monkeypatch.delenv("LASAGNA_LOG_FORMAT", raising=False)
        assert limits.apply_log_format() is False

    def test_an_unknown_log_format_stops_the_app(self, monkeypatch):
        monkeypatch.setenv("LASAGNA_LOG_FORMAT", "yaml")
        with pytest.raises(ValueError, match="LASAGNA_LOG_FORMAT"):
            create_app()


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------


class TestMetrics:
    def _http(self, metrics):
        return [tags for kind, name, tags in metrics if name == "rag.http.request_duration_ms"]

    def test_the_latency_histogram_has_a_status_label(self, auth_off, metrics):
        TestClient(_app_with_probe()).get("/probe")
        assert {"method": "GET", "path": "/probe", "status": "200"} in self._http(metrics)

    def test_an_unmatched_url_is_one_label_value(self, auth_off, metrics):
        client = TestClient(_app_with_probe())
        for path in ("/wp-admin/a", "/.env", "/x/y/z"):
            client.get(path)
        assert {tags["path"] for tags in self._http(metrics)} == {"unmatched"}
        assert {tags["status"] for tags in self._http(metrics)} == {"404"}

    def test_a_fallback_is_counted_by_stage(self, metrics):
        _degraded.record("hyde", "no LLM")
        assert ("counter", "lasagna_degraded_fallbacks", {"stage": "hyde"}) in metrics

    def test_llm_calls_are_counted_by_outcome(self, metrics):
        from lasagna.utils.token_usage import usage_callback_handler

        handler = usage_callback_handler()
        handler.on_llm_end(SimpleNamespace(llm_output={}, generations=[]))
        handler.on_llm_error(RuntimeError("rate limited"))
        assert ("counter", "lasagna_llm_calls", {"outcome": "ok"}) in metrics
        assert ("counter", "lasagna_llm_calls", {"outcome": "error"}) in metrics

    def test_a_failed_ingest_is_counted(self, metrics, monkeypatch):
        import lasagna.api.rag_backend as rag_backend
        import lasagna.core.database as database
        from lasagna.infrastructure.tasks import document_tasks

        monkeypatch.setattr(
            rag_backend,
            "get_rag_backend",
            lambda: SimpleNamespace(_get_document_metadata=lambda doc_id: None),
        )
        monkeypatch.setattr(
            database,
            "get_rag_database",
            lambda: SimpleNamespace(update_document_status=lambda **kw: None),
        )
        with pytest.raises(ValueError):
            document_tasks._process_document_impl("d1", "s1", "/nowhere")
        assert ("counter", "lasagna_ingests", {"outcome": "error"}) in metrics


# ---------------------------------------------------------------------------
# Readiness pings each dependency, bounded
# ---------------------------------------------------------------------------


def _backend(has_collection=lambda name: True):
    return SimpleNamespace(
        embedding_manager=SimpleNamespace(model_name="m", dimension=8),
        get_vector_backend=lambda: SimpleNamespace(
            backend_name="milvus", has_collection=has_collection
        ),
    )


@pytest.fixture
def fast_pings(monkeypatch):
    from lasagna.api.rest.routes import health
    import lasagna.core.database as database

    monkeypatch.setattr(health, "PING_TIMEOUT_S", 0.3)
    monkeypatch.setattr(
        database, "get_rag_database", lambda: SimpleNamespace(get_documents=lambda sid: [])
    )
    return health


class TestReadiness:
    def test_a_hung_vector_store_is_named_and_does_not_hang_the_probe(self, auth_off, fast_pings):
        def hangs(name):
            time.sleep(1.5)
            return True

        # Timed inside the loop: a server's loop keeps running, while
        # asyncio.run (and TestClient) wait for the abandoned thread on exit.
        async def probe():
            started = time.perf_counter()
            response = await fast_pings.readiness(_backend(has_collection=hangs))
            return time.perf_counter() - started, response

        elapsed, response = asyncio.run(probe())
        body = json.loads(response.body)
        assert elapsed < 1.0
        assert response.status_code == 503
        assert body["failing"] == ["vector_backend"]
        assert "no answer within" in body["checks"]["vector_backend"]["detail"]

    def test_a_vector_store_error_is_named(self, auth_off, fast_pings):
        def broken(name):
            raise ConnectionError("connection refused")

        r = _client(_backend(has_collection=broken)).get("/api/v1/health/ready")
        assert r.status_code == 503
        assert r.json()["failing"] == ["vector_backend"]

    def test_redis_down_is_degraded_not_failing(self, auth_off, fast_pings, monkeypatch):
        """Redis is pinged when it is used. Its users fall back (the search cache
        and rate limits to process memory, uploads to local processing), so it no
        longer takes the instance out of rotation."""
        import lasagna.core.config as config_module

        monkeypatch.setattr(
            config_module,
            "rag_config",
            SimpleNamespace(
                collection_name="c",
                cache_backend="redis",
                redis_url="redis://127.0.0.1:1",  # nothing listens on port 1
                celery_enabled=False,
            ),
            raising=False,
        )
        r = _client(_backend()).get("/api/v1/health/ready")
        assert r.status_code == 200, r.json()
        assert r.json()["failing"] == [] and r.json()["degraded"] == ["redis"]
        assert r.json()["status"] == "degraded"
        assert r.json()["checks"]["redis"]["status"] == "degraded"

    def test_hung_pings_do_not_take_the_threads_searches_use(self, auth_off, fast_pings):
        """Searches run on the default executor; a hung ping must not hold it."""
        names = []

        def hangs(name):
            names.append(threading.current_thread().name)
            time.sleep(0.5)
            return True

        async def probe():
            return await fast_pings.readiness(_backend(has_collection=hangs))

        asyncio.run(probe())
        assert names and all(n.startswith("lasagna-ping") for n in names), names

    def test_everything_up_is_ready(self, auth_off, fast_pings):
        r = _client(_backend()).get("/api/v1/health/ready")
        assert r.status_code == 200, r.json()
        assert r.json()["failing"] == []
        assert {"vector_backend", "database", "embedding_model"} <= set(r.json()["checks"])


# ---------------------------------------------------------------------------
# Degraded paths are reported, or refused
# ---------------------------------------------------------------------------


class _MilvusClient:
    """A Milvus client with one stored chunk, enough for a full search()."""

    def describe_collection(self, name, **kw):
        return {
            "collection_name": name,
            "fields": [
                {"name": "chunk_id", "type": "VARCHAR", "is_primary": True, "params": {}},
                {"name": "session_id", "type": "VARCHAR", "params": {}},
                {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 3}},
            ],
        }

    def get_load_state(self, name, **kw):
        return {"state": "Loaded"}

    def search(self, **kw):
        entity = {"document_id": "d1", "content": "tides", "metadata": {}, "embedding": [1, 1, 1]}
        return [[{"id": "d1_chunk_0", "distance": 0.9, "entity": entity}]]


@pytest.fixture
def orchestrator(monkeypatch):
    from lasagna.vector_backends.milvus_backend import MilvusBackend

    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    # Where the session's documents are is read from the suite's database,
    # which other tests fill for these session ids; these tests name one stage.
    monkeypatch.setattr(orch_mod, "_note_documents_elsewhere", lambda session_id, config: None)
    backend = MilvusBackend(db_path="never-opened.db")
    backend._client = _MilvusClient()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = SimpleNamespace()
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {
            i: SimpleNamespace(document_id=i, filename=f"{i}.txt") for i in ids
        }
    )
    return o


PLAIN = dict(
    enable_hybrid_search=False,
    enable_reranking=False,
    enable_query_routing=False,
    enable_query_classification=False,
    enable_search_cache=False,
    enable_hyde=False,
    enable_mmr=False,
    enable_graph_retrieval=False,
    enable_rag_fusion=False,
    enable_query_expansion=False,
    enable_hierarchical=False,
)


class TestDegradedPathsAreVisible:
    def test_the_async_search_carries_the_stages_out_of_its_thread(self):
        o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)

        def search(**kwargs):
            _degraded.record("hybrid", "BM25 index empty")
            return []

        o.search = search

        async def run():
            with _degraded.collecting() as stages:
                await o.async_search(query="q", session_id="s")
            return stages

        assert asyncio.run(run()) == ["hybrid"]

    def test_the_rest_response_names_them(self, auth_off, monkeypatch):
        from lasagna.core import model_downloads

        monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"default": 1})

        async def async_search(**kwargs):
            _degraded.record("hyde", "no LLM")
            return []

        backend = SimpleNamespace(async_search=async_search, _search_orchestrator=None)
        r = _client(backend).post("/api/v1/search", json={"query": "tides"})
        assert r.status_code == 200
        assert r.json()["degraded_stages"] == ["hyde"]

    def test_a_failed_hyde_is_named_by_a_real_search(self, orchestrator):
        def no_llm(*a, **k):
            raise RuntimeError("No LLMProvider configured")

        orchestrator.query_enhancer = SimpleNamespace(
            get_hyde_embedding=no_llm,
            hyde=SimpleNamespace(generate_hypothetical_document_sync=no_llm),
        )
        with _degraded.collecting() as stages:
            orchestrator.search(
                "q",
                "s1",
                top_k=1,
                use_hyde=True,
                query_config_overrides={**PLAIN, "enable_hyde": True},
            )
        assert stages == ["hyde"]

    def test_a_failed_query_expansion_is_named(self, orchestrator):
        class NoLLM:
            def generate_query_variations(self, query, n):
                raise RuntimeError("query variations need an LLM")

        orchestrator.search = lambda *a, **k: [
            SimpleNamespace(chunk=SimpleNamespace(chunk_id="c"), score=1.0, document=None)
        ]
        orchestrator._get_session_config = lambda sid: SimpleNamespace(
            enable_query_expansion=True,
            rerank_top_k=5,
            rrf_k=60,
            with_query_overrides=lambda o: None,
        )
        with _degraded.collecting() as stages:
            orchestrator.search_multi_query(
                "q", "s", top_k=5, n_variations=3, _enhancer_override=NoLLM()
            )
        assert stages == ["query_expansion"]

    def test_a_failed_focus_lookup_searches_nothing(self, orchestrator, monkeypatch):
        import lasagna.core.session_config as session_config

        def unreadable():
            raise ConnectionError("config store down")

        monkeypatch.setattr(
            session_config,
            "get_knowledge_config_manager",
            lambda *a, **k: SimpleNamespace(get_focused_document_ids=lambda sid: unreadable()),
        )
        with pytest.raises(orch_mod.SearchScopeUnavailable, match="focused documents"):
            orchestrator.search("q", "s1", top_k=1, query_config_overrides=PLAIN)

    def test_and_rest_answers_503_for_the_search(self, auth_off, monkeypatch):
        from lasagna.core import model_downloads

        monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"default": 1})

        async def async_search(**kwargs):
            raise orch_mod.SearchScopeUnavailable("focused documents could not be read")

        r = _client(SimpleNamespace(async_search=async_search)).post(
            "/api/v1/search", json={"query": "tides"}
        )
        assert r.status_code == 503
        assert "focused documents" in r.json()["detail"]

    def test_a_failed_native_hybrid_search_is_an_error_not_no_results(self):
        from lasagna.retrieval.search.hybrid import MilvusNativeHybridSearcher

        def down(**kw):
            raise ConnectionError("milvus unavailable")

        searcher = MilvusNativeHybridSearcher(SimpleNamespace(hybrid_search=down), "c", alpha=0.5)
        searcher._native_bm25_supported = True
        with pytest.raises(RuntimeError, match="milvus unavailable"):
            searcher.hybrid_search("q", [0.1, 0.2], "s1", top_k=3)

    def test_and_hybrid_search_falls_back_to_bm25(self):
        from lasagna.retrieval.search.hybrid import HybridSearcher

        def down(**kw):
            raise RuntimeError("Native hybrid search failed: milvus unavailable")

        h = HybridSearcher.__new__(HybridSearcher)
        h._backend_supports_native_hybrid = lambda sid: False
        h.is_native_mode_available = lambda: True
        h._native_searcher = SimpleNamespace(hybrid_search=down)
        h._fallback_hybrid_search = lambda **kw: (
            ["bm25 hit"],
            {"hybrid_mode": "rank_bm25_fallback"},
        )
        config = SimpleNamespace(hybrid_alpha=0.5, enable_hybrid_search=True, sparse_method="bm25")
        with _degraded.collecting() as stages:
            results, metrics = h.search_hybrid(
                "q", [], "s1", query_embedding=[0.1], return_metrics=True, config=config
            )
        assert results == ["bm25 hit"]
        assert metrics["hybrid_mode"] == "rank_bm25_fallback"
        assert stages == ["hybrid"]

    def test_a_store_error_listing_chunks_is_not_an_empty_list(self):
        from lasagna.api._document_pipeline import ChunkStoreUnavailable, DocumentPipeline

        def down(**kw):
            raise ConnectionError("vector store unreachable")

        store = SimpleNamespace(session_filter=lambda c, s: None, query=down)
        p = DocumentPipeline.__new__(DocumentPipeline)
        p._collection_manager = SimpleNamespace(resolve_backend=lambda sid: store)
        p._get_session_config = lambda sid: SimpleNamespace(collection_name="c")
        with pytest.raises(ChunkStoreUnavailable, match="vector store unreachable"):
            p.get_document_chunks("doc_1", "s1")

    def test_and_rest_answers_503_for_the_chunks(self, auth_off):
        from lasagna.api._document_pipeline import ChunkStoreUnavailable

        def get_document_chunks(**kwargs):
            raise ChunkStoreUnavailable("The chunks of doc_1 could not be read")

        r = _client(SimpleNamespace(get_document_chunks=get_document_chunks)).get(
            "/api/v1/documents/doc_1/chunks"
        )
        assert r.status_code == 503
        assert "could not be read" in r.json()["detail"]


# ---------------------------------------------------------------------------
# What the API says about itself
# ---------------------------------------------------------------------------


class TestTheApiDescribesItself:
    def test_the_description_is_plain(self, auth_off):
        info = create_app().openapi()["info"]
        assert info["description"].startswith("REST API for Lasagna")
        assert "Production" not in info["description"]

    def test_errors_have_a_documented_schema(self, auth_off):
        spec = create_app().openapi()
        responses = spec["paths"]["/api/v1/search"]["post"]["responses"]
        for code in ("401", "413", "429", "500", "503"):
            ref = responses[code]["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ErrorResponse"), (code, ref)

    def test_a_search_logs_its_shape_at_debug_without_claiming_hybrid(self, orchestrator, logs):
        orchestrator.search("q", "s1", top_k=1, query_config_overrides=PLAIN)
        shape = [r for r in logs if r["message"].startswith(("Search:", "SOTA search"))]
        assert shape, "the search did not reach its summary line"
        assert all(r["level"].name == "DEBUG" for r in shape)
        assert "hybrid ->" not in shape[0]["message"]
        assert "(dense_only)" in shape[0]["message"]


def test_the_server_settings_are_not_reported_as_unknown():
    """They are read through a table, which the config's own scan cannot see.

    Missing from the list, setting RAG_MAX_BATCH_FILES logged "is not a
    Lasagna setting and was ignored" while the server used it.
    """
    from lasagna.core.config import NON_SETTING_ENV

    assert set(limits._SETTINGS) | {"RAG_API_KEYS"} <= NON_SETTING_ENV
