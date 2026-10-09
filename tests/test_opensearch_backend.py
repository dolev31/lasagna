"""Tests for the OpenSearch vector database backend.

Unit tests run without OpenSearch — they mock the opensearch-py client
entirely and test query construction, pipeline management, hit conversion,
multi-search error handling, and config parsing.

Integration tests (marked @pytest.mark.integration) require a running
OpenSearch instance.

Run unit tests only:
    pytest tests/test_opensearch_backend.py -v -k "not integration"
"""

import json
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from lasagna.vector_backends.filters import (
    FilterExpression,
    FilterOperator,
)

# ---------------------------------------------------------------------------
# Conditional imports — skip gracefully if opensearch-py is not installed
# ---------------------------------------------------------------------------

try:
    from lasagna.vector_backends.opensearch_backend import (
        OPENSEARCH_AVAILABLE,
        OpenSearchBackend,
        _get_default_opensearch_config,
        _neural_sparse_enabled,
        _NEURAL_SPARSE_FIELD,
    )
except ImportError:
    OPENSEARCH_AVAILABLE = False
    OpenSearchBackend = None  # type: ignore
    _get_default_opensearch_config = None  # type: ignore
    _neural_sparse_enabled = None  # type: ignore
    _NEURAL_SPARSE_FIELD = "sparse_tokens"  # type: ignore

pytestmark = pytest.mark.skipif(
    not OPENSEARCH_AVAILABLE,
    reason="opensearch-py not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Create a mock OpenSearch client."""
    client = MagicMock()
    client.ping.return_value = True
    client.indices.exists.return_value = True
    client.indices.get_mapping.return_value = {}
    client.transport.perform_request.return_value = {}
    client.search.return_value = {"hits": {"hits": []}}
    client.msearch.return_value = {"responses": []}
    return client


@pytest.fixture
def backend(mock_client):
    """Create an OpenSearchBackend with a mocked client (no real connection)."""
    b = OpenSearchBackend(
        hosts=["http://localhost:9200"],
        use_ssl=False,
        verify_certs=False,
    )
    b._client = mock_client
    return b


def _make_connected_backend() -> "OpenSearchBackend":
    """Create an OpenSearchBackend with a mock client (standalone, no fixtures).

    Useful for class-based tests that don't use pytest fixtures.
    """
    client = MagicMock()
    client.ping.return_value = True
    client.indices.exists.return_value = True
    client.indices.get_mapping.return_value = {}
    client.transport.perform_request.return_value = {}
    client.search.return_value = {"hits": {"hits": []}}
    client.msearch.return_value = {"responses": []}
    client.delete_by_query.return_value = {"deleted": 0}
    b = OpenSearchBackend(
        hosts=["http://localhost:9200"],
        use_ssl=False,
        verify_certs=False,
    )
    b._client = client
    return b


# ============================================================================
# Pipeline Cache Tests — Bug fixes #1, #3, #7
# ============================================================================


class TestPipelineCache:
    """Test _ensure_hybrid_pipeline() pipeline naming, caching, and error handling."""

    def test_pipeline_name_encodes_alpha(self, backend, mock_client):
        """Bug fix #1: Pipeline name must include alpha weight and mode tag."""
        name = backend._ensure_hybrid_pipeline("my_index", 0.7)
        assert name == "rag-hybrid-my_index-2w-a0.70"

        # Verify PUT was called with the correct pipeline name
        mock_client.transport.perform_request.assert_called_once()
        call_args = mock_client.transport.perform_request.call_args
        assert call_args[0][0] == "PUT"
        assert call_args[0][1] == "/_search/pipeline/rag-hybrid-my_index-2w-a0.70"

    def test_cache_hit_skips_creation(self, backend, mock_client):
        """Cached pipeline with matching alpha should not trigger PUT."""
        backend._pipeline_cache["my_index"] = "rag-hybrid-my_index-2w-a0.70"

        name = backend._ensure_hybrid_pipeline("my_index", 0.7)
        assert name == "rag-hybrid-my_index-2w-a0.70"
        mock_client.transport.perform_request.assert_not_called()

    def test_alpha_change_deletes_old_pipeline(self, backend, mock_client):
        """Bug fix #3: Alpha change must delete old pipeline and create new one."""
        backend._pipeline_cache["my_index"] = "rag-hybrid-my_index-2w-a0.70"

        name = backend._ensure_hybrid_pipeline("my_index", 0.5)
        assert name == "rag-hybrid-my_index-2w-a0.50"

        # Should have called DELETE for old pipeline, then PUT for new
        calls = mock_client.transport.perform_request.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "DELETE"
        assert "a0.70" in calls[0][0][1]
        assert calls[1][0][0] == "PUT"
        assert "a0.50" in calls[1][0][1]

    def test_pipeline_creation_error_raises_runtime(self, backend, mock_client):
        """Bug fix #7: Pipeline creation failure must raise RuntimeError."""
        mock_client.transport.perform_request.side_effect = Exception("timeout")

        with pytest.raises(RuntimeError, match="Cannot create OpenSearch hybrid search pipeline"):
            backend._ensure_hybrid_pipeline("my_index", 0.7)

    def test_drop_collection_cleans_pipeline(self, backend, mock_client):
        """drop_collection must delete cached pipeline."""
        backend._pipeline_cache["my_index"] = "rag-hybrid-my_index-2w-a0.70"

        backend.drop_collection("my_index")

        assert "my_index" not in backend._pipeline_cache
        # DELETE for pipeline should have been called
        pipeline_delete_calls = [
            c
            for c in mock_client.transport.perform_request.call_args_list
            if c[0][0] == "DELETE" and "pipeline" in c[0][1]
        ]
        assert len(pipeline_delete_calls) == 1
        assert "a0.70" in pipeline_delete_calls[0][0][1]

    def test_pipeline_weights_match_alpha_2way(self, backend, mock_client):
        """2-way pipeline body weights must match [alpha, 1-alpha]."""
        backend._ensure_hybrid_pipeline("idx", 0.3)

        put_call = mock_client.transport.perform_request.call_args
        body = put_call[1].get("body") or put_call[0][2] if len(put_call[0]) > 2 else None
        if body is None:
            body = put_call.kwargs.get("body")
        assert body is not None
        weights = body["phase_results_processors"][0]["normalization-processor"]["combination"][
            "parameters"
        ]["weights"]
        assert weights == [0.3, 0.7]

    def test_pipeline_3way_weights(self, backend, mock_client):
        """3-way pipeline splits (1-alpha) equally between sparse and BM25."""
        name = backend._ensure_hybrid_pipeline("idx", 0.7, n_queries=3)
        assert "3w" in name

        put_call = mock_client.transport.perform_request.call_args
        body = put_call[1].get("body") or put_call[0][2] if len(put_call[0]) > 2 else None
        if body is None:
            body = put_call.kwargs.get("body")
        weights = body["phase_results_processors"][0]["normalization-processor"]["combination"][
            "parameters"
        ]["weights"]
        # With alpha=0.7: dense=0.7, remainder=0.3 split equally → sparse=0.15, bm25=0.15
        assert weights == [0.7, 0.15, 0.15]
        assert len(weights) == 3

    def test_pipeline_mode_tag_2w_vs_3w(self, backend, mock_client):
        """Pipeline name must include '2w' or '3w' mode tag."""
        name_2w = backend._ensure_hybrid_pipeline("idx", 0.7, n_queries=2)
        assert "-2w-" in name_2w

        # Reset cache to force creation of 3w pipeline
        backend._pipeline_cache.pop("idx", None)
        mock_client.transport.perform_request.reset_mock()

        name_3w = backend._ensure_hybrid_pipeline("idx", 0.7, n_queries=3)
        assert "-3w-" in name_3w
        assert name_2w != name_3w


# ============================================================================
# Hit Conversion Tests — Bug fixes #4, #6
# ============================================================================


class TestHitConversion:
    """Test _hit_to_result() score normalization, entity population, and highlights."""

    def test_chunk_id_always_in_entity(self, backend):
        """Bug fix #6: chunk_id must always be present in entity."""
        hit = {"_id": "c42", "_source": {"content": "hello"}, "_score": 0.8}
        result = backend._hit_to_result(hit)
        assert result.entity["chunk_id"] == "c42"
        assert result.id == "c42"

    def test_no_fabricated_dense_sparse_scores(self, backend):
        """Bug fix #4: OpenSearch returns combined score only — no dense/sparse breakdown."""
        hit = {"_id": "c1", "_source": {}, "_score": 0.75}
        result = backend._hit_to_result(hit)
        # VectorSearchResult does not have dense_score/sparse_score as separate attributes
        # by default — the backend must not invent component scores
        assert result.score == 0.75
        # raw_score is preserved
        assert result.raw_score == 0.75

    def test_highlights_extracted_from_hit(self, backend):
        """Highlights from BM25 matching should appear in entity."""
        hit = {
            "_id": "c1",
            "_source": {"content": "text"},
            "_score": 0.9,
            "highlight": {"content": ["frag1", "frag2"]},
        }
        result = backend._hit_to_result(hit)
        assert result.entity["highlights"] == ["frag1", "frag2"]

    def test_highlights_flattened_from_multiple_fields(self, backend):
        """Highlights from multiple fields should be flattened into one list."""
        hit = {
            "_id": "c1",
            "_source": {},
            "_score": 0.8,
            "highlight": {
                "content": ["frag1"],
                "title": ["frag2", "frag3"],
            },
        }
        result = backend._hit_to_result(hit)
        assert len(result.entity["highlights"]) == 3
        assert "frag1" in result.entity["highlights"]
        assert "frag2" in result.entity["highlights"]
        assert "frag3" in result.entity["highlights"]

    def test_missing_highlights_no_key(self, backend):
        """When no highlights in hit, entity should not have highlights key."""
        hit = {"_id": "c1", "_source": {}, "_score": 0.5}
        result = backend._hit_to_result(hit)
        assert "highlights" not in result.entity

    def test_score_preserved_unclamped(self, backend):
        """Scores are passed through without clamping to preserve ranking signal."""
        high = backend._hit_to_result({"_id": "c1", "_source": {}, "_score": 1.5})
        assert high.score == 1.5

        low = backend._hit_to_result({"_id": "c2", "_source": {}, "_score": -0.1})
        assert low.score == -0.1

    def test_none_score_becomes_zero(self, backend):
        """None score should normalize to 0.0."""
        result = backend._hit_to_result({"_id": "c1", "_source": {}, "_score": None})
        assert result.score == 0.0

    def test_entity_fields_populated(self, backend):
        """Core entity fields should be populated from _source."""
        hit = {
            "_id": "c1",
            "_source": {
                "document_id": "doc1",
                "session_id": "sess1",
                "content": "test content",
                "metadata": {"page": 1},
            },
            "_score": 0.5,
        }
        result = backend._hit_to_result(hit)
        assert result.entity["document_id"] == "doc1"
        assert result.entity["session_id"] == "sess1"
        assert result.entity["content"] == "test content"
        assert result.entity["metadata"] == {"page": 1}

    def test_none_values_in_source_become_safe_defaults(self, backend):
        """Explicit None values in _source must not leak into entity.

        dict.get(key, default) returns None when key exists with value None.
        Downstream code (ranker, citation_manager) does direct bracket access
        on entity fields — None would cause AttributeError/TypeError crashes.
        """
        hit = {
            "_id": "c1",
            "_source": {
                "document_id": None,
                "session_id": None,
                "content": None,
                "metadata": None,
                "embedding": None,
            },
            "_score": 0.5,
        }
        result = backend._hit_to_result(hit)
        assert result.entity["document_id"] == ""
        assert result.entity["session_id"] == ""
        assert result.entity["content"] == ""
        assert result.entity["metadata"] == {}
        # embedding=None is acceptable (not always returned)
        assert result.entity["embedding"] is None


class TestInsertNoneSafety:
    """Test that insert() converts None field values to safe defaults."""

    def test_none_metadata_becomes_empty_dict(self, backend, mock_client):
        """metadata=None must not be stored as null (would crash downstream)."""
        with patch(
            "lasagna.vector_backends.opensearch_backend.os_bulk",
            return_value=(1, []),
        ) as mock_bulk:
            backend.insert(
                "idx",
                {
                    "chunk_id": "c1",
                    "content": "text",
                    "embedding": [0.1] * 10,
                    "metadata": None,
                },
            )
            actions = mock_bulk.call_args[0][1]
            source = actions[0]["_source"]
            assert source["metadata"] == {}

    def test_none_string_fields_become_empty_string(self, backend, mock_client):
        """None for string fields must not be stored as null."""
        with patch(
            "lasagna.vector_backends.opensearch_backend.os_bulk",
            return_value=(1, []),
        ) as mock_bulk:
            backend.insert(
                "idx",
                {
                    "chunk_id": "c1",
                    "document_id": None,
                    "session_id": None,
                    "content": None,
                    "embedding": [0.1] * 10,
                },
            )
            actions = mock_bulk.call_args[0][1]
            source = actions[0]["_source"]
            assert source["document_id"] == ""
            assert source["session_id"] == ""
            assert source["content"] == ""


# ============================================================================
# Hybrid Search Query Tests — Bug fix #2 + capabilities
# ============================================================================


class TestHybridSearchQuery:
    """Test hybrid_search() query body construction."""

    def _capture_search_body(self, mock_client, backend, **kwargs):
        """Call hybrid_search and capture the search body."""
        defaults = {
            "collection_name": "test_idx",
            "query_text": "machine learning",
            "query_embedding": [0.1] * 10,
            "limit": 5,
            "alpha": 0.7,
        }
        defaults.update(kwargs)

        # Mock pipeline creation
        backend._pipeline_cache["test_idx"] = "rag-hybrid-test_idx-2w-a0.70"

        mock_client.search.return_value = {"hits": {"hits": []}}
        backend.hybrid_search(**defaults)
        return mock_client.search.call_args

    def test_fuzzy_match_in_bm25(self, backend, mock_client):
        """BM25 sub-query must include fuzziness=AUTO."""
        call_args = self._capture_search_body(mock_client, backend)
        body = call_args[1].get("body") or call_args.kwargs.get("body")
        body_str = json.dumps(body)
        assert '"fuzziness": "AUTO"' in body_str or '"fuzziness":"AUTO"' in body_str

    def test_phrase_boost_in_bm25(self, backend, mock_client):
        """BM25 sub-query must include match_phrase with boost."""
        call_args = self._capture_search_body(mock_client, backend)
        body = call_args[1].get("body") or call_args.kwargs.get("body")
        body_str = json.dumps(body)
        assert "match_phrase" in body_str
        assert '"boost": 1.5' in body_str or '"boost":1.5' in body_str

    def test_highlight_block_present(self, backend, mock_client):
        """Hybrid query must include highlight configuration."""
        call_args = self._capture_search_body(mock_client, backend)
        body = call_args[1].get("body") or call_args.kwargs.get("body")
        assert "highlight" in body
        highlight = body["highlight"]
        assert "fields" in highlight
        # Check fragment_size and number_of_fragments
        field_config = list(highlight["fields"].values())[0]
        assert field_config["fragment_size"] == 200
        assert field_config["number_of_fragments"] == 3

    def test_filter_inside_queries_not_post_filter(self, backend, mock_client):
        """Bug fix #2: Filters must be inside sub-queries, NOT as post_filter."""
        filter_expr = FilterExpression.eq("session_id", "sess1")
        call_args = self._capture_search_body(mock_client, backend, filter_expr=filter_expr)
        body = call_args[1].get("body") or call_args.kwargs.get("body")

        # Must NOT have post_filter
        assert "post_filter" not in body

        # Filter should be inside the queries
        body_str = json.dumps(body)
        assert "session_id" in body_str

    def test_collapse_for_group_by_document(self, backend, mock_client):
        """group_by_document=True must add collapse by document_id."""
        call_args = self._capture_search_body(mock_client, backend, group_by_document=True)
        body = call_args[1].get("body") or call_args.kwargs.get("body")
        assert "collapse" in body
        assert body["collapse"]["field"] == "document_id"

    def test_collapse_for_group_by_document_pure_bm25(self, backend, mock_client):
        """group_by_document=True must add collapse even in pure BM25 (alpha=0.0)."""
        mock_client.search.return_value = {"hits": {"hits": []}}
        backend.hybrid_search(
            collection_name="test_idx",
            query_text="test query",
            query_embedding=[0.1] * 10,
            limit=5,
            alpha=0.0,
            group_by_document=True,
        )
        body = mock_client.search.call_args.kwargs.get("body") or mock_client.search.call_args[
            1
        ].get("body")
        assert "collapse" in body
        assert body["collapse"]["field"] == "document_id"

    def test_search_pipeline_param(self, backend, mock_client):
        """Search request must include search_pipeline parameter."""
        call_args = self._capture_search_body(mock_client, backend)
        params = call_args[1].get("params") or call_args.kwargs.get("params", {})
        assert "search_pipeline" in params
        assert params["search_pipeline"].startswith("rag-hybrid-")

    def test_hybrid_query_uses_hybrid_dsl(self, backend, mock_client):
        """Query body must use OpenSearch hybrid DSL."""
        call_args = self._capture_search_body(mock_client, backend)
        body = call_args[1].get("body") or call_args.kwargs.get("body")
        assert "query" in body
        assert "hybrid" in body["query"]
        assert "queries" in body["query"]["hybrid"]
        queries = body["query"]["hybrid"]["queries"]
        assert len(queries) == 2  # k-NN + BM25

    def test_pure_vector_when_alpha_one(self, backend, mock_client):
        """alpha=1.0 should short-circuit to pure vector search."""
        backend._pipeline_cache["test_idx"] = "rag-hybrid-test_idx-2w-a1.00"
        mock_client.search.return_value = {"hits": {"hits": []}}
        backend.hybrid_search(
            collection_name="test_idx",
            query_text="test",
            query_embedding=[0.1] * 10,
            limit=5,
            alpha=1.0,
        )
        body = mock_client.search.call_args.kwargs.get("body") or mock_client.search.call_args[
            1
        ].get("body")
        # Should NOT have hybrid query, just knn
        assert "hybrid" not in json.dumps(body)

    def test_pure_bm25_when_alpha_zero(self, backend, mock_client):
        """alpha=0.0 should short-circuit to pure BM25 search."""
        mock_client.search.return_value = {"hits": {"hits": []}}
        backend.hybrid_search(
            collection_name="test_idx",
            query_text="test",
            query_embedding=[0.1] * 10,
            limit=5,
            alpha=0.0,
        )
        body = mock_client.search.call_args.kwargs.get("body") or mock_client.search.call_args[
            1
        ].get("body")
        # Should NOT have hybrid or knn, just BM25
        body_str = json.dumps(body)
        assert "hybrid" not in body_str
        assert "knn" not in body_str
        assert "match" in body_str


# ============================================================================
# Multi-Search Tests — Bug fix #8
# ============================================================================


class TestMultiSearch:
    """Test multi_search() _msearch request structure and error handling."""

    def test_msearch_request_structure(self, backend, mock_client):
        """_msearch body must alternate header/body pairs."""
        body1 = {"query": {"match_all": {}}, "size": 5}
        body2 = {"query": {"match": {"content": "test"}}, "size": 10}

        mock_client.msearch.return_value = {
            "responses": [
                {"hits": {"hits": []}},
                {"hits": {"hits": []}},
            ]
        }

        backend.multi_search("my_index", [body1, body2])

        msearch_call = mock_client.msearch.call_args
        body_parts = msearch_call.kwargs.get("body") or msearch_call[1].get("body")
        assert len(body_parts) == 4  # 2 headers + 2 bodies
        assert body_parts[0] == {"index": "my_index"}
        assert body_parts[1] == body1
        assert body_parts[2] == {"index": "my_index"}
        assert body_parts[3] == body2

    def test_msearch_sub_error_returns_empty_list(self, backend, mock_client):
        """Bug fix #8: Failed sub-query returns [] and logs warning."""
        hit = {"_id": "c1", "_source": {"content": "ok"}, "_score": 0.5}
        mock_client.msearch.return_value = {
            "responses": [
                {"error": {"type": "search_phase_execution_exception", "reason": "fail"}},
                {"hits": {"hits": [hit]}},
            ]
        }

        results = backend.multi_search("my_index", [{"query": {}}, {"query": {}}])
        assert len(results) == 2
        assert results[0] == []  # Failed sub-query returns empty
        assert len(results[1]) == 1  # Successful sub-query returns results


# ============================================================================
# Index Settings Tests
# ============================================================================


class TestIndexSettings:
    """Test _build_index_settings() and _schema_to_mapping() configuration."""

    def test_shards_replicas_from_env(self, backend):
        """Shards and replicas should be read from environment variables."""
        with patch.dict(os.environ, {"OPENSEARCH_SHARDS": "3", "OPENSEARCH_REPLICAS": "2"}):
            settings = backend._build_index_settings()

        assert settings["index"]["number_of_shards"] == 3
        assert settings["index"]["number_of_replicas"] == 2

    def test_default_shards_replicas(self, backend):
        """Default: 1 shard, 0 replicas (dev mode)."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure env vars are not set
            env = os.environ.copy()
            env.pop("OPENSEARCH_SHARDS", None)
            env.pop("OPENSEARCH_REPLICAS", None)
            with patch.dict(os.environ, env, clear=True):
                settings = backend._build_index_settings()

        assert settings["index"]["number_of_shards"] == 1
        assert settings["index"]["number_of_replicas"] == 0

    def test_language_analyzer_applied(self, backend):
        """Content fields should use the configured language analyzer."""
        from lasagna.vector_backends.base import FieldSchema, CollectionSchema

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True),
                FieldSchema(name="content", dtype="VARCHAR"),
            ]
        )

        with patch.dict(os.environ, {"OPENSEARCH_LANGUAGE": "german"}):
            mapping = backend._schema_to_mapping(schema)

        assert mapping["properties"]["content"]["analyzer"] == "german"

    def test_id_fields_use_keyword(self, backend):
        """ID fields (chunk_id, document_id, session_id) should use keyword type."""
        from lasagna.vector_backends.base import FieldSchema, CollectionSchema

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True),
                FieldSchema(name="document_id", dtype="VARCHAR"),
                FieldSchema(name="session_id", dtype="VARCHAR"),
                FieldSchema(name="content", dtype="VARCHAR"),
            ]
        )
        mapping = backend._schema_to_mapping(schema)

        assert mapping["properties"]["chunk_id"]["type"] == "keyword"
        assert mapping["properties"]["document_id"]["type"] == "keyword"
        assert mapping["properties"]["session_id"]["type"] == "keyword"
        assert mapping["properties"]["content"]["type"] == "text"

    def test_knn_vector_field(self, backend):
        """FLOAT_VECTOR fields should map to knn_vector type."""
        from lasagna.vector_backends.base import FieldSchema, CollectionSchema

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=768),
            ]
        )
        mapping = backend._schema_to_mapping(schema)
        assert mapping["properties"]["embedding"]["type"] == "knn_vector"
        assert mapping["properties"]["embedding"]["dimension"] == 768

    def test_partition_key_always_added(self, backend):
        """partition_key field should always be in mapping."""
        from lasagna.vector_backends.base import FieldSchema, CollectionSchema

        schema = CollectionSchema(
            fields=[FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True)]
        )
        mapping = backend._schema_to_mapping(schema)
        assert "partition_key" in mapping["properties"]
        assert mapping["properties"]["partition_key"]["type"] == "keyword"


# ============================================================================
# Init Config Tests — Bug fix #5
# ============================================================================


class TestInitConfig:
    """Test _get_default_opensearch_config() environment variable parsing."""

    def test_bool_env_parsing_true(self):
        """Bug fix #5: use_ssl and verify_certs must be parsed as booleans."""
        with patch.dict(
            os.environ,
            {
                "OPENSEARCH_USE_SSL": "true",
                "OPENSEARCH_VERIFY_CERTS": "True",
            },
        ):
            config = _get_default_opensearch_config()

        assert config["use_ssl"] is True
        assert config["verify_certs"] is True

    def test_bool_env_parsing_false(self):
        """String 'false' must produce Python False, not truthy string."""
        with patch.dict(
            os.environ,
            {
                "OPENSEARCH_USE_SSL": "false",
                "OPENSEARCH_VERIFY_CERTS": "FALSE",
            },
        ):
            config = _get_default_opensearch_config()

        assert config["use_ssl"] is False
        assert config["verify_certs"] is False

    def test_hosts_comma_separated(self):
        """Multiple hosts should be parsed from comma-separated string."""
        with patch.dict(
            os.environ,
            {
                "OPENSEARCH_HOSTS": "http://host1:9200, http://host2:9200",
            },
        ):
            config = _get_default_opensearch_config()

        assert config["hosts"] == ["http://host1:9200", "http://host2:9200"]

    def test_timeout_parsed_as_int(self):
        """Timeout should be parsed as integer."""
        with patch.dict(os.environ, {"OPENSEARCH_TIMEOUT": "60"}):
            config = _get_default_opensearch_config()
        assert config["timeout"] == 60

    def test_no_auth_when_user_empty(self):
        """http_auth should be None when OPENSEARCH_USER is empty."""
        with patch.dict(os.environ, {"OPENSEARCH_USER": ""}, clear=False):
            config = _get_default_opensearch_config()
        assert config["http_auth"] is None


# ============================================================================
# Capabilities Tests
# ============================================================================


class TestCapabilities:
    """Test backend capabilities reporting."""

    def test_capability_flags(self, backend):
        """OpenSearch should report correct capability flags."""
        caps = backend.capabilities
        assert caps.supports_hybrid_search is True
        assert caps.supports_native_bm25 is True
        # supports_sparse_vectors is dynamic — reflects OPENSEARCH_ENABLE_NEURAL_SPARSE env var
        assert caps.supports_sparse_vectors is False  # disabled by default
        assert caps.supports_hnsw is True
        assert caps.supports_ivf is False
        assert caps.supports_gpu is False
        assert caps.supports_json_filters is True
        assert caps.supports_metadata_filtering is True
        assert caps.supports_partitions is False
        assert caps.supports_dynamic_schema is True

    def test_backend_name(self, backend):
        assert backend.backend_name == "opensearch"


# ============================================================================
# Validation Tests — Surface consistency
# ============================================================================


class TestValidation:
    """Test that opensearch is accepted in API models and configurator schema."""

    def test_config_patch_accepts_opensearch(self):
        """ConfigPatchRequest must validate 'opensearch' as a valid backend."""
        from lasagna.api.rest.models import ConfigPatchRequest

        req = ConfigPatchRequest(vector_backend="opensearch")
        assert req.vector_backend == "opensearch"

    def test_config_patch_rejects_invalid_backend(self):
        """ConfigPatchRequest must reject unknown backend names."""
        from pydantic import ValidationError
        from lasagna.api.rest.models import ConfigPatchRequest

        with pytest.raises(ValidationError):
            ConfigPatchRequest(vector_backend="elasticsearch")

    def test_configurator_schema_includes_opensearch(self):
        """Configurator schema must include opensearch as a vector_backend option."""
        from lasagna.configurator.schema import PARAMETERS

        vector_backend_param = None
        for p in PARAMETERS:
            if p.key == "vector_backend":
                vector_backend_param = p
                break

        assert vector_backend_param is not None, "vector_backend parameter not found in schema"
        option_values = [opt.value for opt in vector_backend_param.enum_options]
        assert "opensearch" in option_values


# ============================================================================
# Connection Lifecycle Tests
# ============================================================================


class TestConnectionLifecycle:
    """Test connect/disconnect behavior."""

    def test_disconnect_clears_pipeline_cache(self, backend, mock_client):
        """Disconnect must clear pipeline cache."""
        backend._pipeline_cache["idx1"] = "pipeline1"
        backend._pipeline_cache["idx2"] = "pipeline2"

        backend.disconnect()

        assert len(backend._pipeline_cache) == 0
        assert backend._client is None

    def test_is_connected_false_after_disconnect(self, backend, mock_client):
        """is_connected() should return False after disconnect."""
        backend.disconnect()
        assert backend.is_connected() is False


# ============================================================================
# Filter Construction Tests
# ============================================================================


class TestFilterConstruction:
    """Test _build_search_filter() combines filter_expr and partition_key."""

    def test_no_filter_returns_none(self, backend):
        result = backend._build_search_filter(None, None)
        assert result is None

    def test_filter_expr_only(self, backend):
        expr = FilterExpression.eq("session_id", "s1")
        result = backend._build_search_filter(expr, None)
        assert result is not None
        assert "term" in json.dumps(result)

    def test_partition_key_only(self, backend):
        result = backend._build_search_filter(None, "tenant_a")
        assert result == {"term": {"partition_key": {"value": "tenant_a"}}}

    def test_combined_filter(self, backend):
        expr = FilterExpression.eq("session_id", "s1")
        result = backend._build_search_filter(expr, "tenant_a")
        assert "bool" in result
        assert "must" in result["bool"]
        assert len(result["bool"]["must"]) == 2


# ============================================================================
# PR2 Enhancement Tests — Highlights, _msearch batch, 3-way hybrid
# ============================================================================


class TestHighlightPropagation:
    """Test highlight fragments flow through HybridSearchResult → RerankedResult → Citation."""

    def test_hybrid_search_result_has_highlights_field(self):
        """HybridSearchResult should have optional highlights field."""
        from lasagna.retrieval.search.hybrid import HybridSearchResult

        result = HybridSearchResult(
            chunk_id="c1",
            document_id="d1",
            content="text",
            metadata={},
            combined_score=0.9,
            highlights=["frag1", "frag2"],
        )
        assert result.highlights == ["frag1", "frag2"]

    def test_hybrid_search_result_highlights_default_none(self):
        from lasagna.retrieval.search.hybrid import HybridSearchResult

        result = HybridSearchResult(
            chunk_id="c1",
            document_id="d1",
            content="text",
            metadata={},
            combined_score=0.9,
        )
        assert result.highlights is None

    def test_reranked_result_has_highlights_field(self):
        """RerankedResult should have optional highlights field."""
        from lasagna.retrieval.search.ranker import RerankedResult

        result = RerankedResult(
            chunk_id="c1",
            document_id="d1",
            content="text",
            metadata={},
            rerank_score=0.8,
            original_rank=0,
            original_score=0.8,
            final_rank=0,
            normalized_score=0.8,
            highlights=["frag1"],
        )
        assert result.highlights == ["frag1"]

    def test_citation_has_highlights_field(self):
        """Citation should have optional highlights field."""
        from lasagna.agentic.citation_manager import Citation

        citation = Citation(
            citation_id="cit1",
            index=0,
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            content="text",
            content_preview="text",
            score=0.9,
            normalized_score=0.9,
            chunk_index=0,
            highlights=["frag1", "frag2"],
        )
        assert citation.highlights == ["frag1", "frag2"]

    def test_citation_to_dict_includes_highlights(self):
        """Citation.to_dict() should include highlights when present."""
        from lasagna.agentic.citation_manager import Citation

        citation = Citation(
            citation_id="cit1",
            index=0,
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            content="text",
            content_preview="text",
            score=0.9,
            normalized_score=0.9,
            chunk_index=0,
            highlights=["match1", "match2"],
        )
        d = citation.to_dict()
        assert d["highlights"] == ["match1", "match2"]

    def test_citation_to_dict_omits_highlights_when_none(self):
        """Citation.to_dict() should not include highlights key when None."""
        from lasagna.agentic.citation_manager import Citation

        citation = Citation(
            citation_id="cit1",
            index=0,
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            content="text",
            content_preview="text",
            score=0.9,
            normalized_score=0.9,
            chunk_index=0,
        )
        d = citation.to_dict()
        assert "highlights" not in d

    def test_citation_response_has_highlights_fields(self):
        """CitationResponse should have highlights and highlight_ranges fields."""
        from lasagna.api.rest.models import CitationResponse

        resp = CitationResponse(
            citation_id="cit1",
            document_id="d1",
            document_name="doc.pdf",
            chunk_id="c1",
            chunk_index=0,
            score=0.9,
            content="text",
            highlights=["frag1", "frag2"],
            highlight_ranges=[[0, 10], [20, 30]],
        )
        assert resp.highlights == ["frag1", "frag2"]
        assert resp.highlight_ranges == [[0, 10], [20, 30]]

    def test_sdk_citation_has_highlights_field(self):
        """SDK Citation should have optional highlights field."""
        from lasagna.sdk.types import Citation

        c = Citation(
            citation_id="cit1",
            chunk_id="c1",
            document_id="d1",
            filename="doc.pdf",
            highlights=["match1"],
        )
        assert c.highlights == ["match1"]


class TestMultiHybridSearch:
    """Test multi_hybrid_search() _msearch batching for RAG Fusion."""

    def test_multi_hybrid_search_request_structure(self, backend, mock_client):
        """Each query should produce header+body pairs with search_pipeline."""
        backend._pipeline_cache["idx"] = "rag-hybrid-idx-2w-a0.70"
        mock_client.msearch.return_value = {
            "responses": [
                {"hits": {"hits": []}},
                {"hits": {"hits": []}},
            ]
        }

        backend.multi_hybrid_search(
            collection_name="idx",
            query_texts=["q1", "q2"],
            query_embeddings=[[0.1] * 10, [0.2] * 10],
            alpha=0.7,
        )

        msearch_call = mock_client.msearch.call_args
        body_parts = msearch_call.kwargs.get("body") or msearch_call[1].get("body")
        assert len(body_parts) == 4  # 2 headers + 2 bodies

        # Each header should have index
        assert body_parts[0]["index"] == "idx"
        assert body_parts[2]["index"] == "idx"

        # search_pipeline is passed as a URL query param, not per-header
        params = msearch_call.kwargs.get("params") or msearch_call[1].get("params", {})
        assert params.get("search_pipeline") == "rag-hybrid-idx-2w-a0.70"

        # Each body should have hybrid query
        assert "hybrid" in json.dumps(body_parts[1])
        assert "hybrid" in json.dumps(body_parts[3])

    def test_multi_hybrid_search_sub_error_returns_empty(self, backend, mock_client):
        """Failed sub-query in _msearch returns [] for that query."""
        backend._pipeline_cache["idx"] = "rag-hybrid-idx-2w-a0.70"
        hit = {"_id": "c1", "_source": {"content": "ok"}, "_score": 0.5}
        mock_client.msearch.return_value = {
            "responses": [
                {"error": {"type": "search_error", "reason": "fail"}},
                {"hits": {"hits": [hit]}},
            ]
        }

        results = backend.multi_hybrid_search(
            collection_name="idx",
            query_texts=["q1", "q2"],
            query_embeddings=[[0.1] * 10, [0.2] * 10],
            alpha=0.7,
        )
        assert len(results) == 2
        assert results[0] == []
        assert len(results[1]) == 1

    def test_multi_hybrid_search_empty_input(self, backend, mock_client):
        """Empty input should return empty list."""
        assert backend.multi_hybrid_search("idx", [], []) == []

    def test_multi_hybrid_search_pads_missing_responses(self, backend, mock_client):
        """Missing responses should be padded with empty lists."""
        backend._pipeline_cache["idx"] = "rag-hybrid-idx-2w-a0.70"
        mock_client.msearch.return_value = {"responses": []}

        results = backend.multi_hybrid_search(
            collection_name="idx",
            query_texts=["q1", "q2"],
            query_embeddings=[[0.1] * 10, [0.2] * 10],
            alpha=0.7,
        )
        assert len(results) == 2
        assert results[0] == []
        assert results[1] == []


class TestMinimumShouldMatch:
    """Test minimum_should_match guard in BM25 queries with filter."""

    def test_pure_bm25_with_filter_has_minimum_should_match(self, backend, mock_client):
        """Pure BM25 (alpha=0) with filter must set minimum_should_match=1.

        Without it, bool {should + filter} makes 'should' optional — docs pass
        the filter but match zero BM25 clauses, returning irrelevant results.
        """
        mock_client.search.return_value = {"hits": {"hits": []}}
        backend.hybrid_search(
            collection_name="test_idx",
            query_text="test query",
            query_embedding=[0.1] * 10,
            alpha=0.0,
            partition_key="tenant_1",
        )
        body = mock_client.search.call_args.kwargs.get("body") or mock_client.search.call_args[
            1
        ].get("body")
        query = body["query"]
        assert query["bool"]["minimum_should_match"] == 1

    def test_pure_bm25_without_filter_no_minimum_should_match(self, backend, mock_client):
        """Pure BM25 without filter should NOT set minimum_should_match (not needed)."""
        mock_client.search.return_value = {"hits": {"hits": []}}
        backend.hybrid_search(
            collection_name="test_idx",
            query_text="test query",
            query_embedding=[0.1] * 10,
            alpha=0.0,
        )
        body = mock_client.search.call_args.kwargs.get("body") or mock_client.search.call_args[
            1
        ].get("body")
        query = body["query"]
        assert "minimum_should_match" not in query["bool"]

    def test_hybrid_body_bm25_with_filter_has_minimum_should_match(self, backend):
        """BM25 sub-query in _build_hybrid_body with filter must set minimum_should_match=1."""
        os_filter = {"term": {"session_id": {"value": "s1"}}}
        body = backend._build_hybrid_body(
            query_text="test",
            query_embedding=[0.1] * 10,
            vector_field="embedding",
            sparse_field="content",
            os_filter=os_filter,
            limit=5,
        )
        bm25_subquery = body["query"]["hybrid"]["queries"][1]
        assert bm25_subquery["bool"]["minimum_should_match"] == 1

    def test_hybrid_body_bm25_without_filter_no_minimum_should_match(self, backend):
        """BM25 sub-query without filter should NOT set minimum_should_match."""
        body = backend._build_hybrid_body(
            query_text="test",
            query_embedding=[0.1] * 10,
            vector_field="embedding",
            sparse_field="content",
            os_filter=None,
            limit=5,
        )
        bm25_subquery = body["query"]["hybrid"]["queries"][1]
        assert "minimum_should_match" not in bm25_subquery["bool"]


class TestMultiHybridSearchAlphaShortCircuits:
    """Test alpha edge-case short-circuits in multi_hybrid_search."""

    def test_alpha_one_uses_pure_vector_search(self, backend, mock_client):
        """alpha=1.0 should short-circuit to individual vector searches."""
        mock_client.search.return_value = {"hits": {"hits": []}}
        results = backend.multi_hybrid_search(
            collection_name="idx",
            query_texts=["q1", "q2"],
            query_embeddings=[[0.1] * 10, [0.2] * 10],
            alpha=1.0,
        )
        assert len(results) == 2
        # Should use search(), not msearch()
        assert mock_client.search.called
        assert not mock_client.msearch.called

    def test_alpha_zero_uses_pure_bm25_search(self, backend, mock_client):
        """alpha=0.0 should short-circuit to individual hybrid_search (pure BM25)."""
        mock_client.search.return_value = {"hits": {"hits": []}}
        results = backend.multi_hybrid_search(
            collection_name="idx",
            query_texts=["q1", "q2"],
            query_embeddings=[[0.1] * 10, [0.2] * 10],
            alpha=0.0,
        )
        assert len(results) == 2
        # Should use search() (via hybrid_search alpha=0), not msearch()
        assert mock_client.search.called
        assert not mock_client.msearch.called


class TestMultiHybridSearchDiagnostics:
    """Test diagnostic logging in multi_hybrid_search."""

    def test_fewer_responses_logs_warning(self, backend, mock_client):
        """When msearch returns fewer responses than expected, log a warning."""
        backend._pipeline_cache["idx"] = "rag-hybrid-idx-2w-a0.70"
        # Return 1 response for 2 queries
        mock_client.msearch.return_value = {"responses": [{"hits": {"hits": []}}]}

        with patch("lasagna.vector_backends.opensearch_backend.logger") as mock_logger:
            results = backend.multi_hybrid_search(
                collection_name="idx",
                query_texts=["q1", "q2"],
                query_embeddings=[[0.1] * 10, [0.2] * 10],
                alpha=0.7,
            )

        assert len(results) == 2
        assert results[1] == []  # Missing response → empty
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "expected 2" in warning_msg
        assert "got 1" in warning_msg


class TestPoolAcquireRelease:
    """Test pool _acquire/_release semaphore safety."""

    def test_acquire_on_disconnected_pool_raises_runtime_error(self):
        """After disconnect, _acquire must raise RuntimeError, not IndexError."""
        from lasagna.vector_backends.pool import PooledVectorBackend
        from lasagna.vector_backends.base import VectorStoreAdapter

        _mock_factory = MagicMock(spec=VectorStoreAdapter)
        mock_backend = MagicMock(spec=VectorStoreAdapter)
        mock_backend.backend_name = "test"
        mock_backend.capabilities = MagicMock()

        pool = PooledVectorBackend(factory=lambda: mock_backend, pool_size=1)
        pool.connect()
        pool.disconnect()

        with pytest.raises(RuntimeError, match="not connected"):
            pool._acquire()

        # Semaphore must NOT be leaked — next connect should work
        pool._pool.append(mock_backend)
        pool._connected = True
        backend = pool._acquire()
        assert backend is mock_backend
        pool._release(backend)

    def test_semaphore_not_leaked_on_acquire_failure(self):
        """If _acquire raises, semaphore permit must be released."""
        from lasagna.vector_backends.pool import PooledVectorBackend

        mock_backend = MagicMock()
        mock_backend.backend_name = "test"
        mock_backend.capabilities = MagicMock()

        pool = PooledVectorBackend(factory=lambda: mock_backend, pool_size=2)
        pool.connect()
        pool.disconnect()

        # Both acquires should fail cleanly without leaking permits
        for _ in range(2):
            with pytest.raises(RuntimeError):
                pool._acquire()

        # After re-connecting, both permits should be available
        pool._pool.append(mock_backend)
        pool._pool.append(mock_backend)
        pool._connected = True
        b1 = pool._acquire()
        b2 = pool._acquire()
        pool._release(b1)
        pool._release(b2)


class TestOpenSearchFilterTranslator:
    """Test _OpenSearchTranslator fixes from Round 7."""

    def test_not_match_all_returns_match_none(self):
        """NOT(match_all) = match_none, not match_all."""
        expr = ~FilterExpression.and_()
        result = expr.to_opensearch()
        assert result == {"match_none": {}}

    def test_or_empty_returns_match_none(self):
        """OR of nothing = match_none (identity element of OR is False)."""
        expr = FilterExpression.or_()
        result = expr.to_opensearch()
        assert result == {"match_none": {}}

    def test_like_sql_escaped_percent(self):
        """SQL \\% should produce literal % in OpenSearch, not wildcard."""
        expr = FilterExpression.like("name", "100\\% complete")
        result = expr.to_opensearch()
        # Should be literal "100% complete", NOT "100* complete"
        assert result == {"wildcard": {"name": {"value": "100% complete"}}}

    def test_like_sql_escaped_underscore(self):
        """SQL \\_ should produce literal _, not single-char wildcard."""
        expr = FilterExpression.like("name", "hello\\_world")
        result = expr.to_opensearch()
        assert result == {"wildcard": {"name": {"value": "hello_world"}}}

    def test_like_unescaped_wildcards_translate(self):
        """Unescaped % and _ should translate to * and ?."""
        expr = FilterExpression.like("name", "%hello_world%")
        result = expr.to_opensearch()
        assert result == {"wildcard": {"name": {"value": "*hello?world*"}}}

    def test_like_literal_asterisk_escaped(self):
        """Literal * in pattern should be escaped for OpenSearch."""
        expr = FilterExpression.like("name", "a*b")
        result = expr.to_opensearch()
        assert result == {"wildcard": {"name": {"value": "a\\*b"}}}

    def test_field_validation_allows_hyphens(self):
        """Hyphenated field names should be accepted."""
        expr = FilterExpression.json_eq("metadata", "my-key", "value")
        result = expr.to_opensearch()
        assert result == {"term": {"metadata.my-key": {"value": "value"}}}


class TestInsertFailedIdFiltering:
    """Test that insert() filters out IDs of failed bulk operations."""

    def test_insert_excludes_failed_ids(self, backend, mock_client):
        """IDs of documents that failed bulk insert should not be returned."""

        with patch("lasagna.vector_backends.opensearch_backend.os_bulk") as mock_bulk:
            mock_bulk.return_value = (
                1,  # success count
                [  # errors list
                    {"index": {"_id": "fail_id", "error": {"type": "mapping_error"}}},
                ],
            )
            ids = backend.insert(
                "test_idx",
                [
                    {"chunk_id": "ok_id", "content": "good"},
                    {"chunk_id": "fail_id", "content": "bad"},
                ],
            )
        assert "ok_id" in ids
        assert "fail_id" not in ids


class TestDocIdIntegerZero:
    """Test that integer ID 0 is preserved, not replaced by UUID."""

    def test_integer_zero_id_preserved(self, backend, mock_client):
        """chunk_id=0 (integer) should be kept, not treated as missing."""
        with patch("lasagna.vector_backends.opensearch_backend.os_bulk") as mock_bulk:
            mock_bulk.return_value = (1, [])
            ids = backend.insert("test_idx", [{"chunk_id": 0, "content": "test"}])
        assert ids == [0]


class TestTimeoutEnvParsing:
    """Test OPENSEARCH_TIMEOUT env var error handling."""

    def test_invalid_timeout_uses_default(self):
        """Non-numeric OPENSEARCH_TIMEOUT should fall back to 30."""
        with patch.dict(os.environ, {"OPENSEARCH_TIMEOUT": "30s"}):
            config = _get_default_opensearch_config()
        assert config["timeout"] == 30

    def test_valid_timeout_parsed(self):
        """Numeric OPENSEARCH_TIMEOUT should be parsed correctly."""
        with patch.dict(os.environ, {"OPENSEARCH_TIMEOUT": "60"}):
            config = _get_default_opensearch_config()
        assert config["timeout"] == 60


class TestBuildHybridBody:
    """Test _build_hybrid_body() construction (extracted from hybrid_search)."""

    def test_2way_body_has_two_subqueries(self, backend):
        """Default hybrid body should have 2 sub-queries (knn + bm25)."""
        body = backend._build_hybrid_body(
            query_text="test",
            query_embedding=[0.1] * 10,
            vector_field="embedding",
            sparse_field="content",
            os_filter=None,
            limit=5,
        )
        queries = body["query"]["hybrid"]["queries"]
        assert len(queries) == 2

    def test_3way_body_has_three_subqueries(self, backend):
        """With neural sparse tokens, body should have 3 sub-queries."""
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "true"}):
            body = backend._build_hybrid_body(
                query_text="test",
                query_embedding=[0.1] * 10,
                vector_field="embedding",
                sparse_field="content",
                os_filter=None,
                limit=5,
                query_tokens={"word1": 0.8, "word2": 0.5},
            )
        queries = body["query"]["hybrid"]["queries"]
        assert len(queries) == 3

        # Third sub-query should be neural_sparse
        body_str = json.dumps(queries[1])
        assert "neural_sparse" in body_str
        assert "sparse_tokens" in body_str

    def test_3way_falls_back_to_2way_without_tokens(self, backend):
        """Without tokens, even with flag enabled, should be 2-way."""
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "true"}):
            body = backend._build_hybrid_body(
                query_text="test",
                query_embedding=[0.1] * 10,
                vector_field="embedding",
                sparse_field="content",
                os_filter=None,
                limit=5,
            )
        queries = body["query"]["hybrid"]["queries"]
        assert len(queries) == 2

    def test_3way_disabled_by_default(self, backend):
        """With tokens but flag off, should be 2-way."""
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "false"}):
            body = backend._build_hybrid_body(
                query_text="test",
                query_embedding=[0.1] * 10,
                vector_field="embedding",
                sparse_field="content",
                os_filter=None,
                limit=5,
                query_tokens={"word1": 0.8},
            )
        queries = body["query"]["hybrid"]["queries"]
        assert len(queries) == 2

    def test_3way_sparse_query_has_filter(self, backend):
        """Neural sparse sub-query should include filter when os_filter is set."""
        os_filter = {"term": {"session_id": {"value": "s1"}}}
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "true"}):
            body = backend._build_hybrid_body(
                query_text="test",
                query_embedding=[0.1] * 10,
                vector_field="embedding",
                sparse_field="content",
                os_filter=os_filter,
                limit=5,
                query_tokens={"word1": 0.8},
            )
        queries = body["query"]["hybrid"]["queries"]
        # Sparse sub-query (index 1) should be wrapped in bool with filter
        sparse_query = queries[1]
        assert "bool" in sparse_query
        assert "filter" in sparse_query["bool"]


class TestNeuralSparseFlag:
    """Test _neural_sparse_enabled() flag and related behavior."""

    def test_flag_off_by_default(self):
        """Neural sparse should be disabled by default."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("OPENSEARCH_ENABLE_NEURAL_SPARSE", None)
            with patch.dict(os.environ, env, clear=True):
                assert _neural_sparse_enabled() is False

    def test_flag_on(self):
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "true"}):
            assert _neural_sparse_enabled() is True

    def test_flag_case_insensitive(self):
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "True"}):
            assert _neural_sparse_enabled() is True

    def test_rank_features_mapping_when_enabled(self, backend):
        """Schema mapping should include rank_features when neural sparse enabled."""
        from lasagna.vector_backends.base import FieldSchema, CollectionSchema

        schema = CollectionSchema(
            fields=[FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True)]
        )
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "true"}):
            mapping = backend._schema_to_mapping(schema)
        assert _NEURAL_SPARSE_FIELD in mapping["properties"]
        assert mapping["properties"][_NEURAL_SPARSE_FIELD]["type"] == "rank_features"

    def test_no_rank_features_when_disabled(self, backend):
        """Schema mapping should NOT include rank_features when flag is off."""
        from lasagna.vector_backends.base import FieldSchema, CollectionSchema

        schema = CollectionSchema(
            fields=[FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True)]
        )
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "false"}):
            mapping = backend._schema_to_mapping(schema)
        assert _NEURAL_SPARSE_FIELD not in mapping["properties"]


class TestSparseTokensInsert:
    """Test sparse tokens inclusion in insert payloads."""

    def test_insert_includes_sparse_tokens_when_enabled(self, backend, mock_client):
        """Insert should include sparse_tokens when neural sparse enabled."""

        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "true"}):
            with patch(
                "lasagna.vector_backends.opensearch_backend.os_bulk", return_value=(1, [])
            ) as mock_bulk:
                backend.insert(
                    "idx",
                    {
                        "chunk_id": "c1",
                        "content": "text",
                        "embedding": [0.1] * 10,
                        "sparse_tokens": {"word": 0.8, "test": 0.5},
                    },
                )

                actions = mock_bulk.call_args[0][1]
                source = actions[0]["_source"]
                assert "sparse_tokens" in source
                assert source["sparse_tokens"] == {"word": 0.8, "test": 0.5}

    def test_insert_skips_sparse_tokens_when_disabled(self, backend, mock_client):
        """Insert should NOT include sparse_tokens when flag is off."""
        with patch.dict(os.environ, {"OPENSEARCH_ENABLE_NEURAL_SPARSE": "false"}):
            with patch(
                "lasagna.vector_backends.opensearch_backend.os_bulk", return_value=(1, [])
            ) as mock_bulk:
                backend.insert(
                    "idx",
                    {
                        "chunk_id": "c1",
                        "content": "text",
                        "embedding": [0.1] * 10,
                        "sparse_tokens": {"word": 0.8},
                    },
                )

                actions = mock_bulk.call_args[0][1]
                source = actions[0]["_source"]
                assert "sparse_tokens" not in source


# ============================================================================
# Integration Tests — Full Lifecycle (requires OpenSearch)
# ============================================================================

_INTEGRATION_HOSTS = [
    h.strip()
    for h in os.environ.get("OPENSEARCH_HOSTS", "http://localhost:9200").split(",")
    if h.strip()
]


def _os_available() -> bool:
    """Check if OpenSearch is reachable for integration tests."""
    if not OPENSEARCH_AVAILABLE:
        return False
    try:
        from opensearchpy import OpenSearch

        client = OpenSearch(
            hosts=_INTEGRATION_HOSTS,
            use_ssl=False,
            verify_certs=False,
            timeout=5,
        )
        return client.ping()
    except Exception:
        return False


def _make_test_schema(dim: int = 8):
    """Build a minimal CollectionSchema for integration tests."""
    from lasagna.vector_backends.base import CollectionSchema, FieldSchema

    return CollectionSchema(
        fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=128),
            FieldSchema(name="document_id", dtype="VARCHAR", max_length=128),
            FieldSchema(name="session_id", dtype="VARCHAR", max_length=128),
            FieldSchema(name="content", dtype="VARCHAR", max_length=65535),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=dim),
            FieldSchema(name="metadata", dtype="JSON"),
        ],
        description="Integration test collection",
    )


def _known_embedding(index: int, dim: int = 8) -> list:
    """One-hot-like vector for deterministic similarity."""
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    return vec


# =============================================================================
# Round 8 tests
# =============================================================================


class TestDeleteEmptyFilterGuard:
    """delete() must raise on empty/None filter to prevent data loss."""

    def test_delete_none_filter_raises(self, backend, mock_client):
        """delete(filter_expr=None) must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot delete with empty filter"):
            backend.delete("test_idx", None)

    def test_delete_empty_filter_raises(self, backend, mock_client):
        """delete(filter_expr=empty) must raise ValueError."""
        empty = FilterExpression(operator=FilterOperator.EQ)  # No field/value → is_empty()
        with pytest.raises(ValueError, match="Cannot delete with empty filter"):
            backend.delete("test_idx", empty)

    def test_delete_valid_filter_works(self, backend, mock_client):
        """Normal delete with a real filter should succeed."""
        mock_client.delete_by_query.return_value = {"deleted": 3}
        result = backend.delete(
            "test_idx",
            FilterExpression.eq("session_id", "s1"),
        )
        assert result == 3
        mock_client.delete_by_query.assert_called_once()


class TestPartitionKeyEmptyString:
    """partition_key='' must not silently bypass tenant isolation."""

    def test_empty_string_partition_key_applied_in_search(self, backend, mock_client):
        """partition_key='' should be applied as a filter (not skipped)."""
        mock_client.search.return_value = {"hits": {"hits": [], "total": {"value": 0}}}
        backend.search(
            "test_idx",
            [0.1] * 768,
            partition_key="",
        )
        body = mock_client.search.call_args[1].get("body") or mock_client.search.call_args[0][0]
        # Partition key should be present in the query even when empty string
        body_str = json.dumps(body)
        assert "partition_key" in body_str

    def test_empty_string_partition_key_applied_in_insert(self, backend, mock_client):
        """partition_key='' should be stored in the document."""
        with patch("lasagna.vector_backends.opensearch_backend.os_bulk") as mock_bulk:
            mock_bulk.return_value = (1, [])
            backend.insert(
                "test_idx",
                [{"chunk_id": "c1", "content": "test", "embedding": [0.1] * 768}],
                partition_key="",
            )
            actions = mock_bulk.call_args[0][1]
            source = actions[0]["_source"]
            assert "partition_key" in source
            assert source["partition_key"] == ""


class TestCosineNormalization:
    """normalize_score for COSINE must be monotonic (no discontinuity at 0)."""

    @staticmethod
    def _normalize(raw_score):
        from lasagna.vector_backends.base import MetricType

        backend = OpenSearchBackend.__new__(OpenSearchBackend)
        return backend.normalize_score(raw_score, MetricType.COSINE, is_distance=False)

    def test_negative_score_maps_below_half(self):
        """Cosine similarity -0.5 → 0.25 (below 0.5)."""
        result = self._normalize(-0.5)
        assert abs(result - 0.25) < 1e-9

    def test_zero_score_maps_to_half(self):
        """Cosine similarity 0.0 → 0.5 (midpoint, not 0.0)."""
        result = self._normalize(0.0)
        assert abs(result - 0.5) < 1e-9

    def test_positive_score_maps_above_half(self):
        """Cosine similarity 0.8 → 0.9."""
        result = self._normalize(0.8)
        assert abs(result - 0.9) < 1e-9

    def test_monotonicity_across_zero(self):
        """Scores must be monotonically increasing across the zero boundary."""
        scores = [-1.0, -0.5, -0.001, 0.0, 0.001, 0.5, 1.0]
        normalized = [self._normalize(s) for s in scores]
        for i in range(1, len(normalized)):
            assert normalized[i] > normalized[i - 1], (
                f"Non-monotonic: normalize({scores[i - 1]})={normalized[i - 1]} >= "
                f"normalize({scores[i]})={normalized[i]}"
            )


class TestFieldRegexDotTraversal:
    """OpenSearch field regex must reject dots to prevent nested traversal."""

    def test_dotted_field_rejected(self):
        """Field 'payload.admin' must be rejected by OpenSearch translator."""
        expr = FilterExpression.eq("payload.admin", True)
        with pytest.raises(ValueError, match="Invalid field name"):
            expr.to_opensearch()

    def test_hyphenated_field_accepted(self):
        """Field 'my-field' must be accepted."""
        expr = FilterExpression.eq("my-field", "value")
        result = expr.to_opensearch()
        assert "my-field" in json.dumps(result)


class TestPoolPrepareIndexParams:
    """PooledVectorBackend must delegate prepare_index_params."""

    def test_prepare_index_params_delegated(self):
        """Pool should forward prepare_index_params to underlying backend."""
        from lasagna.vector_backends.pool import PooledVectorBackend

        mock_backend = MagicMock()
        mock_backend.connect.return_value = None
        mock_backend.backend_name = "test"
        mock_backend.capabilities = MagicMock()
        mock_backend.prepare_index_params.return_value = {"custom": True}

        pool = PooledVectorBackend(factory=lambda: mock_backend, pool_size=1)
        pool.connect()
        result = pool.prepare_index_params()
        assert result == {"custom": True}
        mock_backend.prepare_index_params.assert_called_once()
        pool.disconnect()


@pytest.mark.integration
@pytest.mark.skipif(not _os_available(), reason="OpenSearch not reachable")
class TestOpenSearchIntegration:
    """Integration tests — require a running OpenSearch instance.

    Start OpenSearch:
        docker compose -f docker-compose.yml -f backends/opensearch.yml up -d

    Run these tests:
        pytest tests/test_opensearch_backend.py -v -m integration
    """

    COLLECTION = f"test_os_{uuid.uuid4().hex[:8]}"
    DIM = 8

    @pytest.fixture(autouse=True)
    def _backend(self):
        """Provide a connected OpenSearchBackend, drop test collection on teardown."""
        self._b = OpenSearchBackend(
            hosts=_INTEGRATION_HOSTS,
            use_ssl=False,
            verify_certs=False,
            timeout=10,
        )
        self._b.connect()
        yield self._b
        try:
            self._b.drop_collection(self.COLLECTION)
        except Exception:
            pass
        self._b.disconnect()

    def _create(self):
        self._b.create_collection(self.COLLECTION, _make_test_schema(self.DIM))

    def _insert_sample_docs(self):
        """Insert a standard set of 5 sample documents."""
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "quantum entanglement is a fascinating phenomenon in physics",
                    "embedding": _known_embedding(0, self.DIM),
                    "metadata": {},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "classical mechanics describes macroscopic objects and motion",
                    "embedding": _known_embedding(1, self.DIM),
                    "metadata": {},
                },
                {
                    "chunk_id": "c3",
                    "document_id": "d2",
                    "session_id": "s1",
                    "content": "machine learning algorithms can classify images accurately",
                    "embedding": _known_embedding(2, self.DIM),
                    "metadata": {},
                },
                {
                    "chunk_id": "c4",
                    "document_id": "d2",
                    "session_id": "s2",
                    "content": "deep learning neural networks use backpropagation for training",
                    "embedding": _known_embedding(3, self.DIM),
                    "metadata": {},
                },
                {
                    "chunk_id": "c5",
                    "document_id": "d3",
                    "session_id": "s1",
                    "content": "quantum computing may revolutionize cryptography and security",
                    "embedding": _known_embedding(4, self.DIM),
                    "metadata": {"page": 1, "author": "alice"},
                },
            ],
        )

    # --- A. Collection Management ---

    def test_create_and_has_collection(self):
        assert not self._b.has_collection(self.COLLECTION)
        self._create()
        assert self._b.has_collection(self.COLLECTION)

    def test_describe_collection(self):
        self._create()
        info = self._b.describe_collection(self.COLLECTION)
        assert info.name == self.COLLECTION
        field_names = [f["name"] for f in info.fields]
        assert "chunk_id" in field_names
        assert "embedding" in field_names
        assert "content" in field_names

    def test_drop_collection(self):
        self._create()
        assert self._b.has_collection(self.COLLECTION)
        self._b.drop_collection(self.COLLECTION)
        assert not self._b.has_collection(self.COLLECTION)

    def test_create_collection_idempotent(self):
        self._create()
        self._create()  # Second call should not raise
        assert self._b.has_collection(self.COLLECTION)

    # --- B. Insert & Search ---

    def test_insert_and_search(self):
        self._create()
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "hello world",
                    "embedding": [1, 0, 0, 0, 0, 0, 0, 0],
                    "metadata": {},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "goodbye world",
                    "embedding": [0, 1, 0, 0, 0, 0, 0, 0],
                    "metadata": {},
                },
            ],
        )
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=[1, 0, 0, 0, 0, 0, 0, 0],
            limit=2,
        )
        assert len(results) == 2
        assert results[0].id == "c1"
        assert results[0].score > results[1].score

    def test_insert_returns_ids(self):
        self._create()
        ids = self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "text",
                    "embedding": _known_embedding(0, self.DIM),
                    "metadata": {},
                },
            ],
        )
        assert ids == ["c1"]

    def test_search_limit_respected(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(0, self.DIM),
            limit=1,
        )
        assert len(results) == 1

    def test_search_empty_collection(self):
        self._create()
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(0, self.DIM),
            limit=5,
        )
        assert results == []

    def test_search_returns_metadata(self):
        self._create()
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "text",
                    "embedding": _known_embedding(0, self.DIM),
                    "metadata": {"page": 42, "author": "bob"},
                },
            ],
        )
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(0, self.DIM),
            limit=1,
        )
        assert len(results) == 1
        assert results[0].entity["metadata"]["page"] == 42
        assert results[0].entity["metadata"]["author"] == "bob"

    # --- C. Upsert ---

    def test_upsert_overwrites_existing(self):
        self._create()
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "original",
                    "embedding": _known_embedding(0, self.DIM),
                    "metadata": {},
                },
            ],
        )
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "updated",
                    "embedding": _known_embedding(1, self.DIM),
                    "metadata": {},
                },
            ],
        )
        stats = self._b.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 1

    def test_upsert_preserves_other_docs(self):
        self._create()
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "keep",
                    "embedding": _known_embedding(0, self.DIM),
                    "metadata": {},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "also keep",
                    "embedding": _known_embedding(1, self.DIM),
                    "metadata": {},
                },
            ],
        )
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "replaced",
                    "embedding": _known_embedding(2, self.DIM),
                    "metadata": {},
                },
            ],
        )
        stats = self._b.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 2

    # --- D. Delete ---

    def test_delete_by_filter(self):
        self._create()
        self._insert_sample_docs()
        deleted = self._b.delete(
            self.COLLECTION,
            FilterExpression.eq("document_id", "d2"),
        )
        assert deleted == 2  # c3 and c4
        stats = self._b.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 3

    def test_delete_nonexistent(self):
        self._create()
        self._insert_sample_docs()
        deleted = self._b.delete(
            self.COLLECTION,
            FilterExpression.eq("document_id", "nonexistent"),
        )
        assert deleted == 0

    def test_delete_compound_filter(self):
        self._create()
        self._insert_sample_docs()
        # Delete only docs with session_id=s1 AND document_id=d1
        expr = FilterExpression.eq("session_id", "s1") & FilterExpression.eq("document_id", "d1")
        deleted = self._b.delete(self.COLLECTION, expr)
        assert deleted == 2  # c1, c2
        stats = self._b.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 3

    # --- E. Filter Expressions ---

    def test_search_with_eq_filter(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(0, self.DIM),
            filter_expr=FilterExpression.eq("session_id", "s1"),
            limit=10,
        )
        for r in results:
            assert r.entity["session_id"] == "s1"
        # c4 has session_id="s2", should be excluded
        result_ids = {r.id for r in results}
        assert "c4" not in result_ids

    def test_search_with_ne_filter(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(3, self.DIM),
            filter_expr=FilterExpression.ne("session_id", "s2"),
            limit=10,
        )
        result_ids = {r.id for r in results}
        assert "c4" not in result_ids

    def test_query_with_json_eq_filter(self):
        self._create()
        self._insert_sample_docs()
        rows = self._b.query(
            self.COLLECTION,
            filter_expr=FilterExpression.json_eq("metadata", "author", "alice"),
        )
        assert len(rows) == 1
        assert rows[0]["id"] == "c5"

    def test_search_with_compound_filter(self):
        self._create()
        self._insert_sample_docs()
        expr = FilterExpression.eq("session_id", "s1") & FilterExpression.eq("document_id", "d1")
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(0, self.DIM),
            filter_expr=expr,
            limit=10,
        )
        result_ids = {r.id for r in results}
        assert result_ids == {"c1", "c2"}

    # --- F. Hybrid Search ---

    def test_hybrid_search_basic(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum entanglement",
            query_embedding=_known_embedding(0, self.DIM),
            limit=3,
        )
        assert len(results) >= 1
        # c1 matches both text ("quantum entanglement") and embedding
        assert results[0].id == "c1"

    def test_hybrid_search_with_filter(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="learning",
            query_embedding=_known_embedding(2, self.DIM),
            filter_expr=FilterExpression.eq("session_id", "s1"),
            limit=5,
        )
        for r in results:
            assert r.entity["session_id"] == "s1"
        result_ids = {r.id for r in results}
        assert "c4" not in result_ids  # c4 has session_id=s2

    def test_hybrid_search_alpha_zero_pure_bm25(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum computing cryptography",
            query_embedding=_known_embedding(7, self.DIM),  # Irrelevant embedding
            alpha=0.0,
            limit=3,
        )
        assert len(results) >= 1
        # BM25 should rank text-relevant results; c5 has all three terms
        assert results[0].id == "c5"

    def test_hybrid_search_alpha_one_pure_vector(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="completely irrelevant text",
            query_embedding=_known_embedding(0, self.DIM),
            alpha=1.0,
            limit=2,
        )
        assert len(results) >= 1
        # Pure vector: c1 has embedding matching index 0
        assert results[0].id == "c1"

    def test_hybrid_search_fuzzy_matching(self):
        self._create()
        self._insert_sample_docs()
        # "quantm" is a typo for "quantum", fuzziness=AUTO should handle it
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantm entanglement",
            query_embedding=_known_embedding(0, self.DIM),
            limit=3,
        )
        assert len(results) >= 1
        result_ids = {r.id for r in results}
        assert "c1" in result_ids

    def test_hybrid_search_output_fields(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum",
            query_embedding=_known_embedding(0, self.DIM),
            output_fields=["content", "document_id"],
            limit=2,
        )
        assert len(results) >= 1
        # Output fields should limit what's in entity
        entity = results[0].entity
        assert "content" in entity
        assert "document_id" in entity

    # --- G. Highlights ---

    def test_bm25_highlights_returned(self):
        """Pure BM25 (alpha=0.0) returns highlights on OpenSearch 2.x."""
        self._create()
        self._insert_sample_docs()
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum entanglement",
            query_embedding=_known_embedding(0, self.DIM),
            alpha=0.0,
            limit=3,
        )
        assert len(results) >= 1
        # At least the top result should have highlights
        assert "highlights" in results[0].entity
        assert len(results[0].entity["highlights"]) >= 1

    def test_bm25_highlights_contain_query_terms(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum",
            query_embedding=_known_embedding(0, self.DIM),
            alpha=0.0,
            limit=3,
        )
        assert len(results) >= 1
        highlights = results[0].entity.get("highlights", [])
        assert any("quantum" in h.lower() for h in highlights)

    def test_no_highlights_in_pure_vector_search(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(0, self.DIM),
            limit=2,
        )
        assert len(results) >= 1
        assert "highlights" not in results[0].entity

    # --- H. Multi-Search _msearch ---

    def test_multi_search_returns_per_query_results(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.multi_search(
            self.COLLECTION,
            [
                {
                    "size": 2,
                    "query": {
                        "knn": {"embedding": {"vector": _known_embedding(0, self.DIM), "k": 2}}
                    },
                },
                {
                    "size": 2,
                    "query": {
                        "knn": {"embedding": {"vector": _known_embedding(2, self.DIM), "k": 2}}
                    },
                },
            ],
        )
        assert len(results) == 2
        # First query should return c1 as top result
        assert results[0][0].id == "c1"
        # Second query should return c3 as top result
        assert results[1][0].id == "c3"

    def test_multi_search_result_count(self):
        self._create()
        self._insert_sample_docs()
        queries = [
            {
                "size": 1,
                "query": {"knn": {"embedding": {"vector": _known_embedding(i, self.DIM), "k": 1}}},
            }
            for i in range(3)
        ]
        results = self._b.multi_search(self.COLLECTION, queries)
        assert len(results) == 3

    def test_multi_hybrid_search_batch(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.multi_hybrid_search(
            collection_name=self.COLLECTION,
            query_texts=["quantum entanglement", "machine learning"],
            query_embeddings=[_known_embedding(0, self.DIM), _known_embedding(2, self.DIM)],
            limit=2,
        )
        assert len(results) == 2
        assert results[0][0].id == "c1"  # quantum + embedding(0)
        assert results[1][0].id == "c3"  # ML + embedding(2)

    def test_multi_hybrid_search_with_filter(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.multi_hybrid_search(
            collection_name=self.COLLECTION,
            query_texts=["learning", "quantum"],
            query_embeddings=[_known_embedding(2, self.DIM), _known_embedding(0, self.DIM)],
            filter_expr=FilterExpression.eq("session_id", "s1"),
            limit=5,
        )
        assert len(results) == 2
        for result_list in results:
            for r in result_list:
                assert r.entity["session_id"] == "s1"

    # --- I. Pipeline Management ---

    def test_pipeline_recreated_after_cache_clear(self):
        """Pipeline cache is repopulated on next hybrid_search after being cleared."""
        self._create()
        self._insert_sample_docs()
        # create_collection pre-creates the pipeline; clear cache to simulate cold start
        self._b._pipeline_cache.clear()
        assert self.COLLECTION not in self._b._pipeline_cache
        self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum",
            query_embedding=_known_embedding(0, self.DIM),
            limit=1,
        )
        assert self.COLLECTION in self._b._pipeline_cache

    def test_pipeline_updates_on_alpha_change(self):
        self._create()
        self._insert_sample_docs()
        self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum",
            query_embedding=_known_embedding(0, self.DIM),
            alpha=0.7,
            limit=1,
        )
        pipeline_a = self._b._pipeline_cache[self.COLLECTION]
        assert "a0.70" in pipeline_a

        self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum",
            query_embedding=_known_embedding(0, self.DIM),
            alpha=0.5,
            limit=1,
        )
        pipeline_b = self._b._pipeline_cache[self.COLLECTION]
        assert "a0.50" in pipeline_b
        assert pipeline_a != pipeline_b

    def test_drop_collection_clears_pipeline(self):
        self._create()
        self._insert_sample_docs()
        self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="test",
            query_embedding=_known_embedding(0, self.DIM),
            limit=1,
        )
        assert self.COLLECTION in self._b._pipeline_cache
        self._b.drop_collection(self.COLLECTION)
        assert self.COLLECTION not in self._b._pipeline_cache

    # --- J. Index Settings ---

    def test_index_has_knn_enabled(self):
        self._create()
        client = self._b._client
        settings = client.indices.get_settings(index=self.COLLECTION)
        idx_settings = settings[self.COLLECTION]["settings"]["index"]
        assert idx_settings.get("knn") == "true" or idx_settings.get("knn") is True

    def test_index_mapping_types(self):
        self._create()
        client = self._b._client
        mapping = client.indices.get_mapping(index=self.COLLECTION)
        props = mapping[self.COLLECTION]["mappings"]["properties"]
        assert props["embedding"]["type"] == "knn_vector"
        assert props["chunk_id"]["type"] == "keyword"
        assert props["document_id"]["type"] == "keyword"
        assert props["content"]["type"] == "text"

    # --- K. Query (non-vector) ---

    def test_query_by_filter(self):
        self._create()
        self._insert_sample_docs()
        rows = self._b.query(
            self.COLLECTION,
            filter_expr=FilterExpression.eq("session_id", "s1"),
        )
        assert len(rows) == 4  # c1, c2, c3, c5

    def test_query_with_output_fields(self):
        self._create()
        self._insert_sample_docs()
        rows = self._b.query(
            self.COLLECTION,
            filter_expr=FilterExpression.eq("chunk_id", "c1"),
            output_fields=["content", "document_id"],
        )
        assert len(rows) == 1
        assert "content" in rows[0]
        assert "document_id" in rows[0]

    # --- L. Collection Stats ---

    def test_collection_stats_row_count(self):
        self._create()
        self._insert_sample_docs()
        stats = self._b.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 5

    def test_collection_stats_has_size(self):
        self._create()
        self._insert_sample_docs()
        stats = self._b.get_collection_stats(self.COLLECTION)
        assert "index_size_bytes" in stats
        assert stats["index_size_bytes"] >= 0

    # --- M. Edge Cases ---

    def test_large_batch_insert(self):
        self._create()
        docs = [
            {
                "chunk_id": f"bulk_{i}",
                "document_id": "d_bulk",
                "session_id": "s1",
                "content": f"document number {i} for batch insert test",
                "embedding": _known_embedding(i % self.DIM, self.DIM),
                "metadata": {"idx": i},
            }
            for i in range(200)
        ]
        self._b.insert(self.COLLECTION, docs)
        stats = self._b.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 200

    def test_special_characters_in_content(self):
        self._create()
        content = 'Unicode: \u00e9\u00e0\u00fc\u00f1 — "quotes" & <tags> \n newline \t tab'
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "special",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": content,
                    "embedding": _known_embedding(0, self.DIM),
                    "metadata": {},
                },
            ],
        )
        rows = self._b.query(
            self.COLLECTION,
            filter_expr=FilterExpression.eq("chunk_id", "special"),
        )
        assert len(rows) == 1
        assert rows[0]["content"] == content

    def test_special_characters_in_query(self):
        self._create()
        self._insert_sample_docs()
        # Should not raise even with special characters
        results = self._b.hybrid_search(
            collection_name=self.COLLECTION,
            query_text='test "quotes" & <special> chars!',
            query_embedding=_known_embedding(0, self.DIM),
            limit=2,
        )
        assert isinstance(results, list)

    def test_search_zero_limit(self):
        self._create()
        self._insert_sample_docs()
        results = self._b.search(
            collection_name=self.COLLECTION,
            query_embedding=_known_embedding(0, self.DIM),
            limit=0,
        )
        assert results == []

    # --- N. Connection & Capabilities ---

    def test_connect_disconnect_reconnect(self):
        # Already connected by fixture
        assert self._b.is_connected()
        self._b.disconnect()
        assert not self._b.is_connected()
        self._b.connect()
        assert self._b.is_connected()

    def test_capabilities_integration(self):
        caps = self._b.capabilities
        assert caps.supports_hybrid_search is True
        assert caps.supports_native_bm25 is True
        assert caps.supports_hnsw is True

    def test_backend_name(self):
        assert self._b.backend_name == "opensearch"


@pytest.mark.integration
@pytest.mark.skipif(
    not _os_available() or not (_neural_sparse_enabled and _neural_sparse_enabled()),
    reason="OpenSearch not reachable or neural sparse not enabled",
)
class TestOpenSearchNeuralSparseIntegration:
    """Integration tests for 3-way hybrid search.

    Requires:
        OPENSEARCH_ENABLE_NEURAL_SPARSE=true
        OpenSearch running on localhost:9200
    """

    COLLECTION = f"test_ns_{uuid.uuid4().hex[:8]}"
    DIM = 8

    @pytest.fixture(autouse=True)
    def _backend(self):
        self._b = OpenSearchBackend(
            hosts=_INTEGRATION_HOSTS,
            use_ssl=False,
            verify_certs=False,
            timeout=10,
        )
        self._b.connect()
        yield self._b
        try:
            self._b.drop_collection(self.COLLECTION)
        except Exception:
            pass
        self._b.disconnect()

    def _create(self):
        self._b.create_collection(self.COLLECTION, _make_test_schema(self.DIM))

    def test_3way_index_has_rank_features_mapping(self):
        self._create()
        client = self._b._client
        mapping = client.indices.get_mapping(index=self.COLLECTION)
        props = mapping[self.COLLECTION]["mappings"]["properties"]
        assert "sparse_tokens" in props
        assert props["sparse_tokens"]["type"] == "rank_features"

    def test_3way_sparse_tokens_stored(self):
        self._create()
        self._b.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "test text",
                    "embedding": _known_embedding(0, self.DIM),
                    "metadata": {},
                    "sparse_tokens": {"test": 0.9, "text": 0.8},
                },
            ],
        )
        rows = self._b.query(
            self.COLLECTION,
            filter_expr=FilterExpression.eq("chunk_id", "c1"),
        )
        assert len(rows) == 1

    def test_3way_pipeline_has_three_weights(self):
        self._create()
        pipeline_name = self._b._ensure_hybrid_pipeline(self.COLLECTION, 0.7, n_queries=3)
        assert "3w" in pipeline_name
        # Verify pipeline exists on server
        client = self._b._client
        resp = client.transport.perform_request("GET", f"/_search/pipeline/{pipeline_name}")
        weights = resp[pipeline_name]["phase_results_processors"][0]["normalization-processor"][
            "combination"
        ]["parameters"]["weights"]
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.01


# =============================================================================
# Round 9: Production Readiness Fixes
# =============================================================================


class TestInsertEmptyDataGuard:
    """insert() with an empty list should return [] immediately."""

    def test_empty_list_returns_empty(self):
        backend = OpenSearchBackend.__new__(OpenSearchBackend)
        # Should return [] without touching the client
        result = backend.insert("test_index", [])
        assert result == []

    def test_empty_after_dict_wrap(self):
        """Single dict is wrapped to list first — but a truly empty list is caught."""
        backend = OpenSearchBackend.__new__(OpenSearchBackend)
        result = backend.insert("test_index", [])
        assert result == []


class TestInsertNullEmbeddingWarning:
    """insert() should log a warning when documents have null embeddings."""

    def test_null_embedding_warning(self):
        backend = _make_connected_backend()
        backend._client.indices.exists.return_value = True

        # Mock os_bulk to succeed
        with patch("lasagna.vector_backends.opensearch_backend.os_bulk") as mock_bulk:
            mock_bulk.return_value = (2, [])

            import logging

            with patch.object(
                logging.getLogger("lasagna.vector_backends.opensearch_backend"), "warning"
            ) as mock_warn:
                backend.insert(
                    "test_index",
                    [
                        {"chunk_id": "c1", "content": "text1"},  # No embedding
                        {"chunk_id": "c2", "content": "text2", "embedding": [0.1, 0.2]},
                    ],
                )
                # Should have warned about null embedding
                mock_warn.assert_called_once()
                warning_msg = mock_warn.call_args[0][0]
                assert "1/2 documents without embeddings" in warning_msg
                assert "invisible to vector search" in warning_msg


class TestPartitionKeyDeleteQuery:
    """delete() and query() must use `is not None` for partition_key."""

    def test_delete_with_empty_string_partition_key(self):
        """Empty string partition_key must be applied (not skipped)."""
        backend = _make_connected_backend()
        backend._client.delete_by_query.return_value = {"deleted": 1}
        fe = FilterExpression.eq("session_id", "s1")

        backend.delete("test_index", fe, partition_key="")

        call_body = backend._client.delete_by_query.call_args
        query = call_body[1]["body"]["query"]
        # Should have a bool/must with partition_key term
        assert "bool" in query
        pk_terms = [
            clause
            for clause in query["bool"]["must"]
            if "term" in clause and "partition_key" in clause["term"]
        ]
        assert len(pk_terms) == 1
        assert pk_terms[0]["term"]["partition_key"]["value"] == ""

    def test_query_with_empty_string_partition_key(self):
        """Empty string partition_key must be applied in query()."""
        backend = _make_connected_backend()
        backend._client.search.return_value = {"hits": {"hits": []}}

        backend.query("test_index", partition_key="")

        call_body = backend._client.search.call_args[1]["body"]
        query = call_body["query"]
        assert "bool" in query
        pk_terms = [
            clause
            for clause in query["bool"]["must"]
            if "term" in clause and "partition_key" in clause["term"]
        ]
        assert len(pk_terms) == 1


class TestNotMatchNoneOptimization:
    """NOT(match_none) should produce match_all in OpenSearch translator."""

    def test_not_match_none_returns_match_all(self):
        """NOT(OR()) = NOT(match_none) = match_all."""
        expr = ~FilterExpression.or_()  # NOT(OR-of-nothing)
        result = expr.to_opensearch()
        assert result == {"match_all": {}}

    def test_not_match_all_returns_match_none(self):
        """NOT(AND()) = NOT(match_all) = match_none (regression check)."""
        expr = ~FilterExpression.and_()  # NOT(AND-of-nothing)
        result = expr.to_opensearch()
        assert result == {"match_none": {}}

    def test_double_negation_match_none(self):
        """NOT(NOT(OR())) = NOT(match_all) = match_none."""
        inner = ~FilterExpression.or_()  # match_all
        outer = ~inner  # NOT(match_all) = match_none
        result = outer.to_opensearch()
        assert result == {"match_none": {}}


class TestStaleErrorMessage:
    """The field validation error message should reference the current regex."""

    def test_error_message_mentions_no_dots(self):
        """Error message should mention dots are not allowed."""
        with pytest.raises(ValueError) as exc_info:
            FilterExpression.eq("metadata.session_id", "val").to_opensearch()
        msg = str(exc_info.value)
        assert "dots not allowed" in msg
        assert "JSON_EQ" in msg
