"""Verify adapter classes conform to their ABCs.

Tests cover:
- Subclass relationships (issubclass, not isinstance, to avoid model loading)
- Method signature compatibility (abstract methods are implemented)
- Chunk dataclass consolidation (base + semantic subclass)
- Conversion utility correctness including edge cases
"""

import inspect


# =========================================================================
# Adapter Subclass Conformance
# =========================================================================


def test_reranker_adapters_are_subclass():
    from lasagna.core.base_components import BaseReranker
    from lasagna.retrieval.search.ranker import (
        CrossEncoderRerankerAdapter,
        CombinedRerankerAdapter,
        EnsembleRerankerAdapter,
        MMRRerankerAdapter,
    )

    assert issubclass(CrossEncoderRerankerAdapter, BaseReranker)
    assert issubclass(CombinedRerankerAdapter, BaseReranker)
    assert issubclass(EnsembleRerankerAdapter, BaseReranker)
    assert issubclass(MMRRerankerAdapter, BaseReranker)


def test_chunker_adapters_are_subclass():
    from lasagna.core.base_components import BaseChunker
    from lasagna.processing.chunker.semantic import (
        SemanticChunkerAdapter,
        HierarchicalChunkerAdapter,
        StructureAwareChunkerAdapter,
    )

    assert issubclass(SemanticChunkerAdapter, BaseChunker)
    assert issubclass(HierarchicalChunkerAdapter, BaseChunker)
    assert issubclass(StructureAwareChunkerAdapter, BaseChunker)


def test_embedder_adapter_is_subclass():
    from lasagna.core.base_components import BaseEmbedder
    from lasagna.infrastructure.embedding.manager import (
        EmbeddingManagerAdapter,
    )

    assert issubclass(EmbeddingManagerAdapter, BaseEmbedder)


def test_query_enhancer_adapter_is_subclass():
    from lasagna.core.base_components import BaseQueryEnhancer
    from lasagna.retrieval.enhancement.enhancer import (
        QueryEnhancerAdapter,
    )

    assert issubclass(QueryEnhancerAdapter, BaseQueryEnhancer)


def test_document_processor_adapter_is_subclass():
    from lasagna.core.base_components import BaseDocumentProcessor
    from lasagna.processing.document_processor import (
        DocumentProcessorAdapter,
    )

    assert issubclass(DocumentProcessorAdapter, BaseDocumentProcessor)


# =========================================================================
# Method Signature Compatibility
# =========================================================================


def _has_method(cls, method_name: str) -> bool:
    """Check that a class defines a concrete (non-abstract) method."""
    method = getattr(cls, method_name, None)
    if method is None:
        return False
    return not getattr(method, "__isabstractmethod__", False)


def test_reranker_adapters_implement_rerank():
    """All reranker adapters must implement rerank(query, documents, top_k)."""
    from lasagna.retrieval.search.ranker import (
        CrossEncoderRerankerAdapter,
        CombinedRerankerAdapter,
        EnsembleRerankerAdapter,
        MMRRerankerAdapter,
    )

    for cls in [
        CrossEncoderRerankerAdapter,
        CombinedRerankerAdapter,
        EnsembleRerankerAdapter,
        MMRRerankerAdapter,
    ]:
        assert _has_method(cls, "rerank"), f"{cls.__name__} missing rerank()"
        sig = inspect.signature(cls.rerank)
        params = list(sig.parameters.keys())
        assert "query" in params, f"{cls.__name__}.rerank missing 'query' param"
        assert "documents" in params, f"{cls.__name__}.rerank missing 'documents' param"
        assert "top_k" in params, f"{cls.__name__}.rerank missing 'top_k' param"


def test_chunker_adapters_implement_chunk():
    """All chunker adapters must implement chunk(text)."""
    from lasagna.processing.chunker.semantic import (
        SemanticChunkerAdapter,
        HierarchicalChunkerAdapter,
        StructureAwareChunkerAdapter,
    )

    for cls in [SemanticChunkerAdapter, HierarchicalChunkerAdapter, StructureAwareChunkerAdapter]:
        assert _has_method(cls, "chunk"), f"{cls.__name__} missing chunk()"
        sig = inspect.signature(cls.chunk)
        assert "text" in sig.parameters, f"{cls.__name__}.chunk missing 'text' param"


def test_embedder_adapter_implements_encode():
    from lasagna.infrastructure.embedding.manager import (
        EmbeddingManagerAdapter,
    )

    assert _has_method(EmbeddingManagerAdapter, "encode")
    assert _has_method(EmbeddingManagerAdapter, "encode_query")


def test_query_enhancer_adapter_implements_enhance():
    from lasagna.retrieval.enhancement.enhancer import (
        QueryEnhancerAdapter,
    )

    assert _has_method(QueryEnhancerAdapter, "enhance")
    sig = inspect.signature(QueryEnhancerAdapter.enhance)
    assert "query" in sig.parameters


def test_document_processor_adapter_implements_extract():
    from lasagna.processing.document_processor import (
        DocumentProcessorAdapter,
    )

    assert _has_method(DocumentProcessorAdapter, "extract")
    sig = inspect.signature(DocumentProcessorAdapter.extract)
    assert "file_path" in sig.parameters


# =========================================================================
# Chunk Dataclass Consolidation
# =========================================================================


def test_semantic_chunk_inherits_from_base_chunk():
    """semantic.py Chunk is a subclass of base_components.Chunk."""
    from lasagna.core.base_components import Chunk as BaseChunk
    from lasagna.processing.chunker.semantic import Chunk as SemanticChunk

    assert issubclass(SemanticChunk, BaseChunk)


def test_base_chunk_defaults():
    """Chunk(content=...) works with all defaults — no required fields except content."""
    from lasagna.core.base_components import Chunk

    chunk = Chunk(content="hello world")
    assert chunk.content == "hello world"
    assert chunk.chunk_index == 0
    assert chunk.metadata == {}
    assert chunk.page_number is None
    assert chunk.start_char == 0
    assert chunk.end_char == 0
    assert chunk.section_header is None
    assert chunk.parent_chunk_id is None
    assert chunk.chunk_type == "text"


def test_base_chunk_has_all_fields():
    """The canonical Chunk has all 9 fields from the consolidated definition."""
    from lasagna.core.base_components import Chunk

    chunk = Chunk(
        content="test",
        chunk_index=3,
        start_char=100,
        end_char=200,
        page_number=2,
        section_header="Introduction",
        parent_chunk_id="abc123",
        chunk_type="table",
        metadata={"key": "value"},
    )
    assert chunk.content == "test"
    assert chunk.chunk_index == 3
    assert chunk.start_char == 100
    assert chunk.end_char == 200
    assert chunk.page_number == 2
    assert chunk.section_header == "Introduction"
    assert chunk.parent_chunk_id == "abc123"
    assert chunk.chunk_type == "table"
    assert chunk.metadata == {"key": "value"}


# =========================================================================
# Conversion Utility Correctness
# =========================================================================


class TestScoredDocumentFromDict:
    """Thorough tests for scored_document_from_dict edge cases."""

    def test_happy_path(self):
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict(
            {
                "content": "test content",
                "score": 0.95,
                "document_id": "doc1",
                "id": "chunk1",
                "metadata": {"key": "val"},
            }
        )
        assert sd.content == "test content"
        assert sd.score == 0.95
        assert sd.document_id == "doc1"
        assert sd.chunk_id == "chunk1"
        assert sd.metadata == {"key": "val"}

    def test_score_zero_is_preserved(self):
        """A score of 0.0 must NOT fall through to other keys."""
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict(
            {
                "content": "test",
                "score": 0.0,
                "distance": 0.99,  # should NOT be used
            }
        )
        assert sd.score == 0.0

    def test_score_zero_distance(self):
        """distance=0.0 when score is absent should be preserved."""
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict(
            {
                "content": "test",
                "distance": 0.0,
                "combined_score": 0.5,  # should NOT be used
            }
        )
        assert sd.score == 0.0

    def test_empty_content_is_preserved(self):
        """An empty string content must NOT fall through to str(d)."""
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict(
            {
                "content": "",
                "score": 0.5,
            }
        )
        assert sd.content == ""

    def test_entity_content_fallback(self):
        """Falls back to entity.content when top-level content is absent."""
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict(
            {
                "entity": {"content": "nested"},
                "score": 0.5,
            }
        )
        assert sd.content == "nested"

    def test_str_fallback_when_no_content(self):
        """Falls back to str(d) when no content field exists at all."""
        from lasagna.core.base_components import scored_document_from_dict

        d = {"score": 0.5, "id": "x"}
        sd = scored_document_from_dict(d)
        assert sd.content == str(d)

    def test_missing_metadata_defaults_to_empty_dict(self):
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict({"content": "test", "score": 0.5})
        assert sd.metadata == {}

    def test_chunk_id_prefers_id_over_chunk_id(self):
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict(
            {
                "content": "test",
                "score": 0.5,
                "id": "preferred",
                "chunk_id": "fallback",
            }
        )
        assert sd.chunk_id == "preferred"

    def test_chunk_id_falls_back_to_chunk_id_key(self):
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict(
            {
                "content": "test",
                "score": 0.5,
                "chunk_id": "fallback",
            }
        )
        assert sd.chunk_id == "fallback"

    def test_all_score_keys_missing_defaults_to_zero(self):
        from lasagna.core.base_components import scored_document_from_dict

        sd = scored_document_from_dict({"content": "test"})
        assert sd.score == 0.0


class TestScoredDocumentRoundTrip:
    """Verify to_dict → from_dict preserves all fields."""

    def test_round_trip(self):
        from lasagna.core.base_components import (
            ScoredDocument,
            scored_document_from_dict,
            scored_document_to_dict,
        )

        original = ScoredDocument(
            content="round trip",
            score=0.42,
            document_id="doc99",
            chunk_id="ck7",
            metadata={"page": 3},
        )
        d = scored_document_to_dict(original)
        restored = scored_document_from_dict(d)

        assert restored.content == original.content
        assert restored.score == original.score
        assert restored.document_id == original.document_id
        assert restored.chunk_id == original.chunk_id
        assert restored.metadata == original.metadata

    def test_round_trip_with_zero_score(self):
        from lasagna.core.base_components import (
            ScoredDocument,
            scored_document_from_dict,
            scored_document_to_dict,
        )

        original = ScoredDocument(content="zero", score=0.0, chunk_id="z")
        d = scored_document_to_dict(original)
        restored = scored_document_from_dict(d)
        assert restored.score == 0.0
