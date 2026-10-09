#!/usr/bin/env python3
"""Test script to verify RAG backend initialization.

Run with: pytest tests/test_rag_backend_init.py -v
"""

import pytest


class TestRAGBackendInit:
    """Test RAG Backend initialization."""

    def test_backend_import(self):
        """Test that RAGBackend can be imported from the new location."""
        from lasagna.api.rag_backend import RAGBackend

        assert RAGBackend is not None

    @pytest.mark.needs_embedding_weights
    def test_backend_has_required_attributes(self):
        """Test backend has all required attributes after initialization."""
        from lasagna.api.rag_backend import RAGBackend

        backend = RAGBackend()

        required_attrs = [
            "_vector_backend",
            "embedding_manager",
            "document_processor",
            "chunker",
            "_initialized",
            "_collection_manager",
            "_document_pipeline",
            "_search_orchestrator",
            "_auto_merger",
        ]

        for attr in required_attrs:
            assert hasattr(backend, attr), f"Missing attribute: {attr}"

    @pytest.mark.needs_embedding_weights
    def test_document_processor_methods(self):
        """Test document_processor has required methods."""
        from lasagna.api.rag_backend import RAGBackend

        backend = RAGBackend()

        assert hasattr(backend.document_processor, "detect_file_type"), (
            "document_processor.detect_file_type missing"
        )
        assert hasattr(backend.document_processor, "extract_text"), (
            "document_processor.extract_text missing"
        )

    @pytest.mark.needs_embedding_weights
    def test_singleton_behavior(self):
        """Test that RAGBackend follows singleton pattern."""
        from lasagna.api.rag_backend import RAGBackend

        backend1 = RAGBackend()
        backend2 = RAGBackend()

        assert backend1 is backend2, "Singleton pattern broken (different instances)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
