"""Late chunking: whole-document token embeddings, mean-pooled per chunk.

Late chunking changes the vectors, not the chunk boundaries, so it is an
embedding-time option (``enable_late_chunking``) that composes with any
chunker, not a chunker of its own. These tests pin down what it computes, that
documents longer than the model's window are processed in overlapping windows
instead of truncated, and that an evaluation never gets standard embeddings
under its name.
"""

from __future__ import annotations

import numpy as np
import pytest

from lasagna.core.config import rag_config
from lasagna.core.strict import strict
from lasagna.infrastructure.embedding.manager import EmbeddingManager

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _cached(repo):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


def _manager(model=MINILM, **overrides):
    return EmbeddingManager(
        rag_config.with_query_overrides(
            {
                "embedding_model": model,
                "use_gpu": False,
                "enable_embedding_cache": False,
                **overrides,
            }
        )
    )


class TestItNeverRunsUnderAnotherName:
    def test_an_evaluation_refuses_a_backend_without_token_output(self):
        mgr = _manager()
        mgr._backend = "flag-embedding"
        with strict(), pytest.raises(RuntimeError, match="Late chunking cannot run"):
            mgr.encode_documents_late_chunking("a b", ["a b"])

    def test_serving_falls_back_to_standard_encoding(self, monkeypatch):
        mgr = _manager()
        mgr._backend = "flag-embedding"
        monkeypatch.setattr(
            mgr, "encode_documents_batch", lambda texts, show_progress=True: ["standard"]
        )
        assert mgr.encode_documents_late_chunking("a b", ["a b"]) == ["standard"]


def _pooled(mgr, ids, lo, hi, offsets):
    """Mean of the chunk's tokens, computed from one direct model call on ``ids``."""
    import torch

    tok = mgr.model.tokenizer
    framed = [tok.cls_token_id, *ids, tok.sep_token_id]
    with torch.no_grad():
        hidden = mgr.model[0].auto_model(input_ids=torch.tensor([framed])).last_hidden_state[0]
    hidden = hidden.numpy()[1:-1]
    mask = (offsets[:, 1] > lo) & (offsets[:, 0] < hi)
    v = hidden[mask].mean(axis=0)
    return v / np.linalg.norm(v)


@pytest.mark.skipif(not _cached(MINILM), reason=f"{MINILM} is not cached")
class TestWithARealModel:
    def test_a_chunk_is_the_mean_of_its_tokens_in_the_whole_document(self):
        mgr = _manager()
        text = "The cat sat on the mat. It purred loudly. Then it slept."
        chunks = ["The cat sat on the mat.", "It purred loudly."]
        late = mgr.encode_documents_late_chunking(text, chunks)

        enc = mgr.model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = np.asarray(enc["offset_mapping"])
        assert np.allclose(late[1], _pooled(mgr, enc["input_ids"], 24, 41, offsets), atol=1e-5)
        # The context is the point: standard encoding of the chunk alone differs.
        alone = mgr.encode_documents_batch(chunks[1:], show_progress=False)[0]
        assert not np.allclose(late[1], alone, atol=1e-3)

    def test_a_long_document_is_embedded_in_overlapping_windows(self, monkeypatch):
        mgr = _manager(late_chunking_overlap=32)
        sentence = "The cat sat on the mat. "
        text = sentence * 60  # 7 tokens each: 420 tokens, past MiniLM's 256
        chunks = [sentence.strip()] * 60
        late = mgr.encode_documents_late_chunking(text, chunks)

        enc = mgr.model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], np.asarray(enc["offset_mapping"])
        assert len(ids) == 420
        # Windows of 254 tokens (256 less [CLS] and [SEP]) overlapping by 32:
        # tokens 0..253, then 222..419, of which 254.. are kept.
        first = _pooled(mgr, ids[:254], 0, 23, offsets[:254])
        last_lo = len(text) - len(sentence)
        second = _pooled(mgr, ids[222:], last_lo, last_lo + 23, offsets[222:])
        assert np.allclose(late[0], first, atol=1e-5)
        assert np.allclose(late[-1], second, atol=1e-5)

        # No chunk fell back to standard encoding for being past the window.
        def no_fallback(*a, **k):
            raise AssertionError("a chunk was encoded the standard way")

        monkeypatch.setattr(mgr, "encode_documents_batch", no_fallback)
        mgr.encode_documents_late_chunking(text, chunks)

    def test_a_chunk_that_is_not_a_slice_fails_an_evaluation(self):
        mgr = _manager()
        with strict(), pytest.raises(RuntimeError, match="not slices of the document"):
            mgr.encode_documents_late_chunking("The cat sat.", ["A rewritten proposition."])
