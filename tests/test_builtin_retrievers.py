"""The built-in retrievers do what their papers say, and the pipeline lets them.

`retriever` had no registered component: dense, BM25, SPLADE and hybrid search
existed only inside the orchestrator, so a retriever sweep could compare a
user's retriever with nothing. Each is now a component, and each test below
checks it against the method it is named for on a corpus small enough to score
by hand: the BM25 formula, Reciprocal Rank Fusion's 1/(k + rank), a convex
combination of theoretically normalised scores, a sparse dot product.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.api._search_orchestrator import RetrievalContext
from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.knowledge_evaluation_sdk.stages import stage_of
from lasagna.retrieval.search import hybrid, retrievers
from lasagna.retrieval.search.hybrid import HybridSearcher, SPLADESparseEncoder, _SparseCorpus
from lasagna.vector_backends.base import VectorSearchResult

pytestmark = pytest.mark.skipif(
    not hybrid.BM25_FALLBACK_AVAILABLE, reason="rank_bm25 not installed"
)

BUILT_INS = ["dense", "bm25", "splade", "hybrid_rrf", "hybrid_convex"]

FILLER = "alpha beta gamma delta epsilon zeta theta iota kappa lambd".split()

# Query "fox". Dense order (cosine to q = (1, 0)): c2, c0, c1, c3, c4.
# BM25 order: c0 (fox twice, short), then c3 (fox once). "fox" is in 2 of 5
# passages, so its IDF is positive and no floor applies.
ROWS = [
    {"chunk_id": "c0", "document_id": "d0", "content": "fox fox runs", "vec": [0.8, 0.6]},
    {"chunk_id": "c1", "document_id": "d1", "content": "red barn door", "vec": [0.6, 0.8]},
    {"chunk_id": "c2", "document_id": "d2", "content": "quiet green hills", "vec": [1.0, 0.0]},
    {
        "chunk_id": "c3",
        "document_id": "d3",
        "content": "fox " + " ".join(FILLER),
        "vec": [0.0, 1.0],
    },
    {"chunk_id": "c4", "document_id": "d4", "content": "cold wet stone", "vec": [-1.0, 0.0]},
]


class _Store:
    """A vector store holding ROWS: brute-force cosine search, full scans, lookups."""

    backend_name = "fake"

    def __init__(self, rows=ROWS):
        self.rows = rows
        self.searches, self.lookups = [], []

    def iter_all(self, collection_name, filter_expr=None, output_fields=None, **kw):
        return iter(
            [{k: v for k, v in r.items() if k != "vec"} | {"metadata": {}} for r in self.rows]
        )

    def search(self, collection_name, query_embedding, limit=10, **kw):
        self.searches.append(dict(kw, limit=limit))
        q = np.asarray(query_embedding)

        def cos(r):
            return float(q @ np.asarray(r["vec"]) / np.linalg.norm(q) / np.linalg.norm(r["vec"]))

        ranked = sorted(self.rows, key=cos, reverse=True)[:limit]
        return [
            VectorSearchResult(
                id=r["chunk_id"],
                score=cos(r),
                entity={
                    "document_id": r["document_id"],
                    "content": r["content"],
                    "metadata": {"page": 1},
                    "embedding": r["vec"],
                },
            )
            for r in ranked
        ]

    def query(self, collection_name, filter_expr=None, output_fields=None, limit=100, **kw):
        self.lookups.append(filter_expr)
        return [{"chunk_id": r["chunk_id"], "embedding": r["vec"]} for r in self.rows]


def _searcher():
    s = HybridSearcher(enable_periodic_cleanup=False)
    s.is_native_mode_available = lambda: False
    s._backend_supports_native_hybrid = lambda sid=None: False
    return s


def _context(store=None, **changes):
    fields = dict(
        backend=store or _Store(),
        collection="chunks",
        session_id="sess",
        filter_expr="session-and-documents",
        session_filter="session",
        partition_key=None,
        query_embedding=np.array([1.0, 0.0]),
        hybrid_searcher=_searcher(),
        search_params="params",
    )
    fields.update(changes)
    return RetrievalContext(**fields)


def _config(**values):
    base = dict(
        bm25_k1=1.2,
        bm25_b=0.75,
        rrf_k=60,
        hybrid_fusion_depth=1000,
        hybrid_alpha=0.5,
        hybrid_convex_normalization="tmm",
        metric_type="COSINE",
        normalize_embeddings=True,
        enable_bm25_stemming=True,
    )
    return SimpleNamespace(**{**base, **values})


def _retrieve(name, query="fox", top_k=5, context=None, **config):
    component = registry.get("retriever", name)
    return component.retrieve(
        query, "chunks", top_k=top_k, config=_config(**config), context=context or _context()
    )


def _bm25(tf, length, n, total, avgdl, k1=1.2, b=0.75):
    """Okapi BM25 for one term (Robertson & Zaragoza 2009, eq. 3.15 with RSJ IDF)."""
    idf = math.log((total - n + 0.5) / (n + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))


# Passage lengths after tokenization: c0 3, c1 3, c2 3, c3 11, c4 3.
AVGDL = (3 + 3 + 3 + 11 + 3) / 5


class TestEveryBuiltInIsAConformingRetriever:
    def test_the_stage_lists_all_five_with_their_papers(self):
        listed = {e["name"]: e for e in registry.list_retrievers()}
        assert set(BUILT_INS) <= set(listed)
        for name in BUILT_INS:
            assert listed[name]["implements_contract"] and listed[name]["paper"], name

    @pytest.mark.parametrize("name", BUILT_INS)
    def test_each_passes_the_retriever_contract(self, name):
        report = ComponentTestHarness().validate(registry.get("retriever", name))
        assert report.passed, report.failures

    @pytest.mark.parametrize("name", BUILT_INS)
    def test_each_asks_the_pipeline_for_its_settings_and_store(self, name):
        assert registry.get_class("retriever", name)._forwards_session_config is True

    @pytest.mark.parametrize("name", BUILT_INS)
    def test_called_outside_a_pipeline_search_each_says_what_it_needs(self, name):
        with pytest.raises(RuntimeError, match="context="):
            registry.get("retriever", name).retrieve("fox", "chunks", top_k=3)

    def test_their_settings_belong_to_the_retriever_stage(self):
        for key in (
            "bm25_k1",
            "bm25_b",
            "splade_model",
            "hybrid_fusion_depth",
            "hybrid_convex_normalization",
        ):
            assert stage_of(key) == "retriever", key
            assert hasattr(type(rag_config), key), f"{key} has no config property"


class TestDense:
    def test_it_runs_the_default_vector_search_with_the_sessions_filter(self):
        store = _Store()
        out = _retrieve("dense", top_k=3, context=_context(store))
        assert [d.chunk_id for d in out] == ["c2", "c0", "c1"]
        assert [d.document_id for d in out] == ["d2", "d0", "d1"]
        assert out[0].score == pytest.approx(1.0) and out[0].metadata == {"page": 1}
        call = store.searches[0]
        assert call["filter_expr"] == "session-and-documents"
        assert call["search_params"] == "params" and call["limit"] == 3


class TestBM25:
    def test_scores_are_the_okapi_formula(self):
        out = _retrieve("bm25")
        assert [d.chunk_id for d in out] == ["c0", "c3"], "only passages containing the term"
        assert out[0].score == pytest.approx(_bm25(2, 3, 2, 5, AVGDL))
        assert out[1].score == pytest.approx(_bm25(1, 11, 2, 5, AVGDL))

    def test_b_normalises_length_and_k1_saturates_term_frequency(self):
        # Without length normalisation the long passage's repeated term wins.
        long_doc = {"chunk_id": "long", "document_id": "dl", "content": "fox fox fox "}
        long_doc["content"] += " ".join(FILLER * 3)
        rows = [
            long_doc,
            {"chunk_id": "short", "document_id": "ds", "content": "fox den"},
            *[
                {"chunk_id": f"x{i}", "document_id": f"x{i}", "content": "cold wet"}
                for i in range(3)
            ],
        ]
        context = _context(_Store([dict(r, vec=[1.0, 0.0]) for r in rows]))
        flat = _retrieve("bm25", context=context, bm25_b=0.0)
        normalised = _retrieve("bm25", context=context, bm25_b=1.0)
        assert [d.chunk_id for d in flat] == ["long", "short"]
        assert [d.chunk_id for d in normalised] == ["short", "long"]
        # k1 -> 0 makes BM25 binary in tf: both score the term's IDF alone.
        binary = _retrieve("bm25", context=context, bm25_k1=1e-9, bm25_b=0.0)
        assert binary[0].score == pytest.approx(binary[1].score)

    def test_a_document_filter_is_respected(self):
        out = _retrieve("bm25", context=_context(document_ids=["d3"]))
        assert [d.document_id for d in out] == ["d3"]

    def test_it_reads_the_store_when_the_index_was_never_filled(self):
        # Speed profiles turn hybrid search off, so nothing filled the index at
        # upload; the first query fills it from the store.
        context = _context()
        assert not context.hybrid_searcher.get_session_index("sess").documents
        assert _retrieve("bm25", context=context)
        assert len(context.hybrid_searcher.get_session_index("sess").documents) == 5

    def test_it_returns_new_documents_and_leaves_the_index_alone(self):
        context = _context()
        out = _retrieve("bm25", context=context)
        out[0].metadata["seen_by"] = "a later stage"
        index = context.hybrid_searcher.get_session_index("sess")
        assert all("seen_by" not in d["metadata"] for d in index.documents)


class TestHybridRRF:
    def test_scores_are_the_sum_of_reciprocal_ranks(self):
        out = _retrieve("hybrid_rrf")
        # c0: dense 2, BM25 1. c3: dense 4, BM25 2. Others: dense only.
        expected = {
            "c0": 1 / 62 + 1 / 61,
            "c3": 1 / 64 + 1 / 62,
            "c2": 1 / 61,
            "c1": 1 / 63,
            "c4": 1 / 65,
        }
        assert [d.chunk_id for d in out] == ["c0", "c3", "c2", "c1", "c4"]
        assert {d.chunk_id: d.score for d in out} == pytest.approx(expected)

    def test_rrf_k_is_the_setting(self):
        out = _retrieve("hybrid_rrf", rrf_k=1)
        assert out[0].score == pytest.approx(1 / 3 + 1 / 2)

    def test_fusion_depth_cuts_each_list_before_fusing(self):
        out = _retrieve("hybrid_rrf", hybrid_fusion_depth=1, top_k=2)
        # Depth 2 (never below top_k): dense brings c2, c0; BM25 c0, c3.
        expected = {"c0": 1 / 62 + 1 / 61, "c2": 1 / 61}
        assert {d.chunk_id: d.score for d in out} == pytest.approx(expected)


class TestHybridConvex:
    def test_scores_combine_theoretically_normalised_dense_and_bm25(self):
        out = _retrieve("hybrid_convex")
        lex0, lex3 = _bm25(2, 3, 2, 5, AVGDL), _bm25(1, 11, 2, 5, AVGDL)
        # tmm: cosine from its minimum -1, BM25 from 0, each up to the max seen.
        sem = {"c0": 0.8, "c1": 0.6, "c2": 1.0, "c3": 0.0, "c4": -1.0}
        lex = {"c0": lex0, "c3": lex3}
        expected = {c: 0.5 * (sem[c] + 1) / 2 + 0.5 * lex.get(c, 0.0) / lex0 for c in sem}
        assert {d.chunk_id: d.score for d in out} == pytest.approx(expected)
        assert [d.chunk_id for d in out] == sorted(expected, key=expected.get, reverse=True)

    def test_a_score_missing_from_one_list_is_computed_not_zeroed(self):
        # Depth 3: dense brings c2, c0, c1; BM25 brings c0, c3. c3's similarity
        # is read from its stored vector (cosine 0, so 0.5 under tmm), as the
        # paper computes a missing score; c1's BM25 is scored, and is 0.
        store = _Store()
        out = _retrieve("hybrid_convex", top_k=3, hybrid_fusion_depth=3, context=_context(store))
        lex0, lex3 = _bm25(2, 3, 2, 5, AVGDL), _bm25(1, 11, 2, 5, AVGDL)
        assert store.lookups, "the vector of a chunk only BM25 found was looked up"
        assert {d.chunk_id: d.score for d in out} == pytest.approx(
            {"c0": 0.5 * 0.9 + 0.5, "c2": 0.5, "c3": 0.5 * 0.5 + 0.5 * lex3 / lex0}
        )

    def test_alpha_weighs_the_dense_side(self):
        dense_only = _retrieve("hybrid_convex", hybrid_alpha=1.0)
        lexical_only = _retrieve("hybrid_convex", hybrid_alpha=0.0)
        assert dense_only[0].chunk_id == "c2"
        assert lexical_only[0].chunk_id == "c0"

    def test_min_max_and_z_score_normalise_over_the_fused_set(self):
        lex0, lex3 = _bm25(2, 3, 2, 5, AVGDL), _bm25(1, 11, 2, 5, AVGDL)
        ids = ["c2", "c0", "c1", "c3"]  # the union at depth 3
        sem = np.array([1.0, 0.8, 0.6, 0.0])
        lex = np.array([0.0, lex0, 0.0, lex3])
        for method, phi in (
            ("minmax", lambda v: (v - v.min()) / (v.max() - v.min())),
            ("zscore", lambda v: (v - v.mean()) / v.std()),
        ):
            out = _retrieve(
                "hybrid_convex", top_k=4, hybrid_fusion_depth=3, hybrid_convex_normalization=method
            )
            expected = dict(zip(ids, 0.5 * phi(sem) + 0.5 * phi(lex)))
            assert {d.chunk_id: d.score for d in out} == pytest.approx(expected), method

    def test_an_unknown_normalization_names_the_choices(self):
        with pytest.raises(ValueError, match="tmm"):
            _retrieve("hybrid_convex", hybrid_convex_normalization="softmax")

    def test_tmm_refuses_an_unbounded_similarity(self):
        with pytest.raises(ValueError, match="minmax"):
            _retrieve("hybrid_convex", metric_type="IP", normalize_embeddings=False)


class _FakeSplade:
    """Hand-made SPLADE weights: term 1 is "fox", term 7 an expansion of it."""

    model_name = "fake-splade"
    weights = {
        "fox": {1: 1.0, 7: 0.5},
        "fox fox runs": {1: 0.2},
        "red barn door": {7: 2.0},
        "quiet green hills": {1: 0.5, 7: 0.4},
    }

    def encode(self, texts, batch_size=8):
        out = []
        for t in texts:
            w = self.weights.get(t, {3: 9.0})
            out.append((np.array(list(w), dtype=np.int64), np.array(list(w.values()))))
        return out

    def corpus(self, key, texts):
        return _SparseCorpus(self.encode(texts))


class TestSPLADE:
    def test_scores_are_the_dot_product_of_sparse_weights(self, monkeypatch):
        monkeypatch.setattr(retrievers, "_get_splade_encoder", lambda name: _FakeSplade())
        out = _retrieve("splade")
        # c1 shares no word with "fox" and wins on the expansion term.
        assert [(d.chunk_id, round(d.score, 6)) for d in out] == [
            ("c1", 1.0),
            ("c2", 0.7),
            ("c0", 0.2),
        ]

    def test_it_encodes_with_the_arms_model(self, monkeypatch):
        asked = []
        monkeypatch.setattr(
            retrievers, "_get_splade_encoder", lambda name: asked.append(name) or _FakeSplade()
        )
        _retrieve("splade", splade_model="naver/splade_v2_distil")
        assert asked == ["naver/splade_v2_distil"]


def _tiny_masked_lm(tmp_path, head=True):
    """A two-layer BERT saved locally: a masked LM, or an encoder with no MLM head."""
    transformers = pytest.importorskip("transformers")
    import torch

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "fox", "den", "barn", "red"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = (transformers.BertForMaskedLM if head else transformers.BertModel)(config)
    path = tmp_path / ("mlm" if head else "encoder")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


class TestSPLADEEncoder:
    def test_a_masked_lm_encodes_to_non_negative_sparse_weights(self, tmp_path):
        encoder = SPLADESparseEncoder(_tiny_masked_lm(tmp_path))
        (ids, weights), _ = encoder.encode(["fox den", "red barn"])
        assert len(ids) == len(weights) > 0 and (weights > 0).all()
        hits = encoder.search("fox", ["fox den", "red barn"], key="k", top_k=2)
        assert hits and all(score > 0 for _, score in hits)

    def test_a_checkpoint_without_a_language_model_head_is_refused(self, tmp_path):
        encoder = SPLADESparseEncoder(_tiny_masked_lm(tmp_path, head=False))
        with pytest.raises(RuntimeError, match="language-model head is missing"):
            encoder.encode(["fox"])

    def test_a_model_that_cannot_load_raises_once_and_keeps_raising(self, tmp_path):
        encoder = SPLADESparseEncoder(str(tmp_path / "no-such-model"))
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Cannot load SPLADE model"):
                encoder.encode(["fox"])

    def test_the_sparse_index_scores_by_dot_product(self):
        corpus = _SparseCorpus(
            [
                (np.array([1, 4]), np.array([0.5, 2.0])),
                (np.array([4]), np.array([1.0])),
                (np.array([9]), np.array([3.0])),
            ]
        )
        scores = corpus.score(np.array([4, 1]), np.array([2.0, 1.0]))
        assert scores.tolist() == pytest.approx([4.5, 2.0, 0.0])


def _cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(name, "config.json"), str)


@pytest.mark.skipif(
    not _cached("naver/splade-cocondenser-ensembledistil"), reason="SPLADE++ not cached"
)
def test_splade_plus_plus_prefers_the_passage_that_answers():
    encoder = SPLADESparseEncoder("naver/splade-cocondenser-ensembledistil")
    passages = [
        "The capital of France is Paris.",
        "Bananas are rich in potassium.",
    ]
    hits = encoder.search("what is the capital city of france", passages, key="real", top_k=2)
    assert hits[0][0] == 0


@pytest.mark.skipif(not _cached("BAAI/bge-small-en-v1.5"), reason="bge-small not cached")
def test_dense_and_bm25_disagree_on_a_real_embedding(tmp_path):
    """With a real embedder, the two first stages order a small corpus differently."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    texts = [
        "A feline rested on the rug.",
        "The cat cat cat sat. Cat facts: cats purr.",
        "Stock markets fell sharply last Tuesday.",
    ]
    vectors = model.encode(texts, normalize_embeddings=True)
    rows = [
        {"chunk_id": f"c{i}", "document_id": f"d{i}", "content": t, "vec": list(v)}
        for i, (t, v) in enumerate(zip(texts, vectors))
    ]
    query = "a cat lying on a carpet"
    context = _context(_Store(rows), query_embedding=model.encode(query, normalize_embeddings=True))
    dense = _retrieve("dense", query=query, top_k=3, context=context)
    lexical = _retrieve("bm25", query=query, top_k=3, context=context)
    assert [d.chunk_id for d in dense][-1] == "c2", "the unrelated passage is farthest"
    assert dense[0].chunk_id == "c0", "the paraphrase is nearest in embedding space"
    assert [d.chunk_id for d in lexical] == ["c1", "c0"], "BM25 needs a shared word"


# ---------------------------------------------------------------------------
# The orchestrator's call site
# ---------------------------------------------------------------------------


class _Backend:
    backend_name = "fake"

    def get_collection_stats(self, name):
        return {}

    def search(self, **kw):
        raise AssertionError("the store's own search ran instead of the retriever")


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    backend = _Backend()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )

    def fused(**kw):
        raise AssertionError("the retriever's list was fused with BM25")

    o.hybrid_searcher = SimpleNamespace(search_hybrid=fused, hydrate_session_index=fused)
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._maybe_reload_embedding_manager = lambda config: None
    o.backend = backend
    return o


#: Hybrid search is on, so the default path would fuse with BM25.
ARM = dict(
    enable_hybrid_search=True,
    enable_reranking=False,
    enable_query_routing=False,
    enable_query_classification=False,
    enable_search_cache=False,
    enable_hyde=False,
    enable_mmr=False,
    enable_context_reordering=False,
    enable_hierarchical=False,
    enable_rag_fusion=False,
    enable_query_expansion=False,
    similarity_threshold=0.0,
    hybrid_alpha=0.3,
)


class _Users(BaseRetriever):
    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, collection, top_k=5, **kwargs):
        self.kwargs = kwargs
        return [
            ScoredDocument(content="b", score=0.9, document_id="d2", chunk_id="c2"),
            ScoredDocument(content="a", score=0.5, document_id="d1", chunk_id="c1"),
        ]


class _BuiltIn(_Users):
    _forwards_session_config = True


def test_a_built_in_retriever_gets_the_arms_config_and_the_sessions_store(orchestrator):
    retriever = _BuiltIn()
    out = orchestrator.search(
        "q", "sess1", top_k=5, _retriever_override=retriever, query_config_overrides=ARM
    )
    assert [r.chunk.chunk_id for r in out] == ["c2", "c1"]
    assert retriever.kwargs["config"].hybrid_alpha == 0.3
    context = retriever.kwargs["context"]
    assert isinstance(context, RetrievalContext)
    assert context.backend is orchestrator.backend and context.session_id == "sess1"
    assert context.query_embedding.tolist() == [1.0, 1.0, 1.0]
    assert context.session_filter is not None and context.filter_expr is not None


def test_a_users_retriever_sees_only_the_public_contract(orchestrator):
    retriever = _Users()
    orchestrator.search(
        "q", "sess1", top_k=5, _retriever_override=retriever, query_config_overrides=ARM
    )
    assert retriever.kwargs == {}
