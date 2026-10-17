"""Unified component registry for swappable RAG pipeline components.

The registry enables a swap-and-measure workflow for researchers:

    from lasagna.core.registry import registry
    from lasagna.core.base_components import BaseReranker

    @registry.register("reranker", "my_novel_reranker")
    class MyNovelReranker(BaseReranker):
        def rerank(self, query, documents, top_k=5):
            ...

    # Later, build a pipeline using the custom component
    pipeline = KnowledgePipeline(reranker="my_novel_reranker")

Type-specific convenience methods are also available for backward
compatibility with the retrieval registry API:

    from lasagna.core.registry import registry

    registry.register_reranker("my_colbert", ColBERTReranker)
    reranker_cls = registry.get_reranker("my_colbert")
"""

import inspect
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Type

import os

from loguru import logger


# Valid component types that can be registered.
# NOTE: "graph_node" is intentionally NOT here — nodes are callables stored in
# a separate ``_callables`` dict with dedicated get/register methods, because
# ``registry.get()`` instantiates via ``cls(**kwargs)`` which would *execute*
# a callable node function instead of returning it.
COMPONENT_TYPES = frozenset(
    {
        "embedder",
        "chunker",
        "reranker",
        "retriever",
        "document_processor",
        "query_enhancer",
        "agentic_pipeline",
        "generator",
    }
)

# Aliases allow both names to refer to the same category.
# The retrieval registry used "enhancer" while core used "query_enhancer".
_CATEGORY_ALIASES: Dict[str, str] = {
    "enhancer": "query_enhancer",
}


def _resolve_category(name: str) -> str:
    """Resolve a category name to its canonical form."""
    return _CATEGORY_ALIASES.get(name, name)


def _get_expected_abc(category: str) -> Optional[Type]:
    """Return the expected ABC for a component category, or None."""
    try:
        from .base_components import (
            BaseReranker,
            BaseChunker,
            BaseEmbedder,
            BaseRetriever,
            BaseDocumentProcessor,
            BaseQueryEnhancer,
            BaseAgenticPipeline,
            BaseGenerator,
        )

        return {
            "reranker": BaseReranker,
            "chunker": BaseChunker,
            "embedder": BaseEmbedder,
            "retriever": BaseRetriever,
            "document_processor": BaseDocumentProcessor,
            "query_enhancer": BaseQueryEnhancer,
            "agentic_pipeline": BaseAgenticPipeline,
            "generator": BaseGenerator,
        }.get(category)
    except ImportError:
        return None


def _warn_abc_conformance(category: str, name: str, cls: type) -> None:
    """Warn if *cls* doesn't inherit from the expected ABC for *category*.

    Warns instead of raising so that duck-typed components still work —
    the ABC contract is recommended, not enforced.
    """
    abc = _get_expected_abc(category)
    if not abc:
        return
    try:
        if not issubclass(cls, abc):
            logger.warning(
                f"Registered {category} '{name}' ({cls.__name__}) does not "
                f"inherit from {abc.__name__}. It may not be compatible with "
                f"pipeline tooling."
            )
    except TypeError:
        # cls is not a class (e.g. a function) — warn clearly
        logger.warning(
            f"Registered {category} '{name}': expected a class, got {type(cls).__name__}"
        )


def _resolve_factory(factory: str) -> Type:
    """Import a class from a ``'module.path:ClassName'`` factory string."""
    module_path, _, attr_name = factory.rpartition(":")
    if not module_path or not attr_name:
        raise ImportError(f"Invalid factory string: '{factory}' (expected 'module:Class')")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


#: What entry-point discovery found, and what it could not load.
#:
#: Discovery is best-effort by design -- one broken third-party plugin must not
#: stop the registry from loading -- but "best effort" without a record means a
#: plugin author's component silently does not exist. This is what
#: ``lasagna plugins list`` reads.
_DISCOVERY_LOG: List[Dict[str, Any]] = []


def _record_discovery(
    group: str, name: str, value: str, *, ok: bool, error: Optional[BaseException] = None
) -> None:
    _DISCOVERY_LOG.append(
        {
            "group": group,
            "name": name,
            "value": value,
            "ok": ok,
            "error": f"{type(error).__name__}: {error}" if error else None,
        }
    )


def discovery_log() -> List[Dict[str, Any]]:
    """Entry points seen during discovery, loaded or failed.

    Empty until the registry has been touched, since discovery is lazy.
    """
    return list(_DISCOVERY_LOG)


class _Registration:
    """Internal registration record with optional metadata.

    Supports deferred resolution via ``_factory``: when ``cls`` is None
    and ``_factory`` is a ``"module:ClassName"`` string, the class is
    resolved on first access to avoid heavy imports at registry init.

    ``implements_contract`` is declared by built-ins, which are known to
    implement their stage's interface, so listing one never imports it. For
    everything else it is ``None`` and derived from the class.
    """

    __slots__ = (
        "cls",
        "description",
        "paper",
        "claims",
        "version",
        "_factory",
        "hidden",
        "implements_contract",
    )

    def __init__(
        self,
        cls: Optional[Type] = None,
        description: str = "",
        paper: Optional[str] = None,
        claims: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        _factory: Optional[str] = None,
        hidden: bool = False,
        implements_contract: Optional[bool] = None,
    ):
        self.hidden = hidden
        self.cls = cls
        self.description = description
        from .claims import normalize_claims, unregistered_benchmarks

        self.paper = paper
        # Keyed by benchmark; a claim that names none, or a metric no run
        # records, is refused here, at registration.
        self.claims = normalize_claims(claims)
        # A Python warning, not a log line: the library's log is off unless
        # enabled, and this is for whoever wrote the decorator.
        for warning in unregistered_benchmarks(self.claims):
            warnings.warn(warning, UserWarning, stacklevel=2)
        self.version = version
        self._factory = _factory
        self.implements_contract = implements_contract

    def resolve(self) -> Type:
        """Return the class, lazily resolving from factory if needed."""
        if self.cls is None and self._factory:
            self.cls = _resolve_factory(self._factory)
        if self.cls is None:
            raise ImportError(
                f"Cannot resolve registration (no cls and no factory): {self.description}"
            )
        return self.cls


class ComponentRegistry:
    """Unified registry for RAG pipeline components.

    Thread-safe singleton that stores component implementations
    by type and name. Supports both decorator-based and imperative
    registration for a clean researcher workflow.

    Category aliasing: ``"enhancer"`` and ``"query_enhancer"`` refer
    to the same slot so that both the core and retrieval APIs work
    transparently.
    """

    _instance: Optional["ComponentRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ComponentRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._components: Dict[str, Dict[str, _Registration]] = {
            comp_type: {} for comp_type in COMPONENT_TYPES
        }
        # Separate store for callables (graph nodes).  These are NOT in
        # COMPONENT_TYPES because ``get()`` would instantiate them via
        # ``cls(**kwargs)`` which would *execute* the node function.
        self._callables: Dict[str, _Registration] = {}
        self._data_lock = threading.Lock()

        self._builtins_registered = False
        # (stage, name) of everything the package itself registers.
        self._builtin_names: set = set()

    # =========================================================================
    # Built-in Registration (deferred until first access)
    # =========================================================================

    def _ensure_builtins(self) -> None:
        """Register built-in components on first use (lazy, thread-safe)."""
        if self._builtins_registered:
            return
        # Registration imports entry-point plugins, and a plugin that uses the
        # documented @registry.register decorator calls back in here on the
        # same thread. Waiting for the lock then deadlocked every process that
        # touched the registry; the registering thread proceeds instead.
        if getattr(self, "_registering_thread", None) == threading.get_ident():
            return
        with self._lock:
            if self._builtins_registered:
                return
            self._registering_thread = threading.get_ident()
            try:
                self._register_builtins()
                self._builtins_registered = True
            finally:
                self._registering_thread = None

    def _register_builtins(self) -> None:
        """Register built-in component implementations from the retrieval layer."""
        # Every listed built-in implements its stage's interface, so any two
        # components of a stage are interchangeable: comparing them measures
        # the algorithms, never an interface difference. Names ending in
        # "_adapted" predate that and stay resolvable as unlisted aliases.
        _r = "lasagna.retrieval.search.ranker:"
        _e = "lasagna.retrieval.enhancement.enhancer:"
        _c = "lasagna.processing.chunker.semantic:"
        _a = "lasagna.retrieval.enhancement.augmentation:"
        builtins = [
            # (stage, name, "module:Class", description, paper, aliases)
            (
                "reranker",
                "cross_encoder",
                _r + "CrossEncoderRerankerAdapter",
                "Cross-encoder reranker (the configured reranker_model)",
                # Nogueira & Cho 2019, Passage Re-ranking with BERT (monoBERT).
                "arxiv:1901.04085",
                ["cross_encoder_adapted"],
            ),
            (
                "reranker",
                "combined",
                _r + "CombinedRerankerAdapter",
                "Cross-encoder followed by MMR diversity",
                None,
                ["combined_adapted"],
            ),
            (
                "reranker",
                "ensemble",
                _r + "EnsembleRerankerAdapter",
                "Weighted ensemble of cross-encoders",
                None,
                ["ensemble_adapted"],
            ),
            (
                "reranker",
                "mmr",
                _r + "MMRRerankerAdapter",
                "MMR diversity reranking",
                # Carbonell & Goldstein 1998, The Use of MMR, Diversity-Based
                # Reranking for Reordering Documents and Producing Summaries.
                "doi:10.1145/290941.291025",
                ["mmr_adapted"],
            ),
            (
                "reranker",
                "colbert",
                _r + "ColBERTRerankerAdapter",
                "ColBERT late-interaction reranker with token-level MaxSim",
                "arxiv:2004.12832",
                [],
            ),
            (
                "reranker",
                "llm_listwise",
                _r + "LLMListwiseRerankerAdapter",
                "RankGPT: LLM permutations over windows sliding from the bottom up",
                "arxiv:2304.09542",
                [],
            ),
            (
                "query_enhancer",
                "hyde",
                _e + "HyDEEnhancerAdapter",
                "HyDE: the query plus a hypothetical answer passage, searched and fused",
                "arxiv:2212.10496",
                [],
            ),
            (
                "query_enhancer",
                "query_expansion",
                _e + "QueryExpansionAdapter",
                "LLM paraphrases of the query, searched and fused",
                None,
                [],
            ),
            (
                "query_enhancer",
                "query_decomposition",
                _e + "QueryDecompositionAdapter",
                "Least-to-most decomposition: the sub-questions a question needs, searched and fused",
                "arxiv:2205.10625",
                [],
            ),
            (
                "query_enhancer",
                "combined",
                _e + "QueryEnhancerAdapter",
                "Diverse LLM reformulations of the query (RAG Fusion style)",
                None,
                ["combined_adapted"],
            ),
            (
                "chunker",
                "semantic",
                _c + "SemanticChunkerAdapter",
                "Semantic chunker",
                None,
                ["semantic_adapted"],
            ),
            (
                "chunker",
                "hierarchical",
                _c + "HierarchicalChunkerAdapter",
                "Hierarchical parent-child chunker",
                None,
                ["hierarchical_adapted"],
            ),
            (
                "chunker",
                "structure_aware",
                _c + "StructureAwareChunkerAdapter",
                "Structure-aware chunker",
                None,
                ["structure_aware_adapted"],
            ),
            (
                "embedder",
                "default",
                "lasagna.infrastructure.embedding.manager:EmbeddingManagerAdapter",
                "The configured embedding model",
                None,
                ["default_adapted"],
            ),
            (
                "document_processor",
                "default",
                "lasagna.processing.document_processor:DocumentProcessorAdapter",
                "Built-in extraction (pypdf and pdfplumber, OCR when enabled)",
                None,
                ["default_adapted"],
            ),
            (
                "retriever",
                "dense",
                "lasagna.retrieval.search.retrievers:DenseRetriever",
                "Dense vector search with the configured embedder (the default first stage)",
                "arxiv:2004.04906",
                [],
            ),
            (
                "retriever",
                "bm25",
                "lasagna.retrieval.search.retrievers:BM25Retriever",
                "Okapi BM25 over the session's chunks (bm25_k1, bm25_b)",
                "doi:10.1561/1500000019",
                [],
            ),
            (
                "retriever",
                "splade",
                "lasagna.retrieval.search.retrievers:SPLADERetriever",
                "SPLADE learned sparse retrieval (splade_model)",
                "arxiv:2109.10086",
                [],
            ),
            (
                "retriever",
                "hybrid_rrf",
                "lasagna.retrieval.search.retrievers:HybridRRFRetriever",
                "Dense and BM25 fused by Reciprocal Rank Fusion (rrf_k)",
                "doi:10.1145/1571941.1572114",
                [],
            ),
            (
                "retriever",
                "hybrid_convex",
                "lasagna.retrieval.search.retrievers:HybridConvexRetriever",
                "Convex combination of normalised dense and BM25 scores (hybrid_alpha)",
                "arxiv:2210.11934",
                [],
            ),
            (
                "reranker",
                "monot5",
                "lasagna.retrieval.search.rerankers:MonoT5Reranker",
                "monoT5: P(true) from a seq2seq relevance model (monot5_model)",
                "arxiv:2003.06713",
                [],
            ),
            (
                "reranker",
                "llm_pointwise",
                "lasagna.retrieval.search.rerankers:LLMPointwiseReranker",
                "LLM relevance grade per passage on a 0-3 scale (llm_reranker_model)",
                "arxiv:2309.10621",
                [],
            ),
            (
                "reranker",
                "flashrank",
                "lasagna.retrieval.search.rerankers:FlashRankReranker",
                "FlashRank ONNX cross-encoders on CPU (pip install flashrank)",
                None,
                [],
            ),
            (
                "reranker",
                "cohere",
                "lasagna.retrieval.search.rerankers:CohereReranker",
                "Cohere Rerank API (pip install cohere, COHERE_API_KEY)",
                None,
                [],
            ),
            (
                "generator",
                "concise",
                "lasagna.generation.strategies:ConciseGenerator",
                "One concise sentence from the context (the harness's default prompt)",
                None,
                [],
            ),
            (
                "generator",
                "closed_book",
                "lasagna.generation.strategies:ClosedBookGenerator",
                "The question alone, no passages: the no-retrieval baseline",
                None,
                [],
            ),
            (
                "generator",
                "long_context",
                "lasagna.generation.strategies:LongContextGenerator",
                "The concise prompt over all ten retrieved passages, in rank order",
                None,
                [],
            ),
            (
                "generator",
                "cited",
                "lasagna.generation.strategies:CitedGenerator",
                "Answers citing their supporting passages as [n]",
                "Gao et al. 2023, Enabling Large Language Models to Generate Text with Citations (ALCE)",
                [],
            ),
            (
                "generator",
                "chain_of_thought",
                "lasagna.generation.strategies:ChainOfThoughtGenerator",
                "Step-by-step reasoning, then the final answer",
                "Wei et al. 2022, Chain-of-Thought Prompting Elicits Reasoning in LLMs",
                [],
            ),
            (
                "generator",
                "chain_of_note",
                "lasagna.generation.strategies:ChainOfNoteGenerator",
                "A reading note per passage, then the answer or 'unknown'",
                "Yu et al. 2023, Chain-of-Note: Enhancing Robustness in Retrieval-Augmented LMs",
                [],
            ),
            (
                "query_enhancer",
                "step_back",
                _a + "StepBackEnhancer",
                "Step-back prompting: the question and a more generic step-back question",
                "arxiv:2310.06117",
                [],
            ),
            (
                "query_enhancer",
                "query2doc",
                _a + "Query2DocEnhancer",
                "Query2doc: the query followed by an LLM-written pseudo-document",
                "arxiv:2303.07678",
                [],
            ),
            (
                "query_enhancer",
                "rewrite",
                _a + "RewriteEnhancer",
                "Rewrite-Retrieve-Read: search queries rewritten by an LLM",
                "arxiv:2305.14283",
                [],
            ),
            (
                "query_enhancer",
                "multi_query",
                _a + "MultiQueryEnhancer",
                "RAG-Fusion: LLM-generated search queries, fused by reciprocal rank",
                "arxiv:2402.03367",
                [],
            ),
            (
                "chunker",
                "fixed",
                "lasagna.processing.chunker.splitters:FixedSizeChunker",
                "Fixed-size windows with overlap, in characters, words or tokens",
                None,
                [],
            ),
            (
                "chunker",
                "recursive",
                "lasagna.processing.chunker.splitters:RecursiveChunker",
                "Recursive separator splitting: paragraph, line, sentence, word "
                "(LangChain RecursiveCharacterTextSplitter)",
                "https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters",
                [],
            ),
            (
                "chunker",
                "sentence_window",
                _c + "SentenceWindowChunkerAdapter",
                "Sliding windows of window_size whole sentences",
                None,
                [],
            ),
            (
                "chunker",
                "markdown_header",
                "lasagna.processing.chunker.markdown:MarkdownHeaderChunker",
                "Markdown sections with their header path in metadata "
                "(LangChain MarkdownHeaderTextSplitter)",
                "https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters",
                [],
            ),
            (
                "chunker",
                "proposition",
                "lasagna.processing.chunker.llm_chunkers:PropositionChunker",
                "Propositions written by an LLM as retrieval units (Dense X Retrieval)",
                "arxiv:2312.06648",
                [],
            ),
            (
                "chunker",
                "contextual",
                "lasagna.processing.chunker.llm_chunkers:ContextualChunker",
                "Chunks prefixed with LLM-written document context (Contextual Retrieval)",
                "https://www.anthropic.com/news/contextual-retrieval",
                [],
            ),
            (
                "embedder",
                "sentence_transformers",
                "lasagna.infrastructure.embedding.embedders:SentenceTransformersEmbedder",
                "Any sentence-transformers model (embedding_model), with its model card's "
                "query/document prefixes",
                "arxiv:1908.10084",
                [],
            ),
            (
                "embedder",
                "bge_m3",
                "lasagna.infrastructure.embedding.embedders:BGEM3Embedder",
                "BGE-M3 dense vectors; sparse and multi-vector outputs via FlagEmbedding",
                "arxiv:2402.03216",
                [],
            ),
            (
                "embedder",
                "openai",
                "lasagna.infrastructure.embedding.embedders:OpenAIEmbedder",
                "OpenAI text-embedding-3 small/large, with Matryoshka dimensions",
                "https://platform.openai.com/docs/guides/embeddings",
                [],
            ),
            (
                "embedder",
                "cohere",
                "lasagna.infrastructure.embedding.embedders:CohereEmbedder",
                "Cohere Embed v3/v4 with search_query/search_document input types",
                "https://docs.cohere.com/reference/embed",
                [],
            ),
            (
                "embedder",
                "voyage",
                "lasagna.infrastructure.embedding.embedders:VoyageEmbedder",
                "Voyage AI voyage-3 family with query/document input types",
                "https://docs.voyageai.com/docs/embeddings",
                [],
            ),
            (
                "document_processor",
                "docling",
                "lasagna.processing.docling_processor:DoclingDocumentProcessor",
                "Docling layout analysis, TableFormer tables and OCR, exported as Markdown",
                "arxiv:2408.09869",
                [],
            ),
            (
                "document_processor",
                "markitdown",
                "lasagna.processing.markdown_converters:MarkItDownProcessor",
                "Microsoft MarkItDown: PDF, Office, HTML and more to Markdown",
                "https://github.com/microsoft/markitdown",
                [],
            ),
            (
                "document_processor",
                "pymupdf4llm",
                "lasagna.processing.markdown_converters:PyMuPDF4LLMProcessor",
                "PyMuPDF4LLM: PDF to Markdown with headers and tables (AGPL-3.0)",
                "https://github.com/pymupdf/pymupdf4llm",
                [],
            ),
        ]
        # Deferred, like the agentic pipelines below. Resolving each factory
        # here imported torch, transformers and pymilvus (6.6 s) for anything
        # that touched the registry, `lasagna doctor` and listing included.
        # The modules import nothing optional at module scope, so a deferred
        # built-in is as available as it was when resolved eagerly;
        # tests/test_registry_listing_is_lazy.py resolves each one to check
        # its declared contract.
        for stage, name, factory, description, paper, aliases in builtins:
            self._components[stage][name] = _Registration(
                description=description,
                paper=paper,
                _factory=factory,
                implements_contract=True,
            )
            self._builtin_names.add((stage, name))
            for alias in aliases:
                self._components[stage][alias] = _Registration(
                    description=f"alias of {name}",
                    paper=paper,
                    _factory=factory,
                    hidden=True,
                    implements_contract=True,
                )
                self._builtin_names.add((stage, alias))

        # --- Agentic Pipelines (deferred to avoid heavy agentic imports) ---
        self._components["agentic_pipeline"]["default"] = _Registration(
            description="Built-in structured RAG graph (13 nodes, CRAG, quality gates)",
            _factory="lasagna.agentic.graph_builder:BuiltinStructuredPipeline",
            implements_contract=True,
        )
        self._components["agentic_pipeline"]["legacy_react"] = _Registration(
            description="Legacy ReAct agent with monolithic search tool",
            paper="arxiv:2210.03629",  # Yao et al. 2022, ReAct
            _factory="lasagna.agentic.agentic_rag:LegacyReActPipeline",
            implements_contract=True,
        )
        self._builtin_names.update(
            {("agentic_pipeline", "default"), ("agentic_pipeline", "legacy_react")}
        )
        _l = "lasagna.agentic.loops."
        for name, factory, description, paper in [
            (
                "self_ask",
                _l + "self_ask:SelfAskPipeline",
                "Self-Ask: follow-up questions answered by retrieval until the final answer",
                "arxiv:2210.03350",
            ),
            (
                "ircot",
                _l + "ircot:IRCoTPipeline",
                "IRCoT: chain-of-thought sentences interleaved with retrieval, each the next query",
                "arxiv:2212.10509",
            ),
            (
                "flare",
                _l + "flare:FLAREPipeline",
                "FLARE: retrieve when the tentative next sentence has low-confidence tokens",
                "arxiv:2305.06983",
            ),
            (
                "adaptive",
                _l + "adaptive:AdaptiveRAGPipeline",
                "Adaptive-RAG: a complexity classifier picks no, single-step or multi-step retrieval",
                "arxiv:2403.14403",
            ),
            (
                "self_rag",
                _l + "self_rag:SelfRAGPipeline",
                "Self-RAG (prompted critic): retrieve on demand, critique each passage's answer",
                "arxiv:2310.11511",
            ),
        ]:
            self._components["agentic_pipeline"][name] = _Registration(
                description=description, paper=paper, _factory=factory, implements_contract=True
            )
            self._builtin_names.add(("agentic_pipeline", name))

        # --- Built-in Graph Nodes (deferred) ---
        _node_factories = [
            (
                "plan_retrieval",
                "Plan retrieval and query decomposition",
                "lasagna.agentic.nodes.plan_retrieval:plan_retrieval_node",
            ),
            (
                "retrieve",
                "Knowledge base retrieval hop",
                "lasagna.agentic.nodes.retrieve:retrieve_node",
            ),
            (
                "refine_chunks",
                "Chunk refinement and deduplication",
                "lasagna.agentic.nodes.refine_chunks:refine_chunks_node",
            ),
            (
                "grade_documents",
                "CRAG document quality grading",
                "lasagna.agentic.nodes.grade_documents:grade_documents_node",
            ),
            (
                "generate",
                "LLM answer generation with citations",
                "lasagna.agentic.nodes.generate:generate_node",
            ),
            (
                "rewrite_query",
                "Query rewriting for retry",
                "lasagna.agentic.nodes.rewrite_query:rewrite_query_node",
            ),
            (
                "select_chunks",
                "Citation-aware chunk selection",
                "lasagna.agentic.nodes.select_chunks:select_chunks_node",
            ),
            (
                "format_output",
                "Final output formatting",
                "lasagna.agentic.nodes.format_output:format_output_node",
            ),
            (
                "final_search",
                "Final comprehensive search",
                "lasagna.agentic.nodes.final_search:final_search_node",
            ),
            (
                "web_fallback",
                "Web search fallback (CRAG)",
                "lasagna.agentic.nodes.web_fallback:web_fallback_node",
            ),
            (
                "advance_sub_query",
                "Sub-query index advancement",
                "lasagna.agentic.nodes.advance_sub_query:advance_sub_query_node",
            ),
            (
                "check_hallucination",
                "Hallucination grounding check (ISSUP)",
                "lasagna.agentic.nodes.quality_gates:check_hallucination_node",
            ),
            (
                "check_answer_quality",
                "Answer quality check (ISUSE)",
                "lasagna.agentic.nodes.quality_gates:check_answer_quality_node",
            ),
        ]
        for node_name, desc, factory in _node_factories:
            self._callables[node_name] = _Registration(
                description=desc,
                _factory=factory,
            )

        # --- Entry-point auto-discovery ---
        self._discover_entry_points()

    def _discover_entry_points(self) -> None:
        """Auto-discover components declared via ``lasagna_components`` entry points.

        Third-party packages can register components by adding an entry point::

            [project.entry-points."lasagna_components"]
            reranker.listwise_t5 = "my_package:ListwiseT5Reranker"

        The entry point name is ``{category}.{name}`` and the value is
        the importable path to the component class.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            # Python 3.9 fallback
            try:
                from importlib_metadata import entry_points
            except ImportError:
                return

        try:
            eps = entry_points(group="lasagna_components")
        except TypeError:
            # Python 3.9: entry_points() doesn't accept group=
            eps = getattr(entry_points(), "get", lambda g: [])("lasagna_components") or []

        for ep in eps:
            try:
                parts = ep.name.split(".", 1)
                if len(parts) != 2:
                    logger.warning(
                        f"Skipping entry point '{ep.name}': expected 'category.name' format"
                    )
                    continue

                category_raw, comp_name = parts
                category = _resolve_category(category_raw)

                # Graph nodes go into the separate callable store
                if category == "graph_node":
                    obj = ep.load()
                    if comp_name not in self._callables:
                        self._callables[comp_name] = _Registration(
                            cls=obj,
                            description=f"Auto-discovered via entry point '{ep.name}'",
                        )
                        logger.debug(f"Auto-discovered graph_node '{comp_name}' from {ep.value}")
                    continue

                if category not in COMPONENT_TYPES:
                    logger.warning(
                        f"Skipping entry point '{ep.name}': unknown category '{category_raw}'"
                    )
                    continue

                cls = ep.load()
                store = self._components[category]
                if comp_name not in store:
                    store[comp_name] = _Registration(
                        cls=cls,
                        description=f"Auto-discovered via entry point '{ep.name}'",
                    )
                    logger.debug(f"Auto-discovered {category} '{comp_name}' from {ep.value}")
                _record_discovery("lasagna_components", ep.name, ep.value, ok=True)
            except Exception as exc:
                # Recorded, not just logged. A plugin author whose entry point
                # is malformed sees nothing at all otherwise: their component
                # is simply absent, and the only trace is a warning on a log
                # stream nobody keeps. `lasagna plugins list` reads this.
                _record_discovery("lasagna_components", ep.name, ep.value, ok=False, error=exc)
                logger.warning(f"Failed to load entry point '{ep.name}': {exc}")
                if os.environ.get("LASAGNA_STRICT_PLUGINS"):
                    raise

    # =========================================================================
    # Generic Decorator API
    # =========================================================================

    def register(
        self,
        component_type: str,
        name: str,
        paper: Optional[str] = None,
        claims: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> Callable:
        """Register a component class (decorator).

        Args:
            component_type: Component category (e.g. "reranker", "query_enhancer").
                "enhancer" is accepted as an alias for "query_enhancer".
            name: Unique name for this implementation.
            paper: Optional paper reference (e.g. "arxiv:2312.02969").
            claims: Optional numbers the paper reports, keyed by benchmark
                (e.g. {"beir/scifact": {"ndcg@10": 0.456}}).
            version: Optional version string.

        Returns:
            Decorator that registers the class.

        Example:
            @registry.register("reranker", "my_reranker")
            class MyReranker(BaseReranker):
                ...
        """
        canonical = _resolve_category(component_type)
        if canonical not in COMPONENT_TYPES:
            raise ValueError(
                f"Unknown component type '{component_type}'. Valid types: {sorted(COMPONENT_TYPES)}"
            )

        def decorator(cls: Type) -> Type:
            self._ensure_builtins()
            _warn_abc_conformance(canonical, name, cls)
            with self._data_lock:
                store = self._components[canonical]
                if name in store:
                    logger.warning(f"Overwriting existing {canonical} '{name}' with {cls.__name__}")
                store[name] = _Registration(
                    cls=cls,
                    description=getattr(cls, "__doc__", "") or "",
                    paper=paper,
                    claims=claims,
                    version=version,
                )
            logger.debug(f"Registered {canonical} '{name}' -> {cls.__name__}")
            return cls

        return decorator

    def register_class(
        self,
        component_type: str,
        name: str,
        cls: Type,
        description: str = "",
        override: bool = False,
        paper: Optional[str] = None,
        claims: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> None:
        """Register a component class (imperative, non-decorator).

        Args:
            component_type: Component category.
            name: Unique name for this implementation.
            cls: The component class to register.
            description: Human-readable description.
            override: Replace existing registration if True.
            paper: Optional paper reference.
            claims: Optional numbers the paper reports, keyed by benchmark.
            version: Optional version string.
        """
        canonical = _resolve_category(component_type)
        if canonical not in COMPONENT_TYPES:
            raise ValueError(
                f"Unknown component type '{component_type}'. Valid types: {sorted(COMPONENT_TYPES)}"
            )
        self._ensure_builtins()
        _warn_abc_conformance(canonical, name, cls)

        with self._data_lock:
            store = self._components[canonical]
            if name in store and not override:
                raise ValueError(
                    f"{canonical.title()} '{name}' is already registered. "
                    f"Use override=True to replace it."
                )
            replaces_builtin = (canonical, name) in self._builtin_names
            store[name] = _Registration(
                cls=cls,
                description=description,
                paper=paper,
                claims=claims,
                version=version,
            )
            # The name now holds the caller's class, so it is no longer the
            # package's: the sweep checks its contract and records it as what ran.
            self._builtin_names.discard((canonical, name))
        if replaces_builtin:
            logger.warning(
                f"{canonical} '{name}' replaced the built-in of that name; every sweep "
                f"in this process that names '{name}' now runs {cls.__name__}"
            )
        logger.debug(f"Registered {canonical} '{name}' -> {cls.__name__}")

    # =========================================================================
    # Generic Lookup API
    # =========================================================================

    def get(
        self,
        component_type: str,
        name: str,
        **kwargs,
    ) -> Any:
        """Get an instantiated component by type and name.

        Uses adaptive parameter filtering: if the constructor doesn't
        accept all provided kwargs, filters to accepted parameters.

        Args:
            component_type: Component category.
            name: Registered name.
            **kwargs: Constructor arguments.

        Returns:
            Instantiated component.

        Raises:
            KeyError: If component not found.
        """
        canonical = _resolve_category(component_type)
        if canonical not in COMPONENT_TYPES:
            raise ValueError(
                f"Unknown component type '{component_type}'. Valid types: {sorted(COMPONENT_TYPES)}"
            )
        self._ensure_builtins()
        store = self._components[canonical]
        if name not in store:
            available = list(store.keys())
            raise KeyError(f"No {canonical} named '{name}'. Available: {available}")
        return self._create_instance(canonical, store, name, **kwargs)

    def get_class(
        self,
        component_type: str,
        name: str,
    ) -> Type:
        """Get the component class (without instantiating).

        Args:
            component_type: Component category.
            name: Registered name.

        Returns:
            Component class.

        Raises:
            KeyError: If component not found.
        """
        canonical = _resolve_category(component_type)
        if canonical not in COMPONENT_TYPES:
            raise ValueError(
                f"Unknown component type '{component_type}'. Valid types: {sorted(COMPONENT_TYPES)}"
            )
        self._ensure_builtins()
        store = self._components[canonical]
        if name not in store:
            available = list(store.keys())
            raise KeyError(f"No {canonical} named '{name}'. Available: {available}")
        return store[name].resolve()

    def list(self, component_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List registered components.

        Args:
            component_type: Optional filter by type. If None, lists all.

        Returns:
            Dict mapping component_type -> list of registered names.
        """
        self._ensure_builtins()
        if component_type:
            canonical = _resolve_category(component_type)
            if canonical not in COMPONENT_TYPES:
                raise ValueError(
                    f"Unknown component type '{component_type}'. "
                    f"Valid types: {sorted(COMPONENT_TYPES)}"
                )
            return {canonical: self._listed(self._components[canonical])}
        return {ct: self._listed(names) for ct, names in self._components.items() if names}

    def list_all(self) -> Dict[str, List[str]]:
        """List all registered components across all types."""
        self._ensure_builtins()
        return {ct: self._listed(names) for ct, names in self._components.items()}

    def list_components(self, component_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every listed component of one stage or all, with what it declares.

        Each entry is what the stage's ``list_*`` helper returns, plus
        ``stage`` and ``builtin``. Stages are in name order. Imports no
        component, so it is safe for a CLI listing.
        """
        self._ensure_builtins()
        if component_type:
            stage = _resolve_category(component_type)
            if stage not in COMPONENT_TYPES:
                raise ValueError(
                    f"Unknown component type '{component_type}'. "
                    f"Valid types: {sorted(COMPONENT_TYPES)}"
                )
            stages = [stage]
        else:
            stages = sorted(COMPONENT_TYPES)
        return [
            {"stage": stage, "builtin": (stage, entry["name"]) in self._builtin_names, **entry}
            for stage in stages
            for entry in self._list_category(self._components[stage], stage)
        ]

    @staticmethod
    def _listed(store: Dict[str, "_Registration"]) -> List[str]:
        return [name for name, reg in store.items() if not reg.hidden]

    def provenance(self, component_type: str, name: str) -> Dict[str, Any]:
        """``{"paper", "claims", "version"}`` a component declared, the ones it set.

        Empty for an unknown name or a component that declared none. Resolves
        nothing.
        """
        canonical = _resolve_category(component_type)
        self._ensure_builtins()
        reg = self._components.get(canonical, {}).get(name)
        if reg is None:
            return {}
        declared = {"paper": reg.paper, "claims": reg.claims, "version": reg.version}
        return {key: value for key, value in declared.items() if value}

    def is_builtin(self, component_type: str, name: str) -> bool:
        """Whether the package itself registered this name (not a plugin or user)."""
        self._ensure_builtins()
        return (_resolve_category(component_type), name) in self._builtin_names

    def has(self, component_type: str, name: str) -> bool:
        """Check if a component is registered.

        Raises:
            ValueError: If component_type is not a valid type.
        """
        canonical = _resolve_category(component_type)
        if canonical not in COMPONENT_TYPES:
            raise ValueError(
                f"Unknown component type '{component_type}'. Valid types: {sorted(COMPONENT_TYPES)}"
            )
        self._ensure_builtins()
        return name in self._components[canonical]

    def clear(self, component_type: Optional[str] = None) -> None:
        """Clear registered components.

        Args:
            component_type: If provided, clear only this type. Otherwise clear all.
                Use "graph_node" to clear only the callable node store.
        """
        with self._data_lock:
            if component_type:
                if component_type == "graph_node":
                    self._callables.clear()
                else:
                    canonical = _resolve_category(component_type)
                    if canonical in self._components:
                        self._components[canonical].clear()
            else:
                for ct in self._components:
                    self._components[ct].clear()
                self._callables.clear()
                self._builtins_registered = False

    # =========================================================================
    # Type-Specific Convenience API (backward compat with retrieval registry)
    # =========================================================================

    def register_reranker(
        self,
        name: str,
        cls: Type,
        description: str = "",
        override: bool = False,
    ) -> None:
        """Register a custom reranker."""
        self.register_class("reranker", name, cls, description=description, override=override)

    def register_embedder(
        self,
        name: str,
        cls: Type,
        description: str = "",
        override: bool = False,
    ) -> None:
        """Register a custom embedding model."""
        self.register_class("embedder", name, cls, description=description, override=override)

    def register_chunker(
        self,
        name: str,
        cls: Type,
        description: str = "",
        override: bool = False,
    ) -> None:
        """Register a custom document chunker."""
        self.register_class("chunker", name, cls, description=description, override=override)

    def register_enhancer(
        self,
        name: str,
        cls: Type,
        description: str = "",
        override: bool = False,
    ) -> None:
        """Register a custom query enhancer (alias for query_enhancer)."""
        self.register_class("query_enhancer", name, cls, description=description, override=override)

    def get_reranker(self, name: str) -> Optional[Type]:
        """Get reranker class by name."""
        self._ensure_builtins()
        reg = self._components["reranker"].get(name)
        return reg.resolve() if reg else None

    def get_embedder(self, name: str) -> Optional[Type]:
        """Get embedder class by name."""
        self._ensure_builtins()
        reg = self._components["embedder"].get(name)
        return reg.resolve() if reg else None

    def get_chunker(self, name: str) -> Optional[Type]:
        """Get chunker class by name."""
        self._ensure_builtins()
        reg = self._components["chunker"].get(name)
        return reg.resolve() if reg else None

    def get_enhancer(self, name: str) -> Optional[Type]:
        """Get query enhancer class by name."""
        self._ensure_builtins()
        reg = self._components["query_enhancer"].get(name)
        return reg.resolve() if reg else None

    def list_rerankers(self) -> List[Dict[str, str]]:
        """List all registered rerankers with descriptions."""
        self._ensure_builtins()
        return self._list_category(self._components["reranker"], "reranker")

    def list_embedders(self) -> List[Dict[str, str]]:
        """List all registered embedders with descriptions."""
        self._ensure_builtins()
        return self._list_category(self._components["embedder"], "embedder")

    def list_chunkers(self) -> List[Dict[str, str]]:
        """List all registered chunkers with descriptions."""
        self._ensure_builtins()
        return self._list_category(self._components["chunker"], "chunker")

    def list_enhancers(self) -> List[Dict[str, str]]:
        """List all registered query enhancers with descriptions."""
        self._ensure_builtins()
        return self._list_category(self._components["query_enhancer"], "query_enhancer")

    def list_retrievers(self) -> List[Dict[str, str]]:
        """List all registered retrievers with descriptions."""
        self._ensure_builtins()
        return self._list_category(self._components["retriever"], "retriever")

    def list_document_processors(self) -> List[Dict[str, str]]:
        """List all registered document processors with descriptions."""
        self._ensure_builtins()
        return self._list_category(self._components["document_processor"], "document_processor")

    def create_reranker(self, name: str, **kwargs) -> Any:
        """Create a reranker instance by name."""
        return self.get("reranker", name, **kwargs)

    def create_embedder(self, name: str, **kwargs) -> Any:
        """Create an embedder instance by name."""
        return self.get("embedder", name, **kwargs)

    def create_chunker(self, name: str, **kwargs) -> Any:
        """Create a chunker instance by name."""
        return self.get("chunker", name, **kwargs)

    def create_enhancer(self, name: str, **kwargs) -> Any:
        """Create a query enhancer instance by name."""
        return self.get("query_enhancer", name, **kwargs)

    def unregister_reranker(self, name: str) -> bool:
        """Remove a registered reranker."""
        return self._unregister("reranker", name)

    def unregister_embedder(self, name: str) -> bool:
        """Remove a registered embedder."""
        return self._unregister("embedder", name)

    def unregister_chunker(self, name: str) -> bool:
        """Remove a registered chunker."""
        return self._unregister("chunker", name)

    def unregister_enhancer(self, name: str) -> bool:
        """Remove a registered query enhancer."""
        return self._unregister("query_enhancer", name)

    # =========================================================================
    # Agentic Pipeline Convenience API
    # =========================================================================

    def register_pipeline(
        self,
        name: str,
        cls: Type,
        description: str = "",
        override: bool = False,
        paper: Optional[str] = None,
        claims: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a custom agentic pipeline."""
        self.register_class(
            "agentic_pipeline",
            name,
            cls,
            description=description,
            override=override,
            paper=paper,
            claims=claims,
        )

    def get_pipeline(self, name: str) -> Optional[Type]:
        """Get agentic pipeline class by name."""
        self._ensure_builtins()
        reg = self._components["agentic_pipeline"].get(name)
        return reg.resolve() if reg else None

    def list_pipelines(self) -> List[Dict[str, str]]:
        """List all registered agentic pipelines with descriptions.

        The category argument is what makes ``implements_contract`` appear;
        omitting it, as this used to, silently dropped the interchangeability
        flag for this one stage while every sibling accessor carried it.
        """
        self._ensure_builtins()
        return self._list_category(self._components["agentic_pipeline"], "agentic_pipeline")

    # =========================================================================
    # Graph Node Callable API (separate from COMPONENT_TYPES)
    # =========================================================================

    def register_graph_node(
        self,
        name: str,
        fn: Callable,
        description: str = "",
        override: bool = False,
    ) -> None:
        """Register a graph node callable.

        Unlike component classes, graph nodes are stored in a separate
        callable registry and are never instantiated via ``get()``.

        Args:
            name: Node name (must match a valid graph node slot).
            fn: Async callable with signature ``(state: dict) -> dict``.
            description: Human-readable description.
            override: Replace existing registration if True.
        """
        if not callable(fn):
            raise TypeError(f"Graph node '{name}' must be callable, got {type(fn).__name__}")
        self._ensure_builtins()
        with self._data_lock:
            if name in self._callables and not override:
                raise ValueError(
                    f"Graph node '{name}' is already registered. Use override=True to replace it."
                )
            self._callables[name] = _Registration(
                cls=fn,
                description=description,
            )
        logger.debug(f"Registered graph_node '{name}' -> {getattr(fn, '__name__', repr(fn))}")

    def get_graph_node(self, name: str) -> Callable:
        """Return the registered graph node callable (NOT instantiated).

        Raises:
            KeyError: If node not found.
        """
        self._ensure_builtins()
        reg = self._callables.get(name)
        if not reg:
            available = list(self._callables.keys())
            raise KeyError(f"No graph_node named '{name}'. Available: {available}")
        # Resolve deferred factory if needed
        return reg.resolve()

    def has_graph_node(self, name: str) -> bool:
        """Check if a graph node is registered."""
        self._ensure_builtins()
        return name in self._callables

    def list_graph_nodes(self) -> List[Dict[str, str]]:
        """List all registered graph nodes with descriptions."""
        self._ensure_builtins()
        return self._list_category(self._callables)

    # =========================================================================
    # Reset (for testing)
    # =========================================================================

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. Primarily for testing."""
        with cls._lock:
            cls._instance = None

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _unregister(self, component_type: str, name: str) -> bool:
        canonical = _resolve_category(component_type)
        self._ensure_builtins()
        with self._data_lock:
            store = self._components.get(canonical, {})
            if name in store:
                del store[name]
                return True
            return False

    @staticmethod
    def _list_category(
        store: Dict[str, _Registration], category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        expected_abc = _get_expected_abc(category) if category else None
        result = []
        for name, reg in store.items():
            if reg.hidden:
                continue
            if reg.cls is not None:
                cls_name = reg.cls.__name__
            elif reg._factory:
                # Extract class name from "module.path:ClassName" factory string
                cls_name = reg._factory.rsplit(":", 1)[-1] if ":" in reg._factory else reg._factory
                cls_name = cls_name or "deferred"
            else:
                cls_name = "deferred"
            entry: Dict[str, Any] = {
                "name": name,
                "description": reg.description,
                "class": cls_name,
            }
            # Whether this component implements its stage's ABC — i.e. whether
            # it is actually interchangeable with another component of the same
            # type. Some built-ins predate the typed contracts and take/return
            # different shapes, so comparing one against a contract-compliant
            # component is not like-for-like. Surfacing it lets a caller tell
            # the difference instead of finding out from a TypeError.
            if expected_abc is not None:
                # Never resolved here: importing every deferred factory to ask
                # `issubclass` pulled in torch for a listing. Built-ins declare
                # the answer; a registered class is already loaded. A deferred
                # entry that declares nothing is not claimed.
                if reg.implements_contract is not None:
                    entry["implements_contract"] = reg.implements_contract
                else:
                    cls = reg.cls
                    entry["implements_contract"] = (
                        issubclass(cls, expected_abc) if isinstance(cls, type) else False
                    )
            # Research provenance, included only when the component declared it,
            # so listings of built-in components stay uncluttered. This is what
            # lets a results table put a paper's claimed metric next to the one
            # actually reproduced here.
            if reg.paper:
                entry["paper"] = reg.paper
            if reg.claims:
                entry["claims"] = reg.claims
            if reg.version:
                entry["version"] = reg.version
            result.append(entry)
        return result

    def _create_instance(
        self,
        category: str,
        store: Dict[str, _Registration],
        name: str,
        **kwargs,
    ) -> Any:
        """Create an instance with adaptive parameter filtering."""
        reg = store.get(name)
        if not reg:
            available = ", ".join(store.keys())
            raise ValueError(f"Unknown {category} '{name}'. Available: {available or 'none'}")

        # Resolve deferred factory if needed
        resolved_cls = reg.resolve()

        # Try full kwargs first; if the constructor doesn't accept them,
        # filter to only accepted parameters.  We only catch TypeErrors
        # that look like argument-mismatch errors; other TypeErrors
        # (e.g. bugs inside the constructor) are re-raised immediately.
        try:
            return resolved_cls(**kwargs)
        except TypeError as exc:
            if not kwargs:
                raise  # No kwargs to filter — this is a real constructor bug
            err_msg = str(exc)
            if "argument" not in err_msg and "unexpected keyword" not in err_msg:
                raise  # Doesn't look like a signature mismatch, re-raise

            # Introspect the constructor to filter kwargs.
            try:
                sig = inspect.signature(resolved_cls.__init__)
            except (ValueError, TypeError):
                # Can't introspect (C extension, etc.) — re-raise original
                raise exc from None

            params = sig.parameters
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )

            if accepts_var_keyword:
                # object.__init__ reports (*args, **kwargs) but actually
                # rejects kwargs at runtime when no custom __init__ is
                # defined.  Detect this by checking the MRO.
                has_custom_init = any(
                    "__init__" in klass.__dict__
                    for klass in resolved_cls.__mro__
                    if klass is not object
                )
                if has_custom_init:
                    raise  # Real **kwargs constructor rejected our input
                # No custom __init__ — all kwargs are irrelevant
                logger.debug(
                    f"Constructor for {category} '{name}' inherits "
                    f"object.__init__, dropping kwargs: {set(kwargs)}"
                )
                return resolved_cls()

            accepted = {k: v for k, v in kwargs.items() if k in params}
            logger.debug(
                f"Constructor for {category} '{name}' doesn't accept all kwargs, "
                f"filtered {set(kwargs) - set(accepted)}"
            )
            return resolved_cls(**accepted)


# =============================================================================
# Module-level Singleton
# =============================================================================

#: Global unified component registry.
registry = ComponentRegistry()

# Backward-compat aliases used by the retrieval registry API.
component_registry = registry


def get_component_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return registry
