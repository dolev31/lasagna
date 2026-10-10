"""Fast corpus ingestion pipeline for evaluation experiments.

Bypasses the standard client.upload() → DocumentPipeline path to eliminate
per-document overhead (temp-file I/O, quota reservation, metadata DB writes).
Instead, chunks in-memory text directly, batches embeddings across documents,
and batch-inserts into the vector store.

3-stage pipeline:
  Stage A (chunk)  — N parallel workers via ThreadPoolExecutor
  Stage B (embed)  — 1 thread, cross-doc batching
  Stage C (insert) — 1 thread, batch insert + BM25 append

Thread safety:
  - Non-daemon threads with explicit error channel
  - Hard timeout enforcement
  - No queue.empty() flush logic — deterministic count-based triggers
"""

from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Stage A: chunk one document (stateless, thread-safe for non-semantic)
# ---------------------------------------------------------------------------


#: More failed documents than this share stops the ingest instead of
#: evaluating on what is left.
_MAX_CHUNK_FAILURE_RATE = 0.1


def _chunk_one_doc(
    corpus_id: str,
    text: str,
    valid_id: str,
    session_id: str,
    filename: str,
    config,
    embedding_manager=None,
    chunker=None,
) -> Tuple[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Chunk one document from in-memory text.

    ``chunker`` is the arm's chunker component, when it named one; otherwise
    the built-in dispatch on the config's chunking settings runs.

    Returns:
        (valid_id, child_chunks, parent_chunks_or_None)
        where child_chunks is a list of dicts with keys:
        content, document_id, session_id, filename, metadata, chunk_index,
        start_char, end_char
    """
    from ..processing.chunker.semantic import chunk_document

    enable_hierarchical = getattr(config, "enable_hierarchical", False)
    if chunker is not None:
        # The component owns the split; hierarchical parents are the built-in
        # chunker's feature, not layered on top of someone else's algorithm.
        enable_hierarchical = False
        result = [
            (
                c.content,
                {
                    **(c.metadata or {}),
                    "chunk_index": i,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                },
            )
            for i, c in enumerate(chunker.chunk(text))
        ]
    else:
        result = chunk_document(
            text=text,
            document_id=valid_id,
            session_id=session_id,
            filename=filename,
            embedding_manager=embedding_manager,
            config=config,
            return_parents=enable_hierarchical,
        )

    if enable_hierarchical and isinstance(result, tuple):
        child_tuples, parent_dicts = result
    else:
        child_tuples = result if not isinstance(result, tuple) else result
        parent_dicts = None

    # Convert (text, metadata) tuples to chunk dicts
    chunk_dicts = []
    for idx, (content, metadata) in enumerate(child_tuples):
        chunk_dicts.append(
            {
                "content": content,
                "document_id": valid_id,
                "session_id": session_id,
                "filename": filename,
                "metadata": metadata,
                "chunk_index": idx,
                "start_char": metadata.get("start_char", 0),
                "end_char": metadata.get("end_char", len(content)),
            }
        )

    # --- Optional enrichment features (run per-doc in Stage A) ---

    # Contextual retrieval
    if getattr(config, "enable_contextual_retrieval", False) and text:
        try:
            from ..processing.contextual import ContextualEnricher
            from ..providers.llm_provider import SimpleLLMProvider

            llm = SimpleLLMProvider().get_llm()
            enricher = ContextualEnricher(llm_provider=llm)
            # The same document budget as the contextual chunker component, so
            # the flag and the component measure the same method.
            limit = getattr(config, "contextual_max_document_chars", 10000)
            doc_context = text if limit is None else text[:limit]
            chunk_texts = [c["content"] for c in chunk_dicts]
            enriched_texts = enricher.enrich_batch(doc_context, chunk_texts)
            for chunk, enriched in zip(chunk_dicts, enriched_texts):
                chunk["content"] = enriched
        except Exception as e:
            from ..core.strict import strict_stages

            if strict_stages():
                # An evaluation of contextual retrieval that quietly ingested
                # plain chunks would score the baseline under its name.
                raise RuntimeError(f"Contextual enrichment failed: {e}") from e
            logger.warning(f"Contextual enrichment failed for {valid_id}: {e}")

    # GraphRAG entity extraction
    if getattr(config, "enable_graph_retrieval", False):
        try:
            from ..retrieval.graph import EntityExtractor, get_or_create_session_graph
            from ..providers.llm_provider import SimpleLLMProvider

            from ..utils.model_utils import get_default_model_name

            extraction_model = (
                getattr(config, "graph_extraction_model", "") or get_default_model_name()
            )
            extractor = EntityExtractor(
                entity_types=getattr(config, "graph_entity_types", None),
                model_name=extraction_model,
            )
            llm = SimpleLLMProvider().get_llm(model_name=extraction_model, temperature=0)
            chunk_texts = [c["content"] for c in chunk_dicts]
            extractions = extractor.extract_batch_sync(chunk_texts, llm, max_workers=4)
            graph_store = get_or_create_session_graph(session_id)
            for idx, extraction in enumerate(extractions):
                chunk_id = f"{valid_id}_chunk_{idx}"
                for entity in extraction.get("entities", []):
                    # Note: encode_document uses _encode_lock — this is why
                    # enable_graph_retrieval forces workers=1.
                    entity_emb = embedding_manager.encode_document(entity["name"])
                    graph_store.add_entity(
                        entity_id=entity["name"],
                        entity_type=entity.get("type", "UNKNOWN"),
                        embedding=entity_emb,
                        chunk_id=chunk_id,
                    )
                for rel in extraction.get("relationships", []):
                    graph_store.add_relationship(
                        source_id=rel["source"],
                        target_id=rel["target"],
                        rel_type=rel.get("type", "RELATED"),
                    )
        except Exception as e:
            logger.warning(f"GraphRAG extraction failed for {valid_id}: {e}")

    # RAPTOR (runs embedding internally — forces workers=1)
    if getattr(config, "enable_raptor", False):
        try:
            from ..retrieval.raptor import RaptorTreeBuilder
            from ..providers.llm_provider import SimpleLLMProvider

            llm = SimpleLLMProvider().get_llm(temperature=0)
            builder = RaptorTreeBuilder(
                embedding_manager=embedding_manager,
                llm_provider=llm,
                max_cluster_size=getattr(config, "raptor_max_cluster_size", 10),
                max_levels=getattr(config, "raptor_max_levels", 5),
            )
            chunk_texts = [c["content"] for c in chunk_dicts]
            # Need embeddings for RAPTOR — compute them now (single-threaded)
            chunk_embeddings = embedding_manager.encode_documents_batch(
                chunk_texts,
                show_progress=False,
            )
            tree = builder.build(texts=chunk_texts, embeddings=chunk_embeddings)
            for node in tree.all_nodes():
                if node.level == 0:
                    continue  # leaf nodes already in chunk_dicts
                raptor_idx = len(chunk_dicts)
                chunk_dicts.append(
                    {
                        "content": node.text,
                        "document_id": valid_id,
                        "session_id": session_id,
                        "filename": filename,
                        "metadata": {
                            "document_id": valid_id,
                            "session_id": session_id,
                            "filename": filename,
                            "raptor_level": node.level,
                            "raptor_node_id": node.node_id,
                            "raptor_children": len(node.children),
                        },
                        "chunk_index": raptor_idx,
                        "start_char": 0,
                        "end_char": len(node.text),
                        "_raptor_embedding": node.embedding,  # pre-computed
                    }
                )
        except Exception as e:
            logger.warning(f"RAPTOR tree building failed for {valid_id}: {e}")

    return valid_id, chunk_dicts, parent_dicts


# ---------------------------------------------------------------------------
# Stage B: single-threaded cross-doc embedding
# ---------------------------------------------------------------------------


class _Progress:
    """Periodic "N/M documents, ETA" lines while a corpus is embedded.

    A 5,000-document corpus takes half an hour on CPU, and it used to print
    nothing at the default log level for all of it, so healthy runs were
    killed -- and an interrupted run is the one that leaves a store behind.
    Tagged ``progress`` so the CLI shows it under its warnings-only default.
    """

    def __init__(self, total: int, every_s: float = 30.0):
        self.total, self.every_s = total, every_s
        self.t0 = self.last = time.perf_counter()

    def update(self, done: int) -> None:
        now = time.perf_counter()
        if now - self.last < self.every_s and done < self.total:
            return
        self.last = now
        elapsed = now - self.t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        logger.bind(progress=True).info(
            f"Ingesting corpus: {done}/{self.total} documents embedded "
            f"({rate * 60:.0f}/min, about {eta / 60:.0f} min left)"
            if done < self.total
            else f"Ingesting corpus: {done}/{self.total} documents embedded in "
            f"{elapsed / 60:.1f} min"
        )


def _embed_stage(
    chunk_queue: queue.Queue,
    insert_queue: queue.Queue,
    embedding_manager,
    embed_batch_size: int,
    total_docs: int,
    enable_late_chunking: bool,
    enable_sparse: bool,
    max_content_length: int,
    embedder=None,
):
    """Drain chunk_queue, batch-embed cross-doc, push to insert_queue."""
    pending_chunks: List[Dict[str, Any]] = []
    docs_processed = 0
    progress = _Progress(total_docs)

    while docs_processed < total_docs:
        item = chunk_queue.get()
        doc_id, chunks, full_text = item
        docs_processed += 1

        # Assign chunk_id here (sequential per doc, after all enrichment)
        for idx, chunk in enumerate(chunks):
            if "chunk_index" in chunk:
                chunk["chunk_id"] = f"{chunk['document_id']}_chunk_{chunk['chunk_index']}"
            else:
                chunk["chunk_id"] = f"{chunk['document_id']}_chunk_{idx}"
            # Truncate content
            chunk["content"] = chunk["content"][:max_content_length]

        # Store full_text for late_chunking
        if enable_late_chunking and full_text:
            for chunk in chunks:
                chunk["_full_text"] = full_text

        pending_chunks.extend(chunks)

        # Flush on size threshold or last doc
        if len(pending_chunks) >= embed_batch_size or docs_processed == total_docs:
            _embed_and_forward(
                pending_chunks,
                embedding_manager,
                insert_queue,
                enable_late_chunking,
                enable_sparse,
                embedder=embedder,
            )
            pending_chunks = []
            progress.update(docs_processed)

    # Final flush (in case of rounding)
    if pending_chunks:
        _embed_and_forward(
            pending_chunks,
            embedding_manager,
            insert_queue,
            enable_late_chunking,
            enable_sparse,
            embedder=embedder,
        )
    insert_queue.put(None)  # sentinel


def _embed_and_forward(
    chunks: List[Dict[str, Any]],
    embedding_manager,
    insert_queue: queue.Queue,
    enable_late_chunking: bool,
    enable_sparse: bool,
    embedder=None,
):
    """Embed a batch of chunks and push to insert_queue.

    ``embedder`` is the arm's embedder component, when it named one: the
    corpus is embedded by the same model that embeds the queries.
    """
    if not chunks:
        return

    # Separate chunks with pre-computed embeddings (RAPTOR nodes)
    need_embedding = []
    precomputed = []
    for chunk in chunks:
        if "_raptor_embedding" in chunk:
            precomputed.append(chunk)
        else:
            need_embedding.append(chunk)

    # Embed chunks that need it
    if need_embedding:
        texts = [c["content"] for c in need_embedding]
        if embedder is not None:
            vectors = np.asarray(embedder.encode(list(texts)), dtype=np.float32)
            for chunk, emb in zip(need_embedding, vectors):
                chunk["embedding"] = emb
        elif enable_late_chunking and need_embedding[0].get("_full_text"):
            # Group by document for late chunking
            doc_groups: Dict[str, List[Dict[str, Any]]] = {}
            for chunk in need_embedding:
                did = chunk["document_id"]
                doc_groups.setdefault(did, []).append(chunk)
            for did, group in doc_groups.items():
                full_text = group[0].get("_full_text", "")
                group_texts = [c["content"] for c in group]
                embeddings = embedding_manager.encode_documents_late_chunking(
                    full_text=full_text,
                    chunk_texts=group_texts,
                )
                for chunk, emb in zip(group, embeddings):
                    chunk["embedding"] = emb
        else:
            embeddings = embedding_manager.encode_documents_batch(
                texts,
                show_progress=False,
            )
            for chunk, emb in zip(need_embedding, embeddings):
                chunk["embedding"] = emb

    # Assign pre-computed embeddings
    for chunk in precomputed:
        chunk["embedding"] = chunk.pop("_raptor_embedding")

    # Sparse tokens (OpenSearch)
    if enable_sparse:
        try:
            all_texts = [c["content"] for c in chunks]
            sparse_results = embedding_manager.encode_sparse(all_texts)
            if sparse_results:
                for chunk, sparse in zip(chunks, sparse_results):
                    chunk["sparse_tokens"] = sparse
        except Exception as e:
            logger.debug(f"Sparse token generation skipped: {e}")

    # Clean up internal fields before forwarding
    for chunk in chunks:
        chunk.pop("_full_text", None)

    insert_queue.put(list(chunks))


# ---------------------------------------------------------------------------
# Stage C: single-threaded batch insert
# ---------------------------------------------------------------------------


def _insert_stage(
    insert_queue: queue.Queue,
    resolved_backend,
    collection_name: str,
    insert_batch_size: int,
    hybrid_searcher,
    session_id: str,
    enable_hybrid: bool,
    partition_key: Optional[str],
) -> Tuple[int, Set[str]]:
    """Drain insert_queue, batch-insert into vector store.

    Returns:
        (total_chunks_inserted, set_of_ingested_doc_ids)
    """
    pending: List[Dict[str, Any]] = []
    total_inserted = 0
    ingested_doc_ids: Set[str] = set()

    while True:
        batch = insert_queue.get()
        if batch is None:  # sentinel
            break
        pending.extend(batch)
        if len(pending) >= insert_batch_size:
            n = _flush_insert(
                pending,
                resolved_backend,
                collection_name,
                hybrid_searcher,
                session_id,
                enable_hybrid,
                partition_key,
            )
            total_inserted += n
            for chunk in pending:
                ingested_doc_ids.add(chunk["document_id"])
            pending = []

    # Final flush
    if pending:
        n = _flush_insert(
            pending,
            resolved_backend,
            collection_name,
            hybrid_searcher,
            session_id,
            enable_hybrid,
            partition_key,
        )
        total_inserted += n
        for chunk in pending:
            ingested_doc_ids.add(chunk["document_id"])

    return total_inserted, ingested_doc_ids


def _flush_insert(
    chunks: List[Dict[str, Any]],
    resolved_backend,
    collection_name: str,
    hybrid_searcher,
    session_id: str,
    enable_hybrid: bool,
    partition_key: Optional[str],
) -> int:
    """Prepare and insert a batch of chunks into the vector store."""
    if not chunks:
        return 0

    # Build insert payload (exact parity with _document_pipeline.py:570-579)
    batch_data = []
    bm25_docs = []
    for chunk in chunks:
        embedding = chunk["embedding"]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        insert_doc = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "session_id": chunk.get("session_id", session_id),
            "content": chunk["content"],
            "embedding": embedding,
            "metadata": chunk.get("metadata", {}),
        }
        # The column and metadata["session_id"] must name the same session:
        # search filters on the column, and a chunker's own metadata must not
        # move a chunk into another session.
        insert_doc["metadata"]["session_id"] = insert_doc["session_id"]

        # Optional sparse tokens
        if "sparse_tokens" in chunk and chunk["sparse_tokens"]:
            insert_doc["sparse_tokens"] = chunk["sparse_tokens"]

        batch_data.append(insert_doc)

        bm25_docs.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
            }
        )

    # Vector store insert
    insert_kwargs = {
        "collection_name": collection_name,
        "data": batch_data,
    }
    if partition_key:
        insert_kwargs["partition_key"] = partition_key
    resolved_backend.insert(**insert_kwargs)

    # BM25 append (deferred build)
    if enable_hybrid and hybrid_searcher:
        hybrid_searcher.index_chunks(bm25_docs, session_id, build_now=False)

    return len(batch_data)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def fast_ingest_corpus(
    corpus: Dict[str, str],
    corpus_to_valid: Dict[str, str],
    ready_ids: Set[str],
    session_id: str,
    config,
    backend,
    chunker=None,
    embedder=None,
) -> Tuple[int, Set[str], Set[str]]:
    """3-stage fast corpus ingestion pipeline.

    Args:
        corpus: corpus_id → full text
        corpus_to_valid: corpus_id → valid_id (document_id for vector store)
        ready_ids: doc IDs already indexed (skip these)
        session_id: shared session ID for this corpus
        config: RAGConfig (with session applied)
        backend: RAGBackend instance
        chunker: The arm's chunker component, or None for the built-in.
        embedder: The arm's embedder component, or None for the built-in.

    Returns:
        (total_chunks_inserted, ingested_doc_ids, error_doc_ids)
    """
    # --- Setup (once, before threads) ---
    backend._collection_manager.maybe_reload_embedding_manager(config)
    embedding_manager = backend._collection_manager.embedding_manager

    resolved_backend = backend._collection_manager.resolve_backend(session_id)

    from ..core import tenant
    from ..core.config import rag_config

    # Search reads only the session's partition under tenant_isolation =
    # "partition", so chunks inserted anywhere else are invisible to it. This
    # used to import from a module that does not define the function and treat
    # the ImportError as "no partition": benchmark chunks went to the default
    # partition and retrieval returned nothing, with no error. Same handling as
    # the upload path (api/_document_pipeline.py).
    tenant_mgr = tenant.get_tenant_manager()
    partition_key = tenant_mgr.resolve_partition_key(session_id)
    if partition_key:
        try:
            tenant_mgr.ensure_partition(resolved_backend, config.collection_name, partition_key)
        except Exception as e:
            raise ValueError(f"Failed to create partition for session {session_id}: {e}") from e

    enable_late_chunking = getattr(config, "enable_late_chunking", False)
    enable_hybrid = getattr(config, "enable_hybrid_search", True)
    enable_sparse = (
        hasattr(resolved_backend, "backend_name") and resolved_backend.backend_name == "opensearch"
    )
    max_content_length = getattr(config, "max_content_length", 4000)

    # Determine worker count
    _uses_llm_features = (
        getattr(config, "enable_contextual_retrieval", False)
        or getattr(config, "enable_graph_retrieval", False)
        or getattr(config, "enable_raptor", False)
    )
    chunking_strategy = getattr(config, "chunking_strategy", "fixed")
    _uses_semantic = chunking_strategy in ("semantic",)
    # One worker for a component: nothing says a researcher's chunker is
    # thread-safe, and a deterministic corpus matters more than speed here.
    n_workers = (
        1
        if (_uses_semantic or _uses_llm_features or chunker is not None)
        else rag_config.ingest_extract_workers
    )
    embed_batch_size = rag_config.ingest_embed_batch_size
    insert_batch_size = rag_config.ingest_insert_batch_size

    # Build doc list (skip already-indexed)
    docs_to_ingest = [
        (cid, text, corpus_to_valid[cid])
        for cid, text in corpus.items()
        if corpus_to_valid[cid] not in ready_ids
    ]
    total_docs = len(docs_to_ingest)

    if total_docs == 0:
        return 0, set(), set()

    logger.info(
        f"Fast ingest: {total_docs} docs, workers={n_workers}, "
        f"embed_batch={embed_batch_size}, insert_batch={insert_batch_size}, "
        f"strategy={chunking_strategy}"
    )

    # --- Pipeline execution (blocking, wrapped in asyncio.to_thread) ---
    def _run_pipeline() -> Tuple[int, Set[str], Set[str]]:
        chunk_queue: queue.Queue = queue.Queue(maxsize=max(n_workers * 2, 4))
        insert_queue: queue.Queue = queue.Queue(maxsize=32)
        error_channel: queue.Queue = queue.Queue()
        error_doc_ids: Set[str] = set()
        all_parent_chunks: List[Dict[str, Any]] = []
        parent_chunks_lock = threading.Lock()

        # Wrap stages with error handling
        def _embed_safe():
            try:
                _embed_stage(
                    chunk_queue,
                    insert_queue,
                    embedding_manager,
                    embed_batch_size,
                    total_docs,
                    enable_late_chunking,
                    enable_sparse,
                    max_content_length,
                    embedder=embedder,
                )
            except Exception as exc:
                error_channel.put(("embed", exc))
                insert_queue.put(None)  # unblock C

        insert_result = [0, set()]  # [total_chunks, ingested_doc_ids]

        def _insert_safe():
            try:
                total, doc_ids = _insert_stage(
                    insert_queue,
                    resolved_backend,
                    config.collection_name,
                    insert_batch_size,
                    backend.hybrid_searcher,
                    session_id,
                    enable_hybrid,
                    partition_key,
                )
                insert_result[0] = total
                insert_result[1] = doc_ids
            except Exception as exc:
                error_channel.put(("insert", exc))

        # Start B and C (non-daemon)
        # Each thread runs in a copy of this context, so an evaluation's strict
        # stages reach model loads and enrichment there too (threads do not
        # inherit context variables on their own).
        embed_thread = threading.Thread(
            target=contextvars.copy_context().run, args=(_embed_safe,), name="ingest-embed"
        )
        insert_thread = threading.Thread(
            target=contextvars.copy_context().run, args=(_insert_safe,), name="ingest-insert"
        )
        embed_thread.start()
        insert_thread.start()

        # Stage A: parallel chunking
        t0 = time.perf_counter()
        first_chunk_error = ""
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ingest-chunk") as pool:
            futures = []
            for corpus_id, text, valid_id in docs_to_ingest:
                fut = pool.submit(
                    contextvars.copy_context().run,
                    _chunk_one_doc,
                    corpus_id,
                    text,
                    valid_id,
                    session_id,
                    f"{valid_id}.txt",
                    config,
                    embedding_manager if (_uses_semantic or _uses_llm_features) else None,
                    chunker,
                )
                futures.append((valid_id, text, fut))

            for i, (valid_id, text, fut) in enumerate(futures):
                try:
                    doc_id, chunks, parent_chunks = fut.result(timeout=120)
                    # Accumulate parent chunks
                    if parent_chunks:
                        with parent_chunks_lock:
                            all_parent_chunks.extend(parent_chunks)
                    # Push to embed stage (include full_text for late_chunking)
                    full_text = text if enable_late_chunking else None
                    chunk_queue.put((doc_id, chunks, full_text))
                except Exception as exc:
                    logger.warning(f"Chunking failed for {valid_id}: {exc}")
                    first_chunk_error = first_chunk_error or f"{type(exc).__name__}: {exc}"
                    error_doc_ids.add(valid_id)
                    # Push empty to keep doc count accurate for Stage B
                    chunk_queue.put((valid_id, [], None))

                if (i + 1) % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    logger.info(f"  Chunking progress: {i + 1}/{total_docs} ({rate:.0f} docs/min)")

        # Wait for pipeline to drain
        embed_thread.join(timeout=600)
        insert_thread.join(timeout=600)

        # Check for failures
        if embed_thread.is_alive() or insert_thread.is_alive():
            raise RuntimeError(
                "Fast ingest pipeline timed out — threads still alive "
                f"(embed={embed_thread.is_alive()}, insert={insert_thread.is_alive()})"
            )

        if not error_channel.empty():
            stage, exc = error_channel.get()
            raise RuntimeError(f"Fast ingest pipeline failed in {stage} stage: {exc}") from exc

        if len(error_doc_ids) > _MAX_CHUNK_FAILURE_RATE * total_docs:
            # A chunker that fails on most documents leaves a corpus that
            # scores near zero, and the number would read as the method's.
            raise RuntimeError(
                f"Chunking failed for {len(error_doc_ids)} of {total_docs} documents "
                f"(first: {first_chunk_error})"
            )

        # Register the documents themselves.
        #
        # Without this, chunks land in the vector store with no corresponding
        # row in the metadata database, and every retrieved chunk is discarded
        # when the search orchestrator tries to attach its parent document. The
        # symptom is a benchmark that reports ndcg@10 = 0.0 while retrieval,
        # reranking and MMR have all worked correctly -- a plausible, entirely
        # wrong zero, and the hardest kind of failure to notice.
        if ingested_ids := insert_result[1]:
            try:
                from ..core.database import get_rag_database

                rag_db = get_rag_database()
                now = datetime.now(timezone.utc).isoformat()
                registered = 0
                for doc_id in ingested_ids:
                    ok = rag_db.save_document(
                        {
                            "document_id": doc_id,
                            "session_id": session_id,
                            # Benchmark corpora are in-memory strings, never
                            # files on disk; the id is the only honest name.
                            "filename": f"{doc_id}.txt",
                            "file_size": 0,
                            "file_type": "txt",
                            "status": "ready",
                            "chunks_count": 0,
                            "upload_date": now,
                        },
                        file_path="",
                        collection_name=getattr(config, "collection_name", None),
                    )
                    registered += bool(ok)
                logger.info(f"Registered {registered}/{len(ingested_ids)} benchmark documents")
                if registered != len(ingested_ids):
                    logger.error(
                        f"Only {registered} of {len(ingested_ids)} documents registered; "
                        f"searches will silently drop results for the rest"
                    )
            except Exception as e:
                # Loud: this is the difference between real metrics and zeros.
                logger.error(
                    f"Failed to register ingested documents ({e}). Retrieval will "
                    f"return no results for this corpus."
                )

        # Parent chunks for hierarchical retrieval. After the documents are
        # registered: each parent row references its document, and saving them
        # first failed every time on the foreign key, so parent retrieval and
        # auto-merge silently had no parents in any benchmark run.
        if all_parent_chunks:
            try:
                from ..core.database import get_rag_database

                rag_db = get_rag_database()
                parent_count = rag_db.save_parent_chunks(all_parent_chunks)
                logger.info(f"Saved {parent_count} parent chunks for hierarchical retrieval")
            except Exception as e:
                logger.error(f"Failed to save parent chunks: {e}")

        elapsed = time.perf_counter() - t0
        rate = total_docs / elapsed * 60 if elapsed > 0 else 0
        logger.info(
            f"Fast ingest pipeline complete: {insert_result[0]} chunks from "
            f"{total_docs} docs in {elapsed:.1f}s ({rate:.0f} docs/min), "
            f"{len(error_doc_ids)} errors"
        )

        return insert_result[0], insert_result[1], error_doc_ids

    # Run blocking pipeline in a thread to avoid stalling the event loop
    total_chunks, ingested_ids, error_ids = await asyncio.to_thread(_run_pipeline)
    return total_chunks, ingested_ids, error_ids
