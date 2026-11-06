"""LLM-based entity and relationship extraction from chunk text.

Synchronous implementation (process_document is sync in _document_pipeline.py).
Uses ThreadPoolExecutor for batch parallelism.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, NamedTuple, Optional

from loguru import logger


_EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Entity types to extract: {entity_types}

Return a JSON object with:
- "entities": list of {{"name": str, "type": str}}
- "relationships": list of {{"source": str, "target": str, "type": str}}

Only extract clearly stated entities and relationships.
Return ONLY valid JSON, no explanation.

Text:
{text}

JSON:"""


class EntityExtractor:
    """Synchronous LLM-based entity and relationship extractor."""

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ):
        self._entity_types = entity_types or [
            "PERSON",
            "ORG",
            "CONCEPT",
            "LOCATION",
            "TECHNOLOGY",
            "EVENT",
        ]
        self._model_name = model_name
        #: Chunks whose call failed or whose reply held no readable JSON.
        self.failures = 0
        self._failures_lock = threading.Lock()

    def _failed(self) -> None:
        with self._failures_lock:
            self.failures += 1

    def extract_from_chunk(self, chunk_text: str, llm: Any) -> Dict[str, List[Dict[str, str]]]:
        """Extract entities and relationships from a single chunk.

        Args:
            chunk_text: Text content of the chunk.
            llm: LangChain-compatible LLM instance.

        Returns:
            {"entities": [...], "relationships": [...]}
        """
        if not chunk_text or len(chunk_text) < 50:
            return {"entities": [], "relationships": []}

        prompt = _EXTRACTION_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            text=chunk_text[:3000],  # Limit input length
        )

        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                result = json.loads(json_match.group())
                entities = result.get("entities", [])
                relationships = result.get("relationships", [])
                # Validate structure
                valid_entities = [
                    e for e in entities if isinstance(e, dict) and "name" in e and "type" in e
                ]
                valid_rels = [
                    r
                    for r in relationships
                    if isinstance(r, dict) and "source" in r and "target" in r and "type" in r
                ]
                return {"entities": valid_entities, "relationships": valid_rels}
            logger.debug("Entity extraction failed: the reply holds no JSON object")
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Entity extraction failed: {e}")
        self._failed()
        return {"entities": [], "relationships": []}

    def extract_batch_sync(
        self,
        chunks: List[str],
        llm: Any,
        max_workers: int = 4,
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        """Batch extraction using ThreadPoolExecutor (synchronous).

        Args:
            chunks: List of chunk texts.
            llm: LangChain-compatible LLM instance.
            max_workers: Number of parallel threads.

        Returns:
            List of extraction results, one per chunk.
        """
        if not chunks:
            return []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(
                pool.map(
                    lambda c: self.extract_from_chunk(c, llm),
                    chunks,
                )
            )
        return results


class GraphExtraction(NamedTuple):
    """What `extract_graph` found: one extraction per chunk, and how many chunks failed."""

    extractions: list
    failed: int
    model: str

    @property
    def complete(self) -> bool:
        """Every chunk was extracted (a chunk too short to extract counts as done)."""
        return self.failed == 0


def extract_graph(chunk_texts, config) -> GraphExtraction:
    """Entities and relationships of each chunk, with the extraction model ``config`` names.

    The one place the upload pipeline, the benchmark ingest and a rebuild of a
    reused corpus's graph extract: ``graph_extraction_model``, or the default
    model, at temperature 0. Raises when no LLM can be built; a chunk whose
    call fails is counted in ``failed``, so the document is not recorded as
    extracted and is extracted again later.
    """
    from ...providers.llm_provider import SimpleLLMProvider
    from ...utils.model_utils import get_default_model_name

    model = getattr(config, "graph_extraction_model", "") or get_default_model_name()
    extractor = EntityExtractor(
        entity_types=getattr(config, "graph_entity_types", None), model_name=model
    )
    llm = SimpleLLMProvider().get_llm(model_name=model, temperature=0)
    extractions = extractor.extract_batch_sync(list(chunk_texts), llm, max_workers=4)
    return GraphExtraction(extractions, extractor.failures, model)
