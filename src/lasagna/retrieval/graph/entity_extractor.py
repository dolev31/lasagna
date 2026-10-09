"""LLM-based entity and relationship extraction from chunk text.

Synchronous implementation (process_document is sync in _document_pipeline.py).
Uses ThreadPoolExecutor for batch parallelism.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

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
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Entity extraction failed: {e}")

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
