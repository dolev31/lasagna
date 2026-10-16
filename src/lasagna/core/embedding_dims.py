"""The size of a model's embeddings, read without loading the model.

Setting only ``embedding_model`` (``KnowledgeConfig(embedding_model=
"BAAI/bge-small-en-v1.5")``, ``RAG_EMBEDDING_MODEL=...``) kept the settings
file's ``embedding_dim`` of 1024: the collection was created at 1024, the
384-dimension vectors were refused by the store, the document failed, and
search returned nothing. :class:`~lasagna.core.config.RAGConfig` asks here for
the size of a model set without one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

#: Models whose size is known before they are downloaded: the ones the
#: profiles and the docs name, and the common hosted ones.
KNOWN_DIMS: Dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_read: Dict[str, int] = {}


def known_embedding_dim(model: Optional[str]) -> Optional[int]:
    """The embedding size of ``model``, or None when it cannot be told offline.

    From :data:`KNOWN_DIMS`, else from the model's files: a local folder, or
    the Hugging Face cache (a ``Dense`` layer's ``out_features``, else the
    pooling layer's ``word_embedding_dimension``, else ``hidden_size``).
    Nothing is downloaded.
    """
    if not model:
        return None
    if model in KNOWN_DIMS:
        return KNOWN_DIMS[model]
    if model in _read:
        return _read[model]
    names = [model] if "/" in model else [model, f"sentence-transformers/{model}"]
    for name in names:
        dim = _dim_from_files(lambda filename, name=name: _model_file(name, filename))
        if dim is not None:
            _read[model] = dim  # a size never changes; a miss may, once downloaded
            return dim
    return None


def _model_file(model: str, filename: str) -> Optional[Path]:
    folder = Path(model).expanduser()
    if folder.is_dir():
        path = folder / filename
        return path if path.is_file() else None
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(model, filename)
    except Exception:  # noqa: BLE001 - an unusable cache is "unknown"
        return None
    return Path(hit) if isinstance(hit, str) else None


def _json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, (dict, list)) else None


def _dim_from_files(file) -> Optional[int]:
    modules = _json(file("modules.json"))
    if isinstance(modules, list):
        dense = [m for m in modules if str(m.get("type", "")).endswith("Dense")]
        pooling = [m for m in modules if str(m.get("type", "")).endswith("Pooling")]
        for module, key in (
            (dense[-1:], "out_features"),
            (pooling[:1], "word_embedding_dimension"),
        ):
            if module:
                config = _json(file(f"{module[0].get('path', '')}/config.json"))
                if isinstance(config, dict) and isinstance(config.get(key), int):
                    return config[key]
    config = _json(file("config.json"))
    if isinstance(config, dict) and isinstance(config.get("hidden_size"), int):
        return config["hidden_size"]
    return None
