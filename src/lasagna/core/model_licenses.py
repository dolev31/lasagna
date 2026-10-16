"""License metadata for the model weights Lasagna can download.

Lasagna ships no weights.  Embedding, reranking and sparse-encoding models are
pulled from their upstream host (usually the Hugging Face Hub) the first time
they are used, and each carries its own license -- several of which are more
restrictive than Lasagna's own Apache 2.0.

That matters most in the direction people least expect: Lasagna is research
tooling, so comparing a non-commercial reranker against a permissive one is a
perfectly ordinary thing to do here.  The problem is only when a configuration
tuned in a study is later carried into a product.  So nothing is blocked, but
selecting a restricted model is announced loudly and exactly once.

``COMMERCIAL_USE_RESTRICTED`` models are usable for research and evaluation
under their license terms, but **not** in a commercial deployment without a
separate grant from the model's authors.

Licenses change and this table is a convenience, not legal advice.  Confirm
against the upstream model card before you rely on it -- ``LicenseInfo.url``
links to each one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger
from lasagna.core.provenance import redact_url

__all__ = [
    "LicenseInfo",
    "MODEL_LICENSES",
    "describe",
    "is_commercial_use_restricted",
    "warn_if_restricted",
    "restricted_models",
]


@dataclass(frozen=True)
class LicenseInfo:
    """License terms for a single set of model weights."""

    spdx: str
    """SPDX identifier, or a short label where no SPDX id exists."""

    commercial_use: bool
    """Whether the license permits use in a commercial product."""

    url: str
    """Upstream model card, which is the authoritative source."""

    note: str = ""
    """Anything a user needs to know beyond the identifier."""

    @property
    def restricted(self) -> bool:
        return not self.commercial_use


_PERMISSIVE_MIT = "MIT"
_PERMISSIVE_APACHE = "Apache-2.0"


def _hf(model: str) -> str:
    return f"https://huggingface.co/{model}"


#: Model id -> license terms.  Keys match the identifiers used in
#: ``settings.lasagna.toml`` and the configurator schema.
MODEL_LICENSES: Dict[str, LicenseInfo] = {
    # ---- Embedding models -------------------------------------------------
    "BAAI/bge-m3": LicenseInfo(_PERMISSIVE_MIT, True, _hf("BAAI/bge-m3")),
    "BAAI/bge-large-en-v1.5": LicenseInfo(_PERMISSIVE_MIT, True, _hf("BAAI/bge-large-en-v1.5")),
    "BAAI/bge-base-en-v1.5": LicenseInfo(_PERMISSIVE_MIT, True, _hf("BAAI/bge-base-en-v1.5")),
    "BAAI/bge-small-en-v1.5": LicenseInfo(_PERMISSIVE_MIT, True, _hf("BAAI/bge-small-en-v1.5")),
    "intfloat/e5-large-v2": LicenseInfo(_PERMISSIVE_MIT, True, _hf("intfloat/e5-large-v2")),
    "intfloat/e5-base-v2": LicenseInfo(_PERMISSIVE_MIT, True, _hf("intfloat/e5-base-v2")),
    "intfloat/e5-small-v2": LicenseInfo(_PERMISSIVE_MIT, True, _hf("intfloat/e5-small-v2")),
    "intfloat/multilingual-e5-large": LicenseInfo(
        _PERMISSIVE_MIT, True, _hf("intfloat/multilingual-e5-large")
    ),
    "sentence-transformers/all-MiniLM-L6-v2": LicenseInfo(
        _PERMISSIVE_APACHE, True, _hf("sentence-transformers/all-MiniLM-L6-v2")
    ),
    # ---- Rerankers --------------------------------------------------------
    "cross-encoder/ms-marco-MiniLM-L-6-v2": LicenseInfo(
        _PERMISSIVE_APACHE, True, _hf("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ),
    "cross-encoder/ms-marco-MiniLM-L-12-v2": LicenseInfo(
        _PERMISSIVE_APACHE, True, _hf("cross-encoder/ms-marco-MiniLM-L-12-v2")
    ),
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": LicenseInfo(
        _PERMISSIVE_APACHE, True, _hf("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    ),
    "cross-encoder/ms-marco-electra-base": LicenseInfo(
        _PERMISSIVE_APACHE, True, _hf("cross-encoder/ms-marco-electra-base")
    ),
    "BAAI/bge-reranker-base": LicenseInfo(_PERMISSIVE_MIT, True, _hf("BAAI/bge-reranker-base")),
    "BAAI/bge-reranker-large": LicenseInfo(_PERMISSIVE_MIT, True, _hf("BAAI/bge-reranker-large")),
    "BAAI/bge-reranker-v2-m3": LicenseInfo(
        _PERMISSIVE_APACHE, True, _hf("BAAI/bge-reranker-v2-m3")
    ),
    "mixedbread-ai/mxbai-rerank-large-v1": LicenseInfo(
        _PERMISSIVE_APACHE, True, _hf("mixedbread-ai/mxbai-rerank-large-v1")
    ),
    "colbert-ir/colbertv2.0": LicenseInfo(_PERMISSIVE_MIT, True, _hf("colbert-ir/colbertv2.0")),
    # ---- Restricted: research and evaluation only -------------------------
    "naver/splade-cocondenser-ensembledistil": LicenseInfo(
        "CC-BY-NC-SA-4.0",
        False,
        _hf("naver/splade-cocondenser-ensembledistil"),
        note=(
            "Non-commercial, share-alike. This is Lasagna's default SPLADE "
            "sparse encoder; sparse retrieval is opt-in, so it loads only when "
            "you enable it."
        ),
    ),
    "jinaai/jina-reranker-v2-base-multilingual": LicenseInfo(
        "CC-BY-NC-4.0",
        False,
        _hf("jinaai/jina-reranker-v2-base-multilingual"),
        note="Non-commercial. A commercial license is available from Jina AI.",
    ),
    "BAAI/bge-reranker-v2-gemma": LicenseInfo(
        "Gemma-Terms-of-Use",
        False,
        _hf("BAAI/bge-reranker-v2-gemma"),
        note=(
            "Built on Gemma and inherits the Gemma Terms of Use, which impose "
            "use restrictions and are not an OSI-approved open source license."
        ),
    ),
}

# Models already announced, so a warning is emitted once per process rather
# than on every batch.
_ANNOUNCED: set = set()


def describe(model_name: str) -> Optional[LicenseInfo]:
    """Return license terms for ``model_name``, or None if unknown.

    Unknown is not the same as unrestricted: a model absent from this table
    simply has not been catalogued, and its model card is authoritative.
    """
    if not model_name:
        return None
    info = MODEL_LICENSES.get(model_name)
    if info is not None:
        return info
    # Identifiers are occasionally written with different casing.
    lowered = model_name.lower()
    for key, value in MODEL_LICENSES.items():
        if key.lower() == lowered:
            return value
    return None


def is_commercial_use_restricted(model_name: str) -> bool:
    """True only when the model is known to forbid commercial use."""
    info = describe(model_name)
    return info is not None and info.restricted


def restricted_models() -> Dict[str, LicenseInfo]:
    """Every catalogued model that forbids commercial use."""
    return {name: info for name, info in MODEL_LICENSES.items() if info.restricted}


def warn_if_restricted(model_name: str, *, component: str = "model") -> None:
    """Announce ``model_name``'s license the first time it is loaded.

    Restricted models log at WARNING, permissive ones at DEBUG, and models
    missing from the table log at DEBUG asking the user to check upstream.
    Never raises and never blocks: the caller's job is retrieval, not
    licence enforcement.

    Args:
        model_name: Hugging Face identifier being loaded.
        component: Where it is being used, e.g. "reranker", for the message.
    """
    try:
        if model_name in _ANNOUNCED:
            return
        _ANNOUNCED.add(model_name)

        info = describe(model_name)
        if info is None:
            logger.debug(
                f"{component} '{model_name}' is not in Lasagna's license table; "
                f"check its model card for licensing terms."
            )
            return

        if info.restricted:
            logger.warning(
                f"LICENSE: {component} '{model_name}' is licensed {info.spdx} -- "
                f"research and evaluation only, NOT for commercial use. "
                f"{info.note} See {redact_url(info.url)}"
            )
        else:
            logger.debug(f"{component} '{model_name}' licensed {info.spdx}")
    except Exception:  # pragma: no cover - advisory only, must never break search
        pass
