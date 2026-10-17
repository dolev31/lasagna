"""Built-in rerankers beyond the cross-encoder family: monoT5, LLM grading, FlashRank, Cohere.

Each implements the public ``BaseReranker`` contract and reads the arm's
settings from ``config=`` (the pipeline passes it to built-ins). None changes
the documents it is handed: each returns new ``ScoredDocument`` objects in its
own order, scored by its own model. A missing model, package, key or LLM raises
rather than returning the retrieval order, which would put the control's number
under the reranker's name.
"""

from __future__ import annotations

import copy
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

from lasagna.core.accelerator import device_guard
from lasagna.core.base_components import BaseReranker, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.core.model_downloads import one_weights_copy
from lasagna.core.model_licenses import warn_if_restricted


def _setting(config: Any, key: str) -> Any:
    value = getattr(config, key, None) if config is not None else None
    return value if value is not None else getattr(rag_config, key)


def _ranked(
    documents: Sequence[ScoredDocument], scores: Sequence[float], top_k: int
) -> List[ScoredDocument]:
    """New documents carrying ``scores``, best first; ties keep the input order."""
    order = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), i))
    return [
        ScoredDocument(
            content=documents[i].content,
            score=float(scores[i]),
            document_id=documents[i].document_id,
            chunk_id=documents[i].chunk_id,
            metadata=copy.deepcopy(documents[i].metadata),
        )
        for i in order[:top_k]
    ]


def _chat_model(component: str, model: Optional[str]) -> Any:
    """The configured LLM, or the error that says none is."""
    from lasagna.providers import LLMProviderNotConfigured, get_llm_provider

    try:
        provider = get_llm_provider()
    except LLMProviderNotConfigured as exc:
        raise RuntimeError(f"{component} needs an LLM, and none is configured") from exc
    return provider.get_llm(model_name=model or None, temperature=0)


def _invoke(component: str, llm: Any, prompt: Any, attempts: int = 3) -> str:
    """The response text, retrying a failed call twice before raising."""
    for attempt in range(1, attempts + 1):
        try:
            response = llm.invoke(prompt)
            break
        except Exception as e:
            if attempt == attempts:
                raise RuntimeError(f"{component}: the LLM call failed: {e}") from e
            logger.warning(f"{component} attempt {attempt} failed: {e}")
            time.sleep(2 ** (attempt - 1))
    return str(getattr(response, "content", response))


# ---------------------------------------------------------------------------
# monoT5
# ---------------------------------------------------------------------------


class _MonoT5:
    """A loaded monoT5 checkpoint and the ids of its "true" and "false" tokens."""

    MAX_LENGTH = 512

    def __init__(self, name: str):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("monot5 needs transformers: pip install transformers") from exc
        from .ranker import _detect_device

        warn_if_restricted(name, component="monoT5 reranker")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.device = _detect_device()
        with device_guard(self.device), one_weights_copy():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device).eval()
        # The tokens monoT5 was trained to emit, as pygaggle reads them.
        self.false_id, self.true_id = self.tokenizer.convert_tokens_to_ids(["▁false", "▁true"])
        if self.tokenizer.unk_token_id in (self.false_id, self.true_id):
            raise RuntimeError(f"{name!r} has no 'true'/'false' tokens; it is not a monoT5 model")

    def true_false_logits(self, texts: List[str], batch_size: int) -> np.ndarray:
        """The first decoded token's ``[false, true]`` logits for each input."""
        import torch

        out = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            start_ids = torch.full(
                (enc["input_ids"].shape[0], 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long,
                device=self.device,
            )
            with device_guard(self.device), torch.no_grad():
                logits = self.model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
            out.append(logits[:, [self.false_id, self.true_id]].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))


class MonoT5Reranker(BaseReranker):
    """monoT5 pointwise reranking (Nogueira et al. 2020).

    "Document Ranking with a Pretrained Sequence-to-Sequence Model" (Findings of
    EMNLP 2020). A T5 model fine-tuned to answer ``Query: q Document: d
    Relevant:`` with "true" or "false"; a passage scores the log-probability of
    "true" from a softmax over those two tokens' logits at the first decoding
    step. ``monot5_model`` defaults to castorini/monot5-base-msmarco-10k.
    """

    _forwards_session_config = True

    _models: Dict[str, Any] = {}
    _failed: Dict[str, str] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name or rag_config.monot5_model

    @classmethod
    def _load(cls, name: str) -> Any:
        with cls._lock:
            if name in cls._failed:
                raise RuntimeError(cls._failed[name])
            if name not in cls._models:
                try:
                    cls._models[name] = _MonoT5(name)
                except ImportError:
                    raise
                except Exception as exc:
                    cls._failed[name] = f"Cannot load monoT5 model {name!r}: {exc}"
                    raise RuntimeError(cls._failed[name]) from exc
            return cls._models[name]

    def rerank(
        self, query: str, documents: List[ScoredDocument], top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        if not documents:
            return []
        config = kwargs.get("config")
        name = self._model_name or _setting(config, "monot5_model")
        model = self._load(name)
        texts = [f"Query: {query} Document: {d.content} Relevant:" for d in documents]
        logits = model.true_false_logits(texts, int(_setting(config, "reranker_batch_size")))
        # log softmax over [false, true], keeping "true"
        log_true = logits[:, 1] - np.logaddexp(logits[:, 0], logits[:, 1])
        return _ranked(documents, log_true, top_k)


# ---------------------------------------------------------------------------
# LLM relevance grading
# ---------------------------------------------------------------------------


class LLMPointwiseReranker(BaseReranker):
    """LLM relevance grading, one passage at a time, on a 0-3 scale.

    Thomas et al. 2023, "Large language models can accurately predict searcher
    preferences": the LLM is given the graded scale and asked to consider the
    query's intent, how well the passage matches it (M) and how trustworthy it
    is (T) before a final score (O). The prompt is UMBRELA's adaptation of it to
    passages and the TREC 0-3 scale (Upadhyay et al. 2024). Passages are ordered
    by grade, ties by retrieval order. A response with no readable grade counts
    as 0, as in UMBRELA; if none is readable the call raises. ``llm_reranker_model``
    picks the LLM.
    """

    _forwards_session_config = True

    PROMPT = (
        "Given a query and a passage, you must provide a score on an integer scale of "
        "0 to 3 with the following meanings:\n"
        "0 = represent that the passage has nothing to do with the query,\n"
        "1 = represents that the passage seems related to the query but does not "
        "answer it,\n"
        "2 = represents that the passage has some answer for the query, but the answer "
        "may be a bit unclear, or hidden amongst extraneous information and\n"
        "3 = represents that the passage is dedicated to the query and contains the "
        "exact answer.\n\n"
        "Important Instruction: Assign category 1 if the passage is somewhat related to "
        "the topic but not completely, category 2 if passage presents something very "
        "important related to the entire topic but also has some extra information and "
        "category 3 if the passage only and entirely refers to the topic. If none of "
        "the above satisfies give it category 0.\n\n"
        "Query: {query}\n"
        "Passage: {passage}\n\n"
        "Split this problem into steps:\n"
        "Consider the underlying intent of the search.\n"
        "Measure how well the content matches a likely intent of the query (M).\n"
        "Measure how trustworthy the passage is (T).\n"
        "Consider the aspects above and the relative importance of each, and decide on "
        "a final score (O). Final score must be an integer value only.\n"
        "Do not provide any code in result. Provide each score in the format of: "
        "##final score: score without providing any reasoning."
    )
    #: Concurrent grading calls per query.
    MAX_PARALLEL = 8

    def __init__(self, model: Optional[str] = None, **kwargs):
        self._model = model

    @property
    def model_name(self) -> str:
        return f"llm_pointwise:{self._model or rag_config.llm_reranker_model}"

    @staticmethod
    def parse_grade(text: str) -> Optional[int]:
        """The 0-3 grade in a response, or None when there is none to read."""
        text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
        match = re.search(
            r"final score\s*:?\s*\**\s*(\d+)", text, flags=re.IGNORECASE
        ) or re.search(r"\bO\W{0,3}[:=]\s*(\d+)", text)  # Thomas et al.'s "O": overall
        if match:
            grade = int(match.group(1))
            return grade if 0 <= grade <= 3 else None
        numbers = re.findall(r"\b\d+\b", text)
        if len(numbers) == 1 and 0 <= int(numbers[0]) <= 3:
            return int(numbers[0])
        return None

    def rerank(
        self, query: str, documents: List[ScoredDocument], top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        if not documents:
            return []
        config = kwargs.get("config")
        model = self._model or _setting(config, "llm_reranker_model")
        llm = _chat_model("llm_pointwise", model)
        prompts = [self.PROMPT.format(query=query, passage=d.content) for d in documents]
        with ThreadPoolExecutor(max_workers=min(self.MAX_PARALLEL, len(prompts))) as pool:
            responses = list(pool.map(lambda p: _invoke("llm_pointwise", llm, p), prompts))
        grades = [self.parse_grade(r) for r in responses]
        if all(g is None for g in grades):
            raise RuntimeError(
                f"llm_pointwise: no response from {model!r} contained a 0-3 grade "
                f"(first: {responses[0][:120]!r})"
            )
        return _ranked(documents, [float(g or 0) for g in grades], top_k)


# ---------------------------------------------------------------------------
# Optional-dependency rerankers
# ---------------------------------------------------------------------------


class FlashRankReranker(BaseReranker):
    """FlashRank: small cross-encoders run with ONNX Runtime on the CPU.

    Needs ``pip install flashrank``. ``flashrank_model`` names one of FlashRank's
    models (default ms-marco-TinyBERT-L-2-v2, about 4 MB); they download to
    ``~/.cache/flashrank`` on first use. A listwise FlashRank model returns an
    order without scores, which is kept as the order.
    """

    _forwards_session_config = True

    _rankers: Dict[str, Any] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return f"flashrank:{self._model_name or rag_config.flashrank_model}"

    @classmethod
    def _ranker(cls, name: str) -> Any:
        try:
            from flashrank import Ranker
        except ImportError as exc:
            raise ImportError(
                "The flashrank reranker needs FlashRank: pip install flashrank"
            ) from exc
        with cls._lock:
            if name not in cls._rankers:
                cache = os.path.join(os.path.expanduser("~"), ".cache", "flashrank")
                try:
                    cls._rankers[name] = Ranker(model_name=name, cache_dir=cache)
                except Exception as exc:
                    raise RuntimeError(f"Cannot load FlashRank model {name!r}: {exc}") from exc
            return cls._rankers[name]

    def rerank(
        self, query: str, documents: List[ScoredDocument], top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        if not documents:
            return []
        name = self._model_name or _setting(kwargs.get("config"), "flashrank_model")
        ranker = self._ranker(name)
        from flashrank import RerankRequest

        passages = [{"id": i, "text": d.content} for i, d in enumerate(documents)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        n = len(results)
        scores = [float("-inf")] * len(documents)
        for rank, result in enumerate(results):
            score = result.get("score")
            scores[int(result["id"])] = float(score) if score is not None else float(n - rank)
        return _ranked(documents, scores, top_k)


class CohereReranker(BaseReranker):
    """Cohere's hosted Rerank model, through its API.

    Needs ``pip install cohere`` and an API key in ``COHERE_API_KEY`` (or
    ``CO_API_KEY``). ``cohere_rerank_model`` defaults to rerank-v3.5. The
    model is proprietary and versioned by Cohere, so pin the dated name when a
    result has to be reproduced.
    """

    _forwards_session_config = True

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name
        self._client = None

    @property
    def model_name(self) -> str:
        return f"cohere:{self._model_name or rag_config.cohere_rerank_model}"

    def _get_client(self) -> Any:
        try:
            import cohere
        except ImportError as exc:
            raise ImportError("The cohere reranker needs Cohere's SDK: pip install cohere") from exc
        key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not key:
            raise RuntimeError("The cohere reranker needs COHERE_API_KEY set in the environment")
        if self._client is None:
            client_cls = getattr(cohere, "ClientV2", None) or cohere.Client
            self._client = client_cls(api_key=key)
        return self._client

    def rerank(
        self, query: str, documents: List[ScoredDocument], top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        if not documents:
            return []
        client = self._get_client()
        name = self._model_name or _setting(kwargs.get("config"), "cohere_rerank_model")
        try:
            response = client.rerank(
                model=name,
                query=query,
                documents=[d.content for d in documents],
                top_n=len(documents),
            )
        except Exception as exc:
            raise RuntimeError(f"cohere: the Rerank call failed: {exc}") from exc
        scores = [float("-inf")] * len(documents)
        for result in response.results:
            scores[int(result.index)] = float(result.relevance_score)
        return _ranked(documents, scores, top_k)
