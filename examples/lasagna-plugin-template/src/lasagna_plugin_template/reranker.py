"""A reranker, complete enough to copy.

Subclassing ``BaseReranker`` is the entire contract. There is no registration
call here: the entry point in ``pyproject.toml`` is what makes Lasagna find it,
so the component stays importable and testable on its own.
"""

from __future__ import annotations

import copy
from typing import Any, List

from lasagna.core.base_components import BaseReranker, ScoredDocument


class LengthReranker(BaseReranker):
    """Ranks by word count. Deliberately not a good idea.

    A real component would score query-document relevance. This one is
    obviously wrong on purpose: when you compare it against a control the
    ordering visibly changes, which proves the swap reached execution. A
    plausible component would leave you unable to tell "it ran and agreed"
    from "it never ran".
    """

    def __init__(self, longest_first: bool = True) -> None:
        # Constructible with no required arguments: the pipeline builds
        # components from a config name and cannot supply constructor values.
        self.longest_first = longest_first

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[ScoredDocument]:
        # Deep-copy before touching anything. The pipeline may hand the same
        # objects to more than one component; mutating them makes results
        # depend on the order stages ran in, which is exactly the hidden
        # coupling that makes a measured difference unattributable.
        out = [copy.deepcopy(d) for d in documents]
        for doc in out:
            doc.score = float(len(doc.content.split()))
        out.sort(key=lambda d: d.score, reverse=self.longest_first)
        # Honour top_k: the caller decides how many results it wants.
        return out[:top_k]
