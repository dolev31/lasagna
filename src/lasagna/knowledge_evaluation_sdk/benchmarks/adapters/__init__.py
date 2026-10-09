"""Built-in benchmark adapters.

Called by :meth:`BenchmarkRegistry._register_builtins` on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._registry import BenchmarkRegistry


def register_all(registry: "BenchmarkRegistry") -> None:
    """Import and register every built-in benchmark adapter."""
    from .smoke import register as register_smoke
    from .beir import register as register_beir
    from .squad import register as register_squad
    from .natural_questions import register as register_nq
    from .ms_marco import register as register_msmarco
    from .triviaqa import register as register_triviaqa
    from .hotpotqa_multihop import register as register_hotpotqa
    from .musique import register as register_musique
    from .wiki_multihop import register as register_wiki_multihop
    from .multihop_rag import register as register_multihop_rag
    from .bamboogle import register as register_bamboogle
    from .popqa import register as register_popqa
    from .ragbench import register as register_ragbench
    from .kilt import register as register_kilt
    from .fever import register as register_fever

    # First, so it heads `lasagna benchmark list`: it is the only entry that
    # needs no download and the only sane place to start.
    register_smoke(registry)
    register_beir(registry)
    register_squad(registry)
    register_nq(registry)
    register_msmarco(registry)
    register_triviaqa(registry)
    register_hotpotqa(registry)
    register_musique(registry)
    register_wiki_multihop(registry)
    register_multihop_rag(registry)
    register_bamboogle(registry)
    register_popqa(registry)
    register_ragbench(registry)
    register_kilt(registry)
    register_fever(registry)

    # Private evaluation set (always registered; load() will raise a clear
    # error if the dataset hasn't been built yet)
    try:
        from ...private_eval.adapter import register_private_eval_adapters

        register_private_eval_adapters(registry)
    except ImportError:
        pass  # private_eval package not installed
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to register private_eval adapters: %s", exc)
