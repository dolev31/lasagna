"""``smoke`` — a tiny, self-contained benchmark that needs no network.

Every other benchmark downloads from HuggingFace: hundreds of megabytes before
the first number, and nothing at all behind a firewall or in CI. That has three
consequences this adapter exists to remove.

1. **There was no quickstart that worked.** The documented one paired a
   retrieval-only BEIR set with a mode that cannot score retrieval, so the first
   command a new user ran always failed.
2. **Nothing exercised the live path hermetically.** Live mode — ingest, embed,
   retrieve, rerank, score — is where the interesting failures are, and no test
   could reach it without a multi-gigabyte download.
3. **Defaulting to live mode would have been hostile** without something small
   to default *to*.

The corpus is 24 short documents across four clearly separated domains, with 12
queries and hand-written relevance judgements. It is deliberately easy: a
working pipeline should score near the top. That makes it a smoke test, not a
measurement — it answers "is this wired up correctly", never "is this good".
Reporting a `smoke` score as evidence of retrieval quality would be a category
error, and :meth:`info` says so in the description.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .._base import (
    BaseBenchmarkAdapter,
    BenchmarkCategory,
    BenchmarkEvalMode,
    BenchmarkInfo,
    BenchmarkSample,
)

if TYPE_CHECKING:
    from .._registry import BenchmarkRegistry


# Four domains with disjoint vocabulary, so retrieval failures are unambiguous:
# a wrong answer means the pipeline reached a different topic entirely, not that
# two near-synonyms were confused.
_CORPUS: Dict[str, str] = {
    # -- tardigrades ------------------------------------------------------
    "tardigrade_1": (
        "Tardigrades are microscopic animals often called water bears. They "
        "survive desiccation by entering a state called cryptobiosis, in which "
        "metabolism slows to under one percent of normal."
    ),
    "tardigrade_2": (
        "In cryptobiosis a tardigrade replaces lost water with the sugar "
        "trehalose, which vitrifies into a glassy matrix and holds cell "
        "membranes in place."
    ),
    "tardigrade_3": (
        "Tardigrades have survived direct exposure to the vacuum of space and "
        "to solar ultraviolet radiation during the FOTON-M3 mission in 2007."
    ),
    "tardigrade_4": (
        "A tardigrade has eight legs, each ending in claws. Adults are "
        "typically half a millimetre long."
    ),
    "tardigrade_5": (
        "Damage repair in tardigrades is aided by Dsup, a protein that binds "
        "DNA and shields it from hydroxyl radicals."
    ),
    "tardigrade_6": (
        "Tardigrades are found in moss and lichen, in deep sea sediment, and "
        "on mountain tops above six thousand metres."
    ),
    # -- glassmaking ------------------------------------------------------
    "glass_1": (
        "Soda-lime glass is made by melting silica sand with sodium carbonate "
        "and calcium oxide. The soda lowers the melting point of the silica "
        "from about 1700 to roughly 1000 degrees Celsius."
    ),
    "glass_2": (
        "Adding calcium oxide, known as lime, makes soda-lime glass insoluble "
        "in water. Without it the product is water glass, which dissolves."
    ),
    "glass_3": (
        "Borosilicate glass substitutes boron trioxide for some of the soda, "
        "which lowers thermal expansion and makes the glass resistant to "
        "sudden temperature change."
    ),
    "glass_4": (
        "The float process forms flat glass by floating molten glass on a bath "
        "of molten tin, producing surfaces that need no polishing."
    ),
    "glass_5": (
        "Annealing relieves internal stress by holding glass just below its "
        "softening point and then cooling it slowly."
    ),
    "glass_6": (
        "Lead crystal replaces calcium oxide with lead oxide, raising the "
        "refractive index and giving the characteristic sparkle."
    ),
    # -- Byzantine coinage ------------------------------------------------
    "coin_1": (
        "The solidus was a gold coin introduced by Constantine the Great in "
        "roughly 312 AD. It weighed about 4.5 grams and held its purity for "
        "seven centuries."
    ),
    "coin_2": (
        "The follis was the standard Byzantine copper coin. Its value against "
        "the gold solidus drifted repeatedly during the sixth century."
    ),
    "coin_3": (
        "Byzantine emperors marked coins with a mint mark such as CON for "
        "Constantinople or THES for Thessalonica."
    ),
    "coin_4": (
        "Debasement of the solidus began in earnest under Nikephoros II and "
        "accelerated through the eleventh century."
    ),
    "coin_5": (
        "The hyperpyron replaced the debased solidus after the monetary reform "
        "of Alexios I Komnenos in 1092."
    ),
    "coin_6": (
        "Anonymous folles, struck without an emperor's portrait, carried an "
        "image of Christ Pantokrator instead."
    ),
    # -- lighthouse engineering -------------------------------------------
    "light_1": (
        "The Fresnel lens replaced heavy mirrors in lighthouses. Concentric "
        "rings of prisms bend light into a single beam using a fraction of the "
        "glass a conventional lens would need."
    ),
    "light_2": (
        "Lighthouse characteristics are defined by the pattern of flashes. "
        "Each station has a distinct period so mariners can identify it."
    ),
    "light_3": (
        "The Eddystone lighthouse was rebuilt four times. Smeaton's tower of "
        "1759 used hydraulic lime that cured underwater."
    ),
    "light_4": (
        "Rotation of early lighthouse optics was driven by a falling weight on "
        "a chain, wound by the keeper every few hours."
    ),
    "light_5": (
        "A first-order Fresnel lens stands over two and a half metres tall and "
        "was the largest size in routine service."
    ),
    "light_6": (
        "Automation removed keepers from most lighthouses during the twentieth "
        "century, the last British station being automated in 1998."
    ),
}


# `expected_docs` maps doc id -> graded relevance. 2 = directly answers,
# 1 = same topic and useful context. Grading rather than binary because
# NDCG uses graded gain, and a smoke test should exercise that path.
_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "smoke_q1",
        "question": "How do tardigrades survive drying out?",
        "ground_truth": "They enter cryptobiosis, replacing lost water with trehalose.",
        "expected_docs": {"tardigrade_1": 2, "tardigrade_2": 2, "tardigrade_5": 1},
        "domain": "tardigrades",
    },
    {
        "query_id": "smoke_q2",
        "question": "What protein protects tardigrade DNA from radiation damage?",
        "ground_truth": "Dsup, which binds DNA and shields it from hydroxyl radicals.",
        "expected_docs": {"tardigrade_5": 2, "tardigrade_3": 1},
        "domain": "tardigrades",
    },
    {
        "query_id": "smoke_q3",
        "question": "How large is an adult tardigrade and how many legs does it have?",
        "ground_truth": "About half a millimetre long, with eight legs.",
        "expected_docs": {"tardigrade_4": 2},
        "domain": "tardigrades",
    },
    {
        "query_id": "smoke_q4",
        "question": "Why is sodium carbonate added when making glass?",
        "ground_truth": "It lowers the melting point of silica from about 1700 to about 1000 C.",
        "expected_docs": {"glass_1": 2, "glass_2": 1},
        "domain": "glassmaking",
    },
    {
        "query_id": "smoke_q5",
        "question": "What makes borosilicate glass resistant to thermal shock?",
        "ground_truth": "Boron trioxide replaces some soda, lowering thermal expansion.",
        "expected_docs": {"glass_3": 2},
        "domain": "glassmaking",
    },
    {
        "query_id": "smoke_q6",
        "question": "How is flat window glass manufactured without polishing?",
        "ground_truth": "By floating molten glass on a bath of molten tin.",
        "expected_docs": {"glass_4": 2, "glass_5": 1},
        "domain": "glassmaking",
    },
    {
        "query_id": "smoke_q7",
        "question": "Who introduced the gold solidus and when?",
        "ground_truth": "Constantine the Great, around 312 AD.",
        "expected_docs": {"coin_1": 2, "coin_4": 1},
        "domain": "byzantine_coinage",
    },
    {
        "query_id": "smoke_q8",
        "question": "Which coin replaced the debased solidus, and under which emperor?",
        "ground_truth": "The hyperpyron, under Alexios I Komnenos in 1092.",
        "expected_docs": {"coin_5": 2, "coin_4": 1},
        "domain": "byzantine_coinage",
    },
    {
        "query_id": "smoke_q9",
        "question": "What does the mint mark CON signify on a Byzantine coin?",
        "ground_truth": "It identifies the mint at Constantinople.",
        "expected_docs": {"coin_3": 2},
        "domain": "byzantine_coinage",
    },
    {
        "query_id": "smoke_q10",
        "question": "Why did lighthouses adopt Fresnel lenses instead of mirrors?",
        "ground_truth": ("Concentric prism rings bend light into one beam using far less glass."),
        "expected_docs": {"light_1": 2, "light_5": 1},
        "domain": "lighthouses",
    },
    {
        "query_id": "smoke_q11",
        "question": "How did keepers rotate early lighthouse optics?",
        "ground_truth": "A falling weight on a chain, wound by hand every few hours.",
        "expected_docs": {"light_4": 2},
        "domain": "lighthouses",
    },
    {
        "query_id": "smoke_q12",
        "question": "What did Smeaton use to build the Eddystone lighthouse in 1759?",
        "ground_truth": "Hydraulic lime, which cured underwater.",
        "expected_docs": {"light_3": 2},
        "domain": "lighthouses",
    },
]


class SmokeBenchmarkAdapter(BaseBenchmarkAdapter):
    """A 24-document, 12-query benchmark held entirely in this module."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="smoke",
            description=(
                "Self-contained 24-document smoke test. No download. Verifies a "
                "pipeline is wired up correctly and runs end to end; it is far too "
                "small and too easy to measure retrieval quality, and a score here "
                "should never be reported as a result."
            ),
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.END_TO_END,
            num_queries=len(_QUERIES),
            num_documents=len(_CORPUS),
            paper=None,
            license="Apache-2.0",  # written for this repo, same terms as the code
            tags=["smoke", "offline", "hermetic", "no-download"],
        )

    def available_splits(self) -> List[str]:
        # One split. Anything else must fail loudly rather than quietly serving
        # this one under another name.
        return ["test"]

    def load(
        self,
        split: str = "test",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        if split != "test":
            raise ValueError(
                f"benchmark 'smoke' has only a 'test' split, got {split!r}. "
                f"It is generated in-process, so there is nothing else to serve."
            )

        rows = list(_QUERIES)

        if filters and "domain" in filters:
            wanted = filters["domain"]
            wanted = {wanted} if isinstance(wanted, str) else set(wanted)
            rows = [r for r in rows if r["domain"] in wanted]

        if n_samples is not None and n_samples < len(rows):
            # Same seeded-sample contract as every other adapter, so
            # `sampling_hash` and the provenance digest behave identically here.
            rows = random.Random(seed).sample(rows, n_samples)

        return [
            BenchmarkSample(
                query_id=r["query_id"],
                question=r["question"],
                ground_truth=r["ground_truth"],
                ground_truth_answers=[r["ground_truth"]],
                expected_docs=dict(r["expected_docs"]),
                # Gold context, so the generation-only path can score this
                # benchmark too -- it is the one benchmark that works in every
                # mode, which is what makes it usable as a quickstart.
                context=" ".join(_CORPUS[d] for d in sorted(r["expected_docs"])),
                domain=r["domain"],
                hops=1,
                metadata={"synthetic": True},
            )
            for r in rows
        ]

    def available_filters(self) -> Dict[str, List[str]]:
        return {"domain": sorted({r["domain"] for r in _QUERIES})}

    def load_corpus(
        self,
        cache_dir: Optional[str] = None,
        max_docs: int = 100_000,
    ) -> Optional[Dict[str, str]]:
        """The corpus for live mode.

        Most adapters return ``None`` here, which is why live mode is
        unavailable for them. This one is the reason a live run can be exercised
        without a network.

        ``max_docs`` was accepted and ignored. Harmless at 24 documents, but a
        parameter that silently does nothing is how the harness came to believe
        every adapter honoured the cap.
        """
        if max_docs and max_docs < len(_CORPUS):
            return dict(list(_CORPUS.items())[:max_docs])
        return dict(_CORPUS)


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("smoke", SmokeBenchmarkAdapter())
