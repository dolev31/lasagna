# How the paper used Lasagna

Lasagna is the framework behind the paper *Retrieval Augmented Generation
Compositionality* by Ido Levy, Segev Shlomov and Leshem Choshen (2026). This page
says what the study asked and how the framework made it possible. For the full
results, read the [paper](https://arxiv.org/abs/2610.29722).

## The question

RAG papers report one component at a time. A new reranker is measured against a
baseline with no query expansion and an old chunker. If a reranker claims +4 NDCG
and a query expander claims +3, you cannot tell from the papers whether stacking
them gives +7, +4 or -1.

The study asks whether gains measured one component at a time survive when the
components are combined.

## The design

The study tests 450 configurations across eight benchmarks, with corpora from 5K
to 200K documents. Eight design dimensions vary across the configurations.

To attribute a difference to a single change, every other part of the pipeline has
to stay fixed. That needs a pipeline with swappable stages and a harness that
checks the swap. Lasagna provides both: the [stages](stage-isolation.md), the
[comparability key](comparability.md) and the
[significance tests](significance.md).

## What it found

| Finding | Predicted | Realized |
|---|---|---|
| Retrieval gains do not add up: the three strongest single techniques, combined (NDCG@10) | +47% | +17% |
| Faithfulness gains do carry over: one factor at a time versus the best full pipeline | +28% | +32% |

The best pipeline also changes with corpus size. At 200K documents, the design
that leads at small corpora trails by 13.7% NDCG@10, and the rank flip is visible
from 25K.

## What the paper calls the profiles

The paper's `Max-Reliable` profile is `max_quality` in the code. See
[profiles and configuration](../configuration.md).

## Run your own version

You can ask the same question about your own components. Run the arms alone and
together, then use `lasagna results interaction` on the 2x2. The
[compare a component](../guides/compare-a-component.md) guide shows how to write
the sweep.

Lasagna does not ship the paper's per-query results or a prebuilt index. The
[pre-registration](https://github.com/dolev31/Lasagna-Retrieval-Augmented-Generation-Compositionality)
has the study's plan.
