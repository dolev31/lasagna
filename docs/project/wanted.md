# Wanted

What Lasagna would like to have next, and what "done" means for each. Every item
has an open issue; comment there to claim it.

A method in Lasagna is measured against every other component of its stage, in a
complete pipeline, on the same benchmarks, with everything else held fixed. If
your paper's method is here, anyone can check it against theirs with one sweep,
and the catalog lists it beside its paper.

| If you want to | Start with |
|---|---|
| Add your paper's method | [Components](#components) |
| Make a first pull request | [Good first issues](#good-first-issues) |
| Add a dataset | [Benchmarks](#benchmarks) |
| Run a study | [Research questions](#research-questions) |
| Run Lasagna in production | [Deployment](#deployment) |

---

## Components

Written for authors of RAG methods. The path is six commands, and
[contributing components](../contributing-components.md) walks through each:
write the component, `lasagna component validate`, a smoke run, a sweep against a
built-in control, `lasagna results significance`, then `lasagna results bundle`.
The worked example there is a toy reranker against `cross_encoder` on 100 SciFact
queries: `cross_encoder` wins by +0.0913 ndcg@10 (d_z +0.38, p (Holm) 1.0e-04).

| Stage | Method | Paper | Note |
|---|---|---|---|
| document_processor | Unstructured | [repository](https://github.com/Unstructured-IO/unstructured) | No paper. Layout-aware parsing for mixed files. |
| embedder | jina-embeddings-v2, with late chunking | [2409.04701](https://arxiv.org/abs/2409.04701) | Late chunking exists as a setting; this is the Jina embedder behind it. |
| retriever | RAPTOR | [2401.18059](https://arxiv.org/abs/2401.18059) | The tree code exists behind `enable_raptor`; it needs registering as a retriever. |
| retriever | GraphRAG, LightRAG, HippoRAG, G-Retriever | [2404.16130](https://arxiv.org/abs/2404.16130), [2410.05779](https://arxiv.org/abs/2410.05779), [2405.14831](https://arxiv.org/abs/2405.14831), [2402.07630](https://arxiv.org/abs/2402.07630) | A small graph retriever exists behind `enable_graph_retrieval`; register it first. |
| retriever | ColBERTv2 with PLAID | [2112.01488](https://arxiv.org/abs/2112.01488) | Late interaction as a retriever, with its own index. |
| retriever | ColPali | [2407.01449](https://arxiv.org/abs/2407.01449) | Page images as the documents. |
| reranker | RankZephyr, RankVicuna | [2312.02724](https://arxiv.org/abs/2312.02724), [2309.15088](https://arxiv.org/abs/2309.15088) | Open-weight listwise rerankers, beside the built-in `llm_listwise`. |
| reranker | Setwise | [2310.09497](https://arxiv.org/abs/2310.09497) | Fewer LLM calls than listwise. |
| agentic_pipeline | Search-o1, Search-R1 | [2501.05366](https://arxiv.org/abs/2501.05366), [2503.09516](https://arxiv.org/abs/2503.09516) | A model that reasons and issues its own searches. |
| none yet | RECOMP, LongLLMLingua | [2310.04408](https://arxiv.org/abs/2310.04408), [2310.06839](https://arxiv.org/abs/2310.06839) | Context compression has no stage. Post a short design first: which stage, and what it returns. |

Already in Lasagna, so not on this list: BGE-M3, the ColBERT reranker,
`llm_listwise`, `self_ask`, `ircot`, MarkItDown, PyMuPDF4LLM, OpenTelemetry
tracing, and local models through Ollama, vLLM, Hugging Face and llama.cpp. The
[component catalog](../catalog/components.md) lists all 55.

**Done when:**

- the component is registered with `paper=`, and with `claims=` when the paper
  reports a number on a benchmark Lasagna runs, so every leaderboard prints the
  paper's number beside yours;
- `lasagna component validate` passes;
- any weights it downloads are listed in `src/lasagna/core/model_licenses.py`;
- a results bundle compares it with a built-in control of its stage, at
  `n_runs: 1` when only retrieval is measured and 3 or more with an LLM in the
  loop;
- it ships in Lasagna through a pull request that meets
  [the pull request checklist](../contributing-components.md#pull-request-checklist),
  or as a package of your own with a `lasagna_components` entry point
  ([template](https://github.com/dolev31/lasagna/tree/main/examples/lasagna-plugin-template)).
  Both count; a package of your own needs no review from us.

Your method is not in the table? Propose it with the
[component form](https://github.com/dolev31/lasagna/issues/new?template=component_contribution.yml).
Open items are labeled
[component-wanted](https://github.com/dolev31/lasagna/issues?q=is%3Aopen+label%3Acomponent-wanted).

---

## Good first issues

Small, with the files named and a test to write. Labeled
[good first issue](https://github.com/dolev31/lasagna/issues?q=is%3Aopen+label%3A%22good+first+issue%22).

- **Claims for the built-in components.** A component can declare the number its
  paper reports; no built-in declares one yet. Each one filled in puts a
  published number beside a reproduced one.
- **A benchmark adapter.** MIRACL and BRIGHT, below.

---

## Benchmarks

Every built-in benchmark loads at a pinned revision, states its license and is
listed in `DATASETS.md`. Propose one with the
[benchmark form](https://github.com/dolev31/lasagna/issues/new?template=benchmark_request.yml);
open items are labeled
[benchmark](https://github.com/dolev31/lasagna/issues?q=is%3Aopen+label%3Abenchmark).

| Benchmark | Paper | Why |
|---|---|---|
| MIRACL | [2210.09984](https://arxiv.org/abs/2210.09984) | Lasagna's benchmarks are English. MIRACL has human-judged relevance in 18 languages; start with one. |
| BRIGHT | [2407.12883](https://arxiv.org/abs/2407.12883) | Relevance that needs reasoning, not word overlap. Shows whether a reranker or query enhancer helps when matching is not enough. |

To evaluate on data of your own, no adapter is needed: see
[your own data](../own-data.md).

---

## Research questions

Open questions that Lasagna can answer with a sweep. Write the plan before the
run (the question, what changes, what is held fixed, and the result that would
prove you wrong) with the
[research form](https://github.com/dolev31/lasagna/issues/new?template=research_proposal.yml).
Either answer is worth publishing. Labeled
[research](https://github.com/dolev31/lasagna/issues?q=is%3Aopen+label%3Aresearch).

- **Does composition hold with an open-weight generator?** The
  [paper](https://arxiv.org/abs/2610.29722) tests whether the effect of changing
  one stage depends on the others, and every answer in it comes from one
  proprietary model, GPT-5.1.
- **How much of the scaling loss survives out-of-domain distractors?** The paper
  grows the corpus with distractors drawn from the same five domains as the
  corpus, and calls the result directional rather than a scaling law. Real
  corpora often grow with documents from other domains.
- **How well does the judge agree with people?** Faithfulness is scored by an LLM
  judge, and Lasagna does not yet measure that judge against human labels.

---

## Deployment

For people running Lasagna as a service. Labeled
[deployment](https://github.com/dolev31/lasagna/issues?q=is%3Aopen+label%3Adeployment).

- **Vector backends: Chroma, LanceDB, Elasticsearch.** Six run today. Each new
  one is an adapter class and the shared conformance test, which says where an
  adapter differs. See [vector backends](../guides/vector-backends.md).
- **An OpenAI-compatible answer endpoint, with streaming.** The REST API
  searches, but has no route that retrieves and then answers, and nothing
  streams.
- **Weights & Biases and MLflow.** A run writes a results bundle; nothing sends
  it to a tracker yet.

Also planned for 0.2.0, with no issue yet: export and import of a session's
documents and chunks, a prebuilt Wikipedia index, and batched generation with
vLLM. Open an issue before you start one.

---

## How to claim an item

1. Comment on its issue: "I'm taking this."
2. Open a draft pull request within two weeks, or the claim lapses and someone
   else can take it.
3. Before you ask for review, meet [the bar](contributing.md#the-bar-for-a-pull-request).

Questions go to
[Discussions](https://github.com/dolev31/lasagna/discussions/categories/q-a).
