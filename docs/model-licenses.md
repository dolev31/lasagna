# Model licenses

Lasagna ships **no model weights**. Every embedding, reranking and sparse-encoding
model is downloaded from its upstream host (usually the Hugging Face Hub) the first
time you use it, and each carries its own license, independent of Lasagna's
Apache 2.0.

Most are permissive. Four are **not**, and they are selectable on purpose:
Lasagna exists to compare RAG components, and excluding well-known models would
undercut that. What matters is that you know which one you picked.

> **Short version:** for research, evaluation and benchmarking, everything below
> is fine. Before putting a configuration into a commercial product, check this
> page: four models forbid that use, or do not say they allow it.

## Restricted: research and evaluation only

| Model | License | Where it appears | Notes |
| --- | --- | --- | --- |
| `naver/splade-cocondenser-ensembledistil` | CC-BY-NC-SA-4.0 | Default SPLADE sparse encoder | Non-commercial, **share-alike**. Sparse retrieval is opt-in, so this loads only when you enable it. |
| `jinaai/jina-reranker-v2-base-multilingual` | CC-BY-NC-4.0 | Reranker option | Non-commercial. Jina AI sells a commercial license separately. |
| `BAAI/bge-reranker-v2-gemma` | Gemma Terms of Use | Reranker option | Built on Gemma; inherits use restrictions. Not an OSI-approved open source license. |
| `castorini/monot5-base-msmarco-10k` | none stated | Default `monot5` reranker | The model card states no license. T5-base (Apache-2.0) fine-tuned on MS MARCO, whose terms allow non-commercial research use; ask the authors before using it in a product. Loads only when a sweep or setting names `monot5`. |

Selecting any of these logs a `WARNING` the first time the model loads, naming
the license and linking the model card. The warning is advisory. Lasagna will
not stop you, because for research use there is nothing to stop.

## Permissive: commercial use allowed

| Model | License |
| --- | --- |
| `BAAI/bge-m3` | MIT |
| `BAAI/bge-large-en-v1.5` | MIT |
| `BAAI/bge-base-en-v1.5` | MIT |
| `BAAI/bge-small-en-v1.5` | MIT |
| `BAAI/bge-reranker-base` | MIT |
| `BAAI/bge-reranker-large` | MIT |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 |
| `intfloat/e5-small-v2` | MIT |
| `intfloat/e5-base-v2` | MIT |
| `intfloat/e5-large-v2` | MIT |
| `intfloat/multilingual-e5-large` | MIT |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Apache-2.0 |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Apache-2.0 |
| `cross-encoder/ms-marco-electra-base` | Apache-2.0 |
| `mixedbread-ai/mxbai-rerank-large-v1` | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| `colbert-ir/colbertv2.0` | MIT |
| `ms-marco-TinyBERT-L-2-v2` (FlashRank's default, from `prithivida/flashrank`) | CC-BY-SA-4.0: attribution, and share-alike for adaptations |

A model named without its organization (`all-MiniLM-L6-v2`, the `speed`
profile's embedder) is sentence-transformers' own, and is looked up as such.
`tests/test_model_licenses.py` fails when the default model of a built-in
component, or a profile's model, has no entry here.

## Checking programmatically

```python
from lasagna.core.model_licenses import describe, is_commercial_use_restricted, restricted_models

describe("naver/splade-cocondenser-ensembledistil")
# LicenseInfo(spdx='CC-BY-NC-SA-4.0', commercial_use=False, url='https://huggingface.co/...', note=...)

is_commercial_use_restricted("BAAI/bge-m3")      # False
sorted(restricted_models())                       # every non-commercial model
```

Useful in a pre-deploy check:

```python
from lasagna.core.config import get_rag_config
from lasagna.core.model_licenses import is_commercial_use_restricted

config = get_rag_config()
for model in (config.embedding_model, config.reranker_model):
    if is_commercial_use_restricted(model):
        raise SystemExit(f"{model} is not licensed for commercial use")
```

## Bringing your own model

Lasagna does not restrict which model identifier you configure. Models absent
from the table load normally and log a `DEBUG` note that their license was not
checked. Absence means *uncataloged*, not *unrestricted*.

If you add a model to the configurator, add it to
[`src/lasagna/core/model_licenses.py`](https://github.com/dolev31/lasagna/blob/main/src/lasagna/core/model_licenses.py)
too. `tests/test_model_licenses.py` fails the build when the UI offers a model
the table does not cover, which is what keeps this page honest.

## Caveats

This table is a convenience, not legal advice, and licenses change. The model
card linked from each entry is authoritative; confirm there before relying on
anything here.

Lasagna also redistributes no benchmark corpora; evaluation datasets are fetched
from their official sources and carry their own terms.
