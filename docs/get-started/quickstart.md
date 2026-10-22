# Quickstart

This page indexes one document and searches it, on the `speed` profile: a 92 MB
model, no reranker, no LLM and no API key.

## From Python

```python
import asyncio
from lasagna.sdk import KnowledgeClient, config_for_speed

async def main():
    client = KnowledgeClient(session_id="quickstart", config=config_for_speed())
    doc = await client.upload("paper.pdf")
    await client.documents.wait_for_processing(doc.document_id)  # upload returns once queued
    results = await client.search_documents("What did the ablation show?")
    for r in results.results:
        print(f"[{r.score:.2f}] {r.chunk.content[:120]}")
    await client.close()

asyncio.run(main())
```

`upload()` returns once the file is queued, so wait for processing before you
search. `wait_for_processing` raises `DocumentProcessingError` with the reason
and the fix if the file cannot be indexed.

Run it twice and the second run uploads nothing new: the document is reported
with `duplicate_of`. Pass `allow_duplicates=True` to index it again.

## From the shell

```bash
lasagna configure --profile speed
lasagna upload paper.pdf
lasagna search "what did the ablation show?"
```

`lasagna init` asks the same questions interactively. `lasagna upload --force`
indexes a file again.

To serve the same index over HTTP, run:

```bash
lasagna serve
```

That starts the REST API and the configuration page on `127.0.0.1:8000`. While
it runs, the CLI asks the server instead of opening the store itself, because a
local store opens in one process at a time.

## What to expect

These timings are from an M1 Max (10 cores, 64 GB) on the `speed` profile:

| Step | Time |
|---|---|
| First run, including the 92 MB model download | 30 to 35 s |
| First run, model already cached | 8 s |
| One search, one document (median of ten) | 15 ms |
| One search over 18,442 chunks on Milvus Lite (median) | 32 ms |
| One `lasagna search` command | about 10 s |

A CLI command is slow because each one loads the model and opens the store. Use
`lasagna serve` or the Python client when you search more than once.

The default profile (BGE-M3 plus the mxbai reranker) downloads about 3.2 GB and
takes 117 ms per warm search on the M1 Max GPU and 3.1 s on two CPU threads.

## How settings are kept

Settings belong to a session. Each session has its own embedding model, and its
data lives in `~/.lasagna`. A client created without `config=` reads the
session's saved settings. See [Configuration](../configuration.md) for the rules
about which setting wins.

## Next

[Your first comparison](first-comparison.md) tests a reranker of your own.
