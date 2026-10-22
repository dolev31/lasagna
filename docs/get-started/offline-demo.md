# Offline demo and notebook

Two ways to see the whole workflow without setting up keys or datasets.

## The offline acceptance demo

`examples/acceptance/offline_demo.py` runs every feature of Lasagna 0.1.0 end to
end, offline, in under five minutes. It is in the source checkout, not in the
wheel.

```bash
git clone https://github.com/dolev31/lasagna.git
cd lasagna
pip install -e ".[eval]"
python examples/acceptance/offline_demo.py
```

It has eight steps, and each one does what a user would do and checks what the
user would see:

1. `benchmark synthesize --dry-run` calls no LLM and writes nothing. The real
   run writes 12 questions from the smoke documents, each marked synthetic.
2. `benchmark validate-data` accepts that file and names the row and the field
   of a deliberately broken copy.
3. `lasagna.from_langchain` turns a LangChain-style document compressor into a
   reranker.
4. `benchmark experiment --dry-run` lists the arms, tasks and models of the sweep
   and writes no results.
5. The sweep runs five arms (dense, the wrapped reranker, `closed_book`,
   `long_context` and `no_rag: true`) on both tracks.
6. The same sweep again replays every LLM response from the cache: hit rate 1.0,
   no LLM call, the same numbers.
7. `results bundle`, `results view` and `results leaderboard --format latex`.
8. `lasagna-mcp --help` answers in under 2 s, and the server's default tool list
   has no tool that deletes data.

Nothing needs a network or an API key. A deterministic stub stands in for the
LLM, put in through `set_llm_provider`, so token counting, temperature pinning
and the response cache run as they do for a real provider. Retrieval is real: it
uses the `speed` profile's all-MiniLM-L6-v2 from the Hugging Face cache with
`HF_HUB_OFFLINE=1`, and a socket guard refuses every connection that is not to
your machine. The demo makes its own home folder inside its working folder, so
your `~/.lasagna` is left alone.

It prints one line per step with its time. It exits 1 at the first failed check,
with the reason. Exit 2 means it could not start, for example because a model
is not in the cache. Run it once with `--download-models` to fetch the model,
and use `--workdir DIR` to choose where it works.

## The notebook

[Compare your reranker](https://colab.research.google.com/github/dolev31/lasagna/blob/main/examples/notebooks/compare_your_reranker.ipynb)
opens in Google Colab. It tests a reranker on 100 SciFact queries over the full
corpus, with a cross-encoder as the control. It prints the significance table,
runs the stage isolation check and writes a validated bundle. It needs no API
key and is stored without outputs. The same code is in
`examples/notebooks/compare_your_reranker.py`.
