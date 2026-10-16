# Lasagna plugin template

A complete, installable package that adds **a reranker** and **a benchmark** to
Lasagna. Copy the directory, rename it, replace the two implementations.

This is a real package rather than a cookiecutter on purpose: Lasagna's CI
installs it and runs its tests, so it cannot quietly rot into instructions that
no longer work.

```
src/lasagna_plugin_template/
  reranker.py    a BaseReranker            -> lasagna_components
  benchmark.py   a BaseBenchmarkAdapter    -> lasagna_benchmarks
tests/test_plugin.py
pyproject.toml   the entry points are the whole integration
```

---

## The integration is two entry points

```toml
[project.entry-points."lasagna_components"]
# "{stage}.{name}"; stages: reranker, embedder, chunker, retriever,
# document_processor, query_enhancer, agentic_pipeline, generator
"reranker.template_length" = "lasagna_plugin_template.reranker:LengthReranker"

[project.entry-points.lasagna_benchmarks]
"template_tiny" = "lasagna_plugin_template.benchmark:TinyBenchmark"
```

That's it. No manifest, no registration call, no import from Lasagna at start-up.
`pip install` your package and it appears in the registry.

---

## The loop

```bash
pip install -e .

# 1. Is it discovered?
lasagna plugins list

# 2. Does it satisfy the stage contract?
lasagna component validate lasagna_plugin_template.reranker:LengthReranker

# 3. Does it run end to end?  (speed profile: keyless, small models)
lasagna benchmark run template_tiny --profile speed

# 4. Does it help?  (two arms, everything else held fixed)
lasagna benchmark experiment sweep.yaml

# 5. Package the result so someone else can check it
lasagna results bundle "$HOME/.lasagna/research/results/<run>/results.json" -o run.json
lasagna results validate run.json
```

Keep the quotes in step 5. The run directory is named after the sweep's `name`,
and free text can hold spaces or a `?`, which the shell would split or expand
(zsh stops with "no matches found").

Without `--profile speed`, step 3 runs the `standard` profile, which downloads
about 3.1 GB of models. `benchmark run` measures retrieval unless asked for
more; `--tracks retrieval,generation` adds answer generation, which needs an
LLM API key and the `eval` extra.

Step 4 is the one the template exists for, and it needs `sweep.yaml` rather
than `benchmark run`. A single-arm run produces one number from the stock
pipeline, which reads exactly like a score for your component and is not one.
The sweep runs the same corpus and sample through a control *and* your
component, so the gap between them belongs to your component.

Step 2 is the one worth running before anything else. `implements_contract:
True` in the registry means your component is interchangeable with the
built-ins, not merely that it registered. A component that registers but
does not conform still works in its own pipeline, but a comparison against one
that does conform is measuring the interface difference as well as your idea.

---

## Three things the template does deliberately

**The reranker is obviously wrong.** It ranks by word count. A plausible
component would leave you unable to tell "it ran and happened to agree with the
retrieval order" from "it never ran at all". Something visibly wrong proves the
swap reached execution, and you should keep a control in your own comparisons
for the same reason.

**It deep-copies before touching anything.** The pipeline may hand the same
objects to more than one component. A stage that mutates its input changes
results for whatever runs after it, which makes a measured difference
unattributable: the exact thing this framework exists to prevent.

**The benchmark refuses an unknown split.** Returning the split you *do* have is
worse than failing: a run reporting `split: test` over `validation` data is not
comparable to anyone else's, and nothing downstream can detect it.

---

## Publishing

Once your component beats its control on a real benchmark, the bundle from
step 5 is what makes the claim checkable. It carries the environment, the
library versions, the resolved sweep, the corpus fingerprint, and a
comparability key covering everything held fixed, so a reader can tell whether
your number and theirs belong in the same table.

`lasagna results validate` recomputes that key and the run id from the recorded
fields, so an edited number no longer matches its own bundle.

See [the guide](../../docs/guide.md) for the full workflow, and
[CONTRIBUTING](../../CONTRIBUTING.md) if you want your component in Lasagna
itself rather than in your own package.
