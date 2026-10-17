# Contributing to Lasagna

Thanks for your interest. Lasagna is a research framework, so the most valuable
contributions are usually new components to compare (a reranker, chunker, query
enhancer or agentic pipeline) rather than changes to the core.

## Setup

```bash
git clone https://github.com/dolev31/lasagna && cd lasagna

# Use uv: pip takes about two minutes to resolve ".[all]" (measured with
# pip 26.2), uv a few seconds.
uv pip install -e ".[all]" --group dev

pre-commit install
pytest tests/ --ignore=tests/live --ignore=tests/stress
```

Before you push, `scripts/ci-local.sh` runs what CI runs on a push (lint and
type check, both secret scans, the suite with the coverage floor, backend
conformance, the license check and dependency audit, the plugin template) in
its own environments, and `--full` adds the E2E gate that pull requests run and
the release workflow's build and wheel smoke test.

Only have pip? Install the extras you actually need instead of `[all]`:

```bash
pip install -e ".[eval]" --group dev      # enough for the test suite
```

The default suite is hermetic: no network, no external services, and it never
touches your real `~/.lasagna` store. `tests/live/` needs services running
(`lasagna serve`); `tests/stress/` is long-running.

## Adding a component

Implement the stage's base class and register it. Declaring `paper` and `claims`
is encouraged: a run records them, and on a run of the benchmark a claim names,
its leaderboards (terminal, markdown, LaTeX) and `results view` show the
published number beside the reproduced one. Key claims by benchmark (add
`"split"` if the paper names one); a claim that names no benchmark is refused
at registration.

```python
from lasagna.core.registry import registry
from lasagna.core.base_components import BaseReranker

@registry.register(
    "reranker", "my_reranker", paper="Doe et al. 2026", claims={"beir/scifact": {"ndcg@10": 0.456}}
)
class MyReranker(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        ...
```

Match that signature exactly. `**kwargs` is not decoration: the pipeline passes
keywords a component never declared, and one without it raises `TypeError` the
first time it is called for real.

The decorator only runs when the module is imported, which is fine in-tree and
not enough for a separate package. To ship one, declare an entry point (see
[the plugin template](examples/lasagna-plugin-template/)) and check that discovery
found it with `lasagna plugins list`.

Eight types are pluggable: `embedder`, `chunker`, `reranker`, `retriever`,
`query_enhancer`, `document_processor`, `agentic_pipeline`, `generator`. Every
one but `agentic_pipeline` also takes a plain function with one decorator
(`@lasagna.reranker` and so on); see
[docs/contributing-components.md](docs/contributing-components.md).

**Your component must not read state it did not declare.** Stage isolation is
what makes an interaction estimate attributable rather than an artifact, so
`tests/test_stage_isolation_behavioural.py` enforces it. A component that
mutates its input will fail there, and that failure is the test working.
(`tests/test_abc_conformance.py` is a different check: it verifies shape
(subclassing and method signatures) and says nothing about runtime behavior.)

### If it downloads model weights

Add the model to
[`src/lasagna/core/model_licenses.py`](src/lasagna/core/model_licenses.py) with
its SPDX identifier, whether it permits commercial use, and a link to the model
card. `tests/test_model_licenses.py` fails the build when a selectable model has
no entry. That is deliberate, so nobody ships weights whose terms were never
checked.

Non-commercial models are welcome; comparing them is the point. They just have to
be labeled.

## Pull requests

- One logical change per PR, with tests.
- Run `ruff check .` and `ruff format .` (pre-commit does this).
- Explain *why* in the description; the diff already shows what.
- Never commit credentials, real corpora, or absolute paths from your machine.
  CI rejects all three.

## Reporting bugs

Include the Lasagna version, Python version, vector backend, and a minimal
reproduction. For retrieval-quality issues, the config you ran and the query
matter more than the stack trace.

## Disagreeing about a number

Show a failing test or a reproducible demonstration and the argument is over,
whoever wrote the code. If it is a published result rather than the library,
open a [result dispute](.github/ISSUE_TEMPLATE/result_dispute.yml); running
`lasagna results compare` usually settles it without anyone having to rule.
See [GOVERNANCE.md](GOVERNANCE.md).

## Security

Do not open a public issue for vulnerabilities: see [SECURITY.md](SECURITY.md).

## License

Contributions are accepted under [Apache 2.0](LICENSE). Do not paste in code you
did not write or that carries an incompatible license; Lasagna vendors no
third-party source and we intend to keep it that way.
