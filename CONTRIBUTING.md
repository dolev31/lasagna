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

The suite keeps the Hugging Face hub offline and downloads nothing: a test that
needs weights you have not cached skips and names them. To run those tests and
download what they name, as CI's weights job does:
`HF_HUB_OFFLINE=0 pytest tests/ --ignore=tests/live --ignore=tests/stress -m needs_embedding_weights`.

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

Match that signature. The pipeline calls `rerank(query, documents, top_k=k)` and
passes nothing else to a component of yours, so a `rerank` without `**kwargs`
runs too, and `lasagna component validate` passes it. Keep `**kwargs` anyway: the
built-in rerankers also receive the session's `config` that way, and a component
that accepts it keeps working if a later release passes it on.

The decorator only runs when the module is imported, which is fine in-tree and
not enough for a separate package. To ship one, declare an entry point (see
[the plugin template](examples/lasagna-plugin-template/)) and check that discovery
found it with `lasagna plugins list`.

Eight types are pluggable: `embedder`, `chunker`, `reranker`, `retriever`,
`query_enhancer`, `document_processor`, `agentic_pipeline`, `generator`. Every
one but `agentic_pipeline` also takes a plain function with one decorator
(`@lasagna.reranker` and so on); see
[docs/contributing-components.md](docs/contributing-components.md).

**Your component must not change what another stage sees, and should read only
what its stage hands it.** Stage isolation is what makes an interaction
estimate attributable rather than an artifact. Part of it is enforced:
`tests/test_stage_isolation_behavioural.py` fails when a registered reranker or
chunker mutates its input or when chunks share a metadata dict (that failure is
the test working), `lasagna component validate` checks your component does not
mutate its input, and every sweep compares the configuration that reached each
stage across its arms. Reading state the stage did not hand over (a global, the
session store, a file) is not checked by anything, so keep to the arguments.
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

## Quality bar

A person who will maintain your code after you move on reads every pull
request, so the bar is the same for everyone: a first-time contributor, a
maintainer, or an AI coding agent working for either. The "PR quality" check
([`scripts/check_pr_body.py`](scripts/check_pr_body.py)) tests what a machine
can test before a reviewer opens the diff. Run it yourself first:

```bash
python scripts/check_pr_body.py --file body.md --diff origin/main
```

Looking for something to work on? [Wanted](docs/project/wanted.md) lists what
Lasagna needs, with a definition of done for each.

### Every pull request

- One logical change, with tests. A refactor and a feature are two pull
  requests. For a new stage, a change to the core, or anything over about 500
  lines, open an issue first, so the design is agreed before the code is
  written.
- The [template](.github/PULL_REQUEST_TEMPLATE.md) is filled in: why, what
  changed, how it was tested, evidence, agent use, and the checklist. Explain
  *why*; the diff already shows what.
- `scripts/ci-local.sh` passes, or at least `ruff check .`, `ruff format --check .`
  and the suite. Pre-commit runs ruff for you.
- New behavior and every bug fix come with a test that fails without the change.
  A change under `src/lasagna/` with nothing changed under `tests/` fails the
  check, unless a maintainer adds `no-new-test` to a pure refactor.
- Anything a user sees gets a line in `CHANGELOG.md` under Unreleased. A
  maintainer adds `no-changelog` when nothing changed for users.
- Every suppression says why: `# noqa: BLE001 - reason`,
  `# type: ignore[arg-type]`, `# pragma: no cover - reason`,
  `pytest.skip("reason")`. An `except` that only passes says in a comment why the
  error is safe to ignore.
- A TODO links its issue, as `TODO(#12)`. A new module under `src/lasagna/` opens
  with a docstring that says what it is for.
- No benchmark number moves unless that is the point, and then the description
  says which numbers and why. A change to a metric's definition bumps
  `METRICS_VERSION` (see [GOVERNANCE.md](GOVERNANCE.md#breaking-changes)). A
  maintainer adds `needs-evidence` when a description says a number moves but
  does not show it.
- Never commit credentials, real corpora, or absolute paths from your machine.
  CI rejects all three.
- It was tested in the production environment: the Docker Compose stack, built
  from the branch ([below](#tested-in-the-production-environment)). "How it was
  tested" shows the `docker compose` commands you ran and what came out.

### Tested in the production environment

Lasagna runs in production as the Docker Compose stack in `src/lasagna/deploy`.
A local install can behave differently: another version of the vector store,
other storage settings, other system libraries. Code that ran only outside the
stack is untested here, so every pull request runs its change in the stack,
built from its branch, and shows it.

1. **Get Docker.** Each of these is free:
   - [Rancher Desktop](https://rancherdesktop.io/), open source, on macOS,
     Windows and Linux.
   - Docker Engine, on Linux.
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/), for
     personal use, education and small businesses.
   - A GitHub Codespace on your fork. It runs Docker, and a personal account has
     free monthly hours. Use it when your machine, or the sandbox your agent
     works in, has no Docker daemon.
2. **Build the stack from your branch, and start it.**

   ```bash
   cd src/lasagna/deploy
   cp .env.example .env
   # In .env: fill RAG_JWT_SECRET and POSTGRES_PASSWORD (for example with
   # `openssl rand -hex 32`), and set IMAGE_EXTRAS=eval if your evidence runs
   # `lasagna benchmark`. VECTOR_BACKEND=milvus is the default; a change to
   # another store sets it to that file in backends/ and uses that file below.
   docker compose -f docker-compose.yml -f backends/milvus.yml up -d --build
   docker compose -f docker-compose.yml -f backends/milvus.yml ps
   ```

3. **Run your change there.** The command behind your evidence runs inside the
   api container:

   ```bash
   docker compose -f docker-compose.yml -f backends/milvus.yml exec api \
     lasagna benchmark run smoke --profile speed --tracks retrieval -o /tmp/smoke
   ```

   In the stack built from `main` this prints `smoke` with ndcg@10 0.9464, so a
   different number there is your change. A stress or live test runs from your
   checkout against the stack's ports, for example with
   `QDRANT_URL=http://localhost:6333` against the Qdrant stack. A change to one
   vector backend runs against that backend's file in `backends/`.
4. **Show it.** Under "How it was tested", paste the `docker compose` commands,
   the image lines from `docker compose ps`, and what came out. The "PR quality"
   check fails while that section names no `docker compose` command, and the
   checklist has a box for it.

Running it is your part of the pull request. Maintainers do not run your
evidence for you, in Docker or anywhere else, and a pull request whose change
has not run in the stack goes back to its author. A maintainer may rerun your
command before merging, to check your result; that rerun never stands in for
it.

### Done, by kind of change

| Change | Done when |
|---|---|
| Component | `lasagna component validate` passes; `paper=` and `claims=` are declared when it implements a published method; any weights it downloads have a license entry ([above](#if-it-downloads-model-weights)); a results bundle compares it with a built-in control, at `n_runs: 1` for retrieval with fixed models and 3 or more with an LLM in the loop. |
| Benchmark | It loads at a pinned `hf_revision`; its terms are in `DATASETS.md` (`lasagna benchmark licenses -o ...`); its tests run offline, as the whole suite does; the README's benchmark count is updated (the suite fails until it is). |
| Bug fix | The test that reproduces the bug is in the same pull request and fails on `main`. |
| Docs | `mkdocs build --strict` passes, and every command shown was run. |
| REST API or MCP | `tests/snapshots/` is updated in the same pull request, and an incompatible change follows the deprecation rule in [GOVERNANCE.md](GOVERNANCE.md#breaking-changes). |

### AI coding agents

Agents are welcome, and the bar does not move for them.

- A person opens the pull request, reads every line of the diff, and names
  themselves under "Read in full by:". That person answers review comments.
- "Written with:" names the agent, or "none". Using an agent is not a mark
  against a pull request; hiding it is. A maintainer may add `agent-assisted`,
  for the record.
- An agent's confidence is not evidence. A test that was run is, and so is the
  output of `lasagna results significance`.
- [AGENTS.md](AGENTS.md) gives an agent this repository's rules. Point yours at
  it.

### Below the bar

The check fails and says what is missing. Fix the description or the diff and it
runs again. A pull request a reviewer still finds below the bar (no reason for
the change, untested behavior, a diff its author cannot explain) is closed with
a link to this section. That is a statement about the pull request, not about
you: open it again once it meets the bar. A pull request whose description does
not match its diff, or one of many opened at once, is closed without review.

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
