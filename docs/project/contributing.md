# Contributing

The most useful contributions are new components to compare: a reranker, chunker,
query enhancer or agentic pipeline. Changes to the core are welcome too.

## Set up

```bash
git clone https://github.com/dolev31/lasagna && cd lasagna
uv pip install -e ".[all]" --group dev
pre-commit install
pytest tests/ --ignore=tests/live --ignore=tests/stress
```

The suite keeps the Hugging Face hub offline and downloads nothing. A test that
needs weights you have not cached skips and names them.

## Add a component

Read [write a component](../guides/write-a-component.md) for the short path, and
[contributing components](../contributing-components.md) for the full contract,
the conformance checks and the pull request checklist.

## Build these docs

The docs use MkDocs. Install the pinned toolchain and build in strict mode, which
fails on any broken link:

```bash
pip install --group docs
mkdocs build --strict
mkdocs serve
```

Three pages are generated from the code: the [CLI reference](../reference/cli.md),
the [catalog](../catalog/stages.md) and the [settings reference](../reference/settings.md).
Do not edit them by hand. Each has a script in `scripts/` and a test that fails
when the page is stale. Every `lasagna ...` command in a docs code block is
checked against the real command line.

## The rest

The [contributing guide](https://github.com/dolev31/lasagna/blob/main/CONTRIBUTING.md)
covers pull requests, bug reports and how to disagree about a number. The
[code of conduct](https://github.com/dolev31/lasagna/blob/main/CODE_OF_CONDUCT.md)
applies to every space.
