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

## Find something to work on

[Wanted](wanted.md) lists the components, benchmarks and studies the project
would like to have, each with what "done" means. Comment on an item's issue to
claim it.

## The bar for a pull request

Every pull request is read in full by a person who will maintain it, so the bar
is the same whoever or whatever wrote the code. A check runs on every pull
request and fails until:

- the description fills in the template: why, what changed, how it was tested,
  the evidence, and which agent was used, if any;
- a person names themselves, by GitHub handle, as having read every line of the
  diff;
- every suppression in the lines added names its code and gives a reason, and
  every TODO links its issue;
- a change to `src/lasagna/` comes with a test and a changelog line.

Run it before you open the pull request:

```bash
python scripts/check_pr_body.py --file body.md --diff origin/main
```

AI coding agents are welcome under the same bar. A pull request below it is
closed with a link to
[the quality bar](https://github.com/dolev31/lasagna/blob/main/CONTRIBUTING.md#quality-bar),
and can be opened again once it meets it.

## The rest

The [contributing guide](https://github.com/dolev31/lasagna/blob/main/CONTRIBUTING.md)
covers the definition of done for each kind of change, bug reports and how to
disagree about a number. The
[code of conduct](https://github.com/dolev31/lasagna/blob/main/CODE_OF_CONDUCT.md)
applies to every space.
