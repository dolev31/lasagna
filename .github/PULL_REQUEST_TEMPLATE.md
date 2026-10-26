<!--
Fill in every section. The "PR quality" check reads this description and fails
while a section is empty, still holds only these comments, or a required box is
unticked. The bar, and why it exists:
https://github.com/dolev31/lasagna/blob/main/CONTRIBUTING.md#quality-bar
-->

## Why

<!-- The problem this solves, or the question it answers. Link the issue if
there is one ("Closes #12"). The diff shows what changed; say why. -->

## What changed

<!-- Two to five lines a reviewer reads before the diff. Name any public
function, CLI flag, setting or REST field that is new or behaves differently. -->

## How it was tested

<!-- Name the test that fails without this change and passes with it, and the
command you ran. A docs-only change: say how you checked it (for example
`mkdocs build --strict`). -->

## Evidence

<!-- If this can move a benchmark number: paste the `lasagna results
significance` output, or attach the bundle from `lasagna results bundle`,
against a built-in control. If it cannot, write one line that starts with
"No number can move:" and say why. -->

## Agent use

<!-- AI coding agents are welcome, under the same bar. Say which you used, or
"none", and name the person who read every line of the diff and answers for it.
Keep the two lines below and fill them in. -->

Written with:
Read in full by:

## Checklist

- [ ] This PR makes one logical change.
- [ ] `scripts/ci-local.sh` passes, or at least `ruff check .`, `ruff format --check .` and the test suite.
- [ ] Every new `noqa`, `type: ignore` or `pragma: no cover` names its code and gives a reason.
- [ ] `CHANGELOG.md` has a line under Unreleased, or nothing a user sees changed.
- [ ] I read every line of the diff, and I can explain each one.
