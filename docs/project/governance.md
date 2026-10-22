# Governance

Lasagna has one maintainer today, Ido Levy ([@dolev31](https://github.com/dolev31)),
who has the final say on what merges. That describes where the project is. It is
not a permanent structure. If a second person reviews regularly and their judgment
is good, they get commit rights, and the governance file changes.

## Correctness is settled by evidence

Deciding what merges does not decide what is true. If you can show that a metric,
a statistic or a benchmark adapter is wrong, that ends the argument, whoever wrote
it. The bar is a failing test or a reproducible demonstration.

## Result disputes

If you think a published result is wrong, open a result dispute on GitHub. Most
resolve with one command:

```bash
lasagna results compare theirs.json yours.json
```

The field-level diff usually shows that the two runs held different things fixed.
The [comparability key](../concepts/comparability.md) exists for this.

Read the full
[governance file](https://github.com/dolev31/lasagna/blob/main/GOVERNANCE.md).
