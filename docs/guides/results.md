# Read and publish results

Every run writes a `results.json` and prints its path. Each command on this page
takes it.

## Answer a question about the run

| Question | Command |
|---|---|
| Is the gap real, across every pair of arms? | `lasagna results significance results.json --all` |
| Could this run have detected the effect I care about? | `lasagna results power results.json baseline candidate` |
| Do two techniques compose, or overlap? | `lasagna results interaction results.json` (a 2x2 run) |
| Which queries did an arm lose, at retrieval or generation? | `lasagna results per-query results.json --sort-by ndcg@10` |
| Are two runs comparable at all? | `lasagna results compare a.json b.json` |

`significance` runs a paired sign-flip test on every pair of arms and corrects
the p-values with Holm-Bonferroni. It ends in a box that says what the table
supports. Green names each pair that survives and which side won. Yellow says
nothing survived and points at `power`. The
[significance concept page](../concepts/significance.md) explains the method.

## Share the whole run as one page

```bash
lasagna results view results.json -o view.html
```

The page carries the versions and run id, the stage isolation check, and, for each
benchmark, the leaderboard, the significance table with its verdict, every metric
per arm, and a per-query diff between two arms. Use `--diff A B` to pick the arms.
The default is the top two. Each diff shows what each arm retrieved.

It is one file with no external requests. It opens offline, and you can attach it
to a review.

## Put a leaderboard in a paper

```bash
mkdir -p bundles
lasagna results bundle results.json -o bundles/run.json
lasagna results leaderboard bundles/ --format latex
```

`--format latex` prints one booktabs table for each comparability group. Rows in
one table held the same configuration fixed. Rows in different tables are never
ranked together. With `-o`, it writes one `.tex` file per benchmark instead. Arm
names are escaped for LaTeX.

Every `results` command that reads a bundle checks it as `results validate` does.
A bundle that fails is refused with its problems listed, or left out of a folder's
leaderboard with its name printed. A failure means a number was edited after the
bundle was made, or provenance is missing. `--allow-invalid` on `leaderboard` and
`view` shows the bundle anyway, with every number marked INVALID. A `results.json`
is not a bundle, and is shown as not validated.

## Publish a number someone else can check

```bash
lasagna results bundle results.json -o run.json
lasagna results validate run.json
```

The bundle carries the library versions, the hardware, the resolved sweep, the
corpus fingerprint and the comparability key. `validate` recomputes the key, the
run id and every aggregate from the per-query scores.

It also lists integrity warnings that do not invalidate a bundle:

- `SINGLE_RUN`: one run of a pipeline in which something samples.
- `UNRESOLVED_JUDGE_ALIAS`: a judge named by a floating alias.
- `PARTIAL_CORPUS`: some corpus documents failed to ingest.

`--strict` turns the warnings into failures. A run records `nothing_samples` when
only retrieval ran and no task called an LLM. One run of it is complete and raises
no `SINGLE_RUN`.

See [comparability and provenance](../concepts/comparability.md) for what the key
holds fixed. The [results schema](../reference/results-schema.md) lists every
field.

## The results hub

The [results hub](https://github.com/dolev31/lasagna/blob/main/hub/README.md) runs
a public leaderboard from bundles. A submission is a pull request that adds one
JSON file. The hub marks a row verified only when someone else's bundle reproduces
it: within its spread, which needs at least two runs, or by equalling its mean when
nothing in the run samples.

## Commands

| Command | What it does |
|---|---|
| `lasagna results significance` | Test every pair of arms. |
| `lasagna results power` | Estimate whether the run could detect an effect. |
| `lasagna results interaction` | Test whether two techniques compose. |
| `lasagna results per-query` | List the queries an arm lost. |
| `lasagna results compare` | Check whether two runs are comparable. |
| `lasagna results view` | Write a one-file HTML report. |
| `lasagna results bundle` | Package a run with its provenance. |
| `lasagna results validate` | Re-check a bundle. |
| `lasagna results leaderboard` | Build leaderboards from bundles. |
| `lasagna results schema` | Print the bundle schema. |

Every option is in the [CLI reference](../reference/cli.md).
