# Comparability and provenance

Two numbers can be compared only if everything except the thing under test was
held fixed. Nothing in a plain results table says what was held fixed. Lasagna
records it, and hashes it into a comparability key.

## The comparability key

The key is a 16-character hash over the configuration a comparison holds fixed:

- the metrics version
- the benchmark, split, number of samples, sampling seed and sample-id digest
- the corpus fingerprint and its provenance
- the resolved run mode
- the embedder model and dimension
- the chunker
- the retriever's top-k
- the vector backend
- the judge model
- BM25 stemming
- the stage-manifest digest of every stage the run does not vary

The key excludes the dimension under test. That is the design. If the key covered
everything, no two rows would be comparable. If it covered nothing, all rows would
be comparable no matter what they measured. The dimension under test is recorded
separately as `varied_dimensions`.

A field that was not captured is recorded as missing, so it hashes differently from
a field that was captured. An unknown embedder cannot match a known one.

## The metrics version

The metrics version is part of the key. If the definition of NDCG@10 changes, old
numbers fall out of comparison instead of being mixed with new ones. This release
is metrics version 1.7. Results from earlier builds never share a leaderboard group
with it.

## Provenance in a bundle

`lasagna results bundle` packages a run with what a reader needs to judge it:

- the Lasagna version, Python version and platform, with no paths or host names
- the resolved sweep
- the corpus fingerprint
- the comparability key of each benchmark and arm
- the per-query scores
- the integrity warnings and blocking problems

`lasagna results validate` recomputes the key, the run id and every aggregate from
the per-query scores. A number edited after the run no longer matches.

That is an integrity check against mistakes and simple edits. It is not a
signature: anyone who edits a number can recompute the run id. The hub therefore
treats results as self-reported.

## Leaderboards group by key

A leaderboard groups rows by benchmark and comparability key. Rows in one group
held the same configuration fixed, so you can rank them. Rows in different groups
are never ranked together. Use `lasagna results compare a.json b.json` to see the
field-level difference between two runs.

## Reproduction by collision

The [results hub](https://github.com/dolev31/lasagna/blob/main/hub/README.md)
marks a row verified when a different submitter's bundle with the same
comparability key lands inside the row's mean and spread. When nothing in the run
samples, the hub accepts an exact match of the mean.

## Related pages

- [Read and publish results](../guides/results.md)
- [Results schema](../reference/results-schema.md)
- [Stage isolation](stage-isolation.md)
