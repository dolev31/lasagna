# Significance testing

`lasagna results significance` answers one question: is the gap between two arms
larger than the noise in your queries? This page says how it decides, and why it
uses these methods.

## The paired sign-flip test

For each query, take the score of arm A minus the score of arm B. These are the
paired differences. If the arms were equivalent, each difference would be as
likely to be positive as negative. So every way of flipping the signs is equally
likely.

The p-value is the share of sign patterns whose summed difference is at least as
large, in absolute value, as the one you observed.

- A difference of exactly zero cannot change sign. It drops out.
- With at most 30 non-zero differences, Lasagna lists every pattern, so the p-value
  is exact.
- Above that, it draws random patterns (10,000 by default) and reports
  `(count + 1) / (draws + 1)`, which is never exactly zero.
- Scores are averaged across runs of the same arm first, so repeated runs do not
  count as extra queries.

Alongside the p-value you get a 95% confidence interval for the mean difference,
from a t distribution, and Cohen's d_z for the paired design (`mean / std` of the
differences).

### Why not a bootstrap

RAG evaluations often have 10 to 50 queries. At that size the sign-flip test with
a t interval holds its stated level. The paired bootstrap that Lasagna used before
did not: at 12 to 20 queries it rejected a true null 8 to 9 percent of the time at
alpha 0.05.

## Many arms: Holm-Bonferroni

Compare ten arms and you have 45 pairs. At alpha 0.05, a few pairs will look
significant by chance. `significance` corrects for that with the Holm-Bonferroni
step-down procedure:

1. Sort the p-values from smallest to largest.
2. Compare the smallest with `alpha / m`, the next with `alpha / (m - 1)`, and so
   on.
3. Stop at the first p-value that fails. It and every larger one are not
   significant.

This controls the chance of any false positive across the whole family, and it is
less strict than a plain Bonferroni correction.

## Reading the verdict box

The table ends in a box.

- **Green** names each pair that survives the correction and says which side won.
- **Yellow** says no pair survived and points at `lasagna results power`.

Yellow does not mean the arms are the same. Either they are the same, or the run
was too small to tell. `power` separates the two.

## Power

`lasagna results power results.json baseline candidate` reports the smallest
difference the run could have detected. Given the spread of the paired
differences, it computes the minimum detectable effect for the significance level
and power you ask for (alpha 0.05 and power 0.80 by default).

It is computed after the run, from the variance the run actually showed. An
up-front calculation would need a variance estimate that nobody has before the
run. If your observed difference is smaller than the minimum detectable effect,
the honest conclusion is "inconclusive", not "no difference". The report also
carries the pair's own uncorrected p-value, because a difference below the minimum
can still be significant.

## Interaction

`lasagna results interaction` takes a 2x2 run and asks whether two techniques
compose. It estimates the double difference:

```text
interaction = (both - neither) - ((only A - neither) + (only B - neither))
```

Positive means the techniques compound. Negative means they overlap, because both
fix the same errors. Two papers cannot answer this, because the interaction is not
a function of the two separate gains. See the
[paper study](the-paper-study.md) for why it matters.

## Where the code lives

The methods are `BenchmarkResults.significance_test`,
`pairwise_significance`, `minimum_detectable_effect` and `interaction_test` in
[results.py](https://github.com/dolev31/lasagna/blob/main/src/lasagna/knowledge_evaluation_sdk/results.py).
The [metrics reference](https://github.com/dolev31/lasagna/blob/main/src/lasagna/knowledge_evaluation_sdk/docs/metrics.md#8-statistical-testing)
gives every parameter and return key.
