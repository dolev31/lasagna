# Citing Lasagna

If you publish work that uses Lasagna, please cite it. GitHub renders
[CITATION.cff](https://github.com/dolev31/lasagna/blob/main/CITATION.cff) as a
ready-made entry.

## The paper

The paper is on arXiv as [2610.29722](https://arxiv.org/abs/2610.29722).

```bibtex
@misc{levy2026retrieval,
  title         = {Retrieval Augmented Generation Compositionality},
  author        = {Levy, Ido and Shlomov, Segev and Choshen, Leshem},
  year          = {2026},
  eprint        = {2610.29722},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.29722}
}
```

## The software

The DOI [10.5281/zenodo.23548068](https://doi.org/10.5281/zenodo.23548068) names
every version of the software and resolves to the latest one. Each release also has
its own DOI on Zenodo.

## Publish the bundle with your number

If you report a number from the evaluation harness, publish its results bundle
with it:

```bash
lasagna results bundle results.json -o run.json
```

The bundle carries what a reader needs to decide whether your number and theirs
measure the same thing. See [comparability and provenance](../concepts/comparability.md).
