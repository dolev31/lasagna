"""Write results.csv from a run of beir_tier1.yaml.

    lasagna benchmark experiment examples/reproduction/beir_tier1.yaml
    python examples/reproduction/results_csv.py <the run's Output folder> > results.csv

Each row is one method on one dataset: NDCG@10 from the method's arm, recall@100
from its ``_k100`` arm, the query count, how many queries got fewer than ten
documents, and where the numbers came from: the run's date, the package and
metrics versions it recorded, and the commit of the source checkout it ran
from (``environment.git_commit``; pass ``--commit`` when the run was made
outside a git checkout). Diff the output against the results.csv in this
folder to compare a rerun with the published numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

METHODS = (
    "bm25",
    "bm25_k1.2_b0.75",
    "dense_minilm",
    "dense_bge_small",
    "hybrid_rrf_minilm",
    "hybrid_rrf_bge_small",
    "bm25_ce",
)
DATASETS = ("beir/scifact", "beir/nfcorpus")
COLUMNS = (
    "method",
    "dataset",
    "split",
    "ndcg@10",
    "recall@100",
    "n_queries",
    "short_lists_at_10",
    "run_date",
    "package_version",
    "metrics_version",
    "commit",
)


def rows(run: dict, commit: str = "") -> list:
    """One row per method and dataset, from a run's results.json."""
    meta = run["experiment_config"]
    commit = commit or (meta.get("environment") or {}).get("git_commit") or ""
    tasks = {(t["benchmark_name"], t["config_name"]): t for t in run["results"] if not t["error"]}
    out = []
    for dataset in DATASETS:
        for method in METHODS:
            at_10, at_100 = tasks[(dataset, method)], tasks[(dataset, f"{method}_k100")]
            scored = at_10["per_sample_scores"]
            out.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "split": at_10["split"],
                    "ndcg@10": f"{at_10['metrics']['ndcg@10']:.4f}",
                    "recall@100": f"{at_100['metrics']['recall@100']:.4f}",
                    "n_queries": len(scored),
                    "short_lists_at_10": sum(
                        1 for s in scored if len(set(s.get("_retrieved_ids") or [])) < 10
                    ),
                    "run_date": meta["created_at"][:10],
                    "package_version": meta["package_version"],
                    "metrics_version": meta["metrics_version"],
                    "commit": commit[:7],
                }
            )
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="the run's Output folder, or its results.json")
    parser.add_argument("--commit", default="", help="the commit the run was made at")
    args = parser.parse_args(argv)
    path = Path(args.run)
    run = json.loads((path / "results.json" if path.is_dir() else path).read_text())
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows(run, args.commit))


if __name__ == "__main__":
    main()
