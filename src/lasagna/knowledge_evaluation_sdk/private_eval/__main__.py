"""CLI entrypoint for private evaluation dataset management and experiment runs.

Usage::

    # Build the dataset (downloads benchmarks from HuggingFace)
    python -m lasagna.knowledge_evaluation_sdk.private_eval build

    # Show dataset statistics
    python -m lasagna.knowledge_evaluation_sdk.private_eval stats

    # Verify dataset integrity (checksums + leakage check)
    python -m lasagna.knowledge_evaluation_sdk.private_eval verify

    # Register adapters in benchmark registry (for experiment runner)
    python -m lasagna.knowledge_evaluation_sdk.private_eval register

    # Run all (or a subset of) YAML experiments concurrently
    python -m lasagna.knowledge_evaluation_sdk.private_eval run-suite
    python -m lasagna.knowledge_evaluation_sdk.private_eval run-suite --dir experiments/ --concurrency 4

    # Generate master cross-experiment report from completed results
    python -m lasagna.knowledge_evaluation_sdk.private_eval report
    python -m lasagna.knowledge_evaluation_sdk.private_eval report --results-dir ~/.lasagna/research/results/ --output report.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_DATASET_DIR = str(Path(__file__).parent / "dataset")
_EXPERIMENTS_DIR = str(Path(__file__).parent / "experiments")


def cmd_build(args: argparse.Namespace) -> None:
    """Build the private evaluation dataset."""
    from .builder import DatasetBuilder

    builder = DatasetBuilder(
        seed=args.seed,
        output_dir=args.output_dir or _DATASET_DIR,
        n_distractors_per_benchmark=args.distractors,
    )
    manifest = builder.build()
    print(manifest.summary())


def cmd_stats(args: argparse.Namespace) -> None:
    """Show dataset statistics."""
    from .manifest import DatasetManifest

    manifest_path = os.path.join(args.output_dir or _DATASET_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"No manifest found at {manifest_path}. Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    manifest = DatasetManifest.load(manifest_path)
    print(manifest.summary())


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify dataset integrity."""
    from .manifest import DatasetManifest

    dataset_dir = args.output_dir or _DATASET_DIR
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"No manifest found at {manifest_path}. Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    manifest = DatasetManifest.load(manifest_path)

    # Check leakage
    leakage = manifest.check_leakage()
    if leakage:
        print("LEAKAGE DETECTED:")
        for v in leakage:
            print(f"  {v}")
    else:
        print("No leakage detected.")

    # Verify checksums
    results = manifest.verify_checksums(dataset_dir)
    all_ok = True
    for filename, ok in results.items():
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {filename}: {status}")

    if all_ok:
        print("\nAll checksums verified.")
    else:
        print("\nSome checksums failed!", file=sys.stderr)
        sys.exit(1)


def cmd_plan_ingestions(args: argparse.Namespace) -> None:
    """Analyze YAML experiments and display ingestion signature grouping."""
    import glob
    import json
    import yaml

    experiments_dir = args.dir or _EXPERIMENTS_DIR
    yaml_files = sorted(glob.glob(os.path.join(experiments_dir, "*.yaml")))
    if not yaml_files:
        print(f"No YAML files found in {experiments_dir}")
        return

    from ..experiment import _INGESTION_TIME_KEYS, _CORPUS_ID_MAP

    # Parse each YAML and compute ingestion signatures
    signatures: dict = {}  # sig → list of (yaml_file, config_name)
    for yf in yaml_files:
        try:
            with open(yf) as f:
                spec = yaml.safe_load(f)
        except Exception:
            continue

        benchmarks = spec.get("benchmarks", [])
        overrides = spec.get("pipeline", {}).get("config_overrides", [])
        base_profile = spec.get("pipeline", {}).get("base_profile", "standard")

        for bench in benchmarks:
            bench_name = bench.get("name", "")
            corpus_id = _CORPUS_ID_MAP.get(bench_name, bench_name)

            for cfg in overrides:
                cfg_name = cfg.get("name", "unnamed")
                profile = cfg.get("profile", base_profile)
                ingestion_keys = {
                    k: v for k, v in cfg.items() if k in _INGESTION_TIME_KEYS and k != "name"
                }
                sig_key = f"{corpus_id}|{profile}|{json.dumps(ingestion_keys, sort_keys=True)}"

                if sig_key not in signatures:
                    signatures[sig_key] = {
                        "corpus_id": corpus_id,
                        "profile": profile,
                        "ingestion_overrides": ingestion_keys,
                        "consumers": [],
                    }
                signatures[sig_key]["consumers"].append((os.path.basename(yf), cfg_name))

    print(f"Found {len(yaml_files)} YAML files")
    print(f"Unique ingestion signatures: {len(signatures)}")
    print()
    for i, (sig, info) in enumerate(sorted(signatures.items()), 1):
        print(f"  [{i}] corpus={info['corpus_id']}, profile={info['profile']}")
        if info["ingestion_overrides"]:
            print(f"      overrides: {info['ingestion_overrides']}")
        print(f"      consumers ({len(info['consumers'])}):")
        for yf, cfg in info["consumers"][:5]:
            print(f"        {yf} → {cfg}")
        if len(info["consumers"]) > 5:
            print(f"        ... and {len(info['consumers']) - 5} more")
        print()


def cmd_run_paper(args: argparse.Namespace) -> None:
    """Run experiment phases with structured output directory."""
    import shutil

    output_dir = args.output_dir
    phases = [p.strip().upper() for p in args.phases.split(",")]
    experiments_dir = args.dir or _EXPERIMENTS_DIR

    os.makedirs(output_dir, exist_ok=True)
    for subdir in ["corpora", "experiments", "paper/tables", "paper/figures", "paper/data"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Phases: {phases}")
    print(f"Experiments directory: {experiments_dir}")
    print()

    # Copy corpus files to output
    dataset_dir = _DATASET_DIR
    corpus_dir = os.path.join(dataset_dir, "corpus")
    if os.path.exists(corpus_dir):
        dest = os.path.join(output_dir, "corpora")
        for f in os.listdir(corpus_dir):
            src = os.path.join(corpus_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, f))
        print(f"Copied corpus files to {dest}")

    # Write provenance
    import platform

    provenance = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "phases": phases,
        "experiments_dir": experiments_dir,
        "dataset_dir": dataset_dir,
    }
    try:
        import subprocess

        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        provenance["git_commit"] = git_sha
    except Exception:
        provenance["git_commit"] = "unknown"

    import json as _json

    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        _json.dump(provenance, f, indent=2)
    print(f"Provenance written to {os.path.join(output_dir, 'provenance.json')}")
    print()
    print("To run experiments, use run-suite with phase-specific YAML files:")
    print(f"  python -m ...private_eval run-suite --dir {experiments_dir}")


def cmd_register(args: argparse.Namespace) -> None:
    """Register private eval adapters in benchmark registry."""
    from ..benchmarks import benchmark_registry
    from .adapter import register_private_eval_adapters

    dataset_dir = args.output_dir or _DATASET_DIR
    register_private_eval_adapters(benchmark_registry, dataset_dir)
    print(
        f"Registered private_eval/retrieval, private_eval/e2e_generation, "
        f"private_eval/hallucination (+ deprecated generation redirect) "
        f"(dataset_dir={dataset_dir})"
    )


def cmd_run_suite(args: argparse.Namespace) -> None:
    """Run all YAML experiments concurrently."""
    from .suite_runner import SuiteRunner

    experiments_dir = args.dir or _EXPERIMENTS_DIR

    # Optionally filter to explicit YAML files passed as positional args
    experiment_files = args.experiments or None

    runner = SuiteRunner(
        experiment_files=experiment_files,
        experiments_dir=experiments_dir,
        max_parallel_experiments=args.concurrency,
        skip_existing=not args.force,
    )

    def _progress(name: str, status: str, elapsed: float) -> None:
        if status == "started":
            print(f"  [start] {name}")
        elif status == "done":
            print(f"  [ ok  ] {name} ({elapsed:.1f}s)")
        else:
            print(f"  [FAIL ] {name} ({elapsed:.1f}s)", file=sys.stderr)

    print(f"Running experiments from: {experiments_dir}")
    summary = asyncio.run(runner.run_all(progress_callback=_progress))

    ok = sum(1 for r in summary.values() if r["status"] == "ok")
    failed = sum(1 for r in summary.values() if r["status"] == "failed")
    print(f"\nSuite done: {ok} ok, {failed} failed")
    if failed:
        sys.exit(1)


def cmd_supervise(args: argparse.Namespace) -> None:
    """Monitor running or completed experiments in real time."""
    from .supervisor import ExperimentSupervisor

    results_dir = args.results_dir or os.path.join(
        os.path.expanduser("~"), ".lasagna", "research", "results"
    )
    sup = ExperimentSupervisor(
        results_dir=results_dir,
        experiments_dir=args.experiments_dir or _EXPERIMENTS_DIR,
        ir_metric=args.ir_metric,
        min_quality_threshold=args.min_quality,
        max_latency_s=args.max_latency,
        max_tokens=args.max_tokens,
    )
    if args.watch:
        sup.watch(interval_s=args.interval)
    else:
        sup.print_status()


def cmd_report(args: argparse.Namespace) -> None:
    """Generate master cross-experiment report."""
    from .master_report import MasterReportGenerator

    results_dir = args.results_dir or os.path.join(
        os.path.expanduser("~"), ".lasagna", "research", "results"
    )
    output_path = args.output or os.path.join(results_dir, "master_report.md")

    print(f"Loading results from: {results_dir}")
    gen = MasterReportGenerator(
        results_dir=results_dir,
        ir_metric=args.ir_metric,
    ).load()

    gen.save(output_path)
    print(f"Master report saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="private_eval",
        description="Private evaluation dataset and experiment runner for RAG optimization",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help=f"Dataset output directory (default: {_DATASET_DIR})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build
    build_parser = subparsers.add_parser("build", help="Build the dataset")
    build_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    build_parser.add_argument(
        "--distractors",
        type=int,
        default=500,
        help="Distractor documents per benchmark (default: 500)",
    )

    # stats
    subparsers.add_parser("stats", help="Show dataset statistics")

    # verify
    subparsers.add_parser("verify", help="Verify dataset integrity")

    # register
    subparsers.add_parser("register", help="Register adapters in benchmark registry")

    # run-suite
    suite_parser = subparsers.add_parser(
        "run-suite",
        help="Run all YAML experiments concurrently (fastest way to evaluate all configs)",
    )
    suite_parser.add_argument(
        "--dir",
        default=None,
        help=f"Directory containing experiment YAML files (default: {_EXPERIMENTS_DIR})",
    )
    suite_parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Max parallel experiments (default: 0 = unlimited)",
    )
    suite_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even if results.json already exists",
    )
    suite_parser.add_argument(
        "experiments",
        nargs="*",
        help="Optional explicit YAML file paths to run (overrides --dir)",
    )

    # supervise
    sup_parser = subparsers.add_parser(
        "supervise",
        help="Monitor running/completed experiments: progress, metrics vs baseline, anomalies, ETA",
    )
    sup_parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh status table every --interval seconds until all experiments complete",
    )
    sup_parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval in seconds for --watch mode (default: 30)",
    )
    sup_parser.add_argument(
        "--results-dir",
        default=None,
        help="Experiment results root directory (default: ~/.lasagna/research/results/)",
    )
    sup_parser.add_argument(
        "--experiments-dir",
        default=None,
        help=f"YAML experiments directory (default: {_EXPERIMENTS_DIR})",
    )
    sup_parser.add_argument(
        "--ir-metric",
        default="ndcg@10",
        help="Primary IR metric for quality comparisons (default: ndcg@10)",
    )
    sup_parser.add_argument(
        "--min-quality",
        type=float,
        default=0.30,
        help="Anomaly threshold: configs below this quality score trigger a warning (default: 0.30)",
    )
    sup_parser.add_argument(
        "--max-latency",
        type=float,
        default=60.0,
        help="Anomaly threshold: avg per-query latency above this (seconds) triggers a warning (default: 60)",
    )
    sup_parser.add_argument(
        "--max-tokens",
        type=float,
        default=20_000_000.0,
        help=(
            "Spend budget in tokens — triggers a warning when exceeded "
            "(default: 20,000,000). Was --max-cost in USD; prices are not a "
            "property of a run, and the dollar figure it compared against was "
            "always $0.00 because nothing wrote it."
        ),
    )

    # report
    report_parser = subparsers.add_parser(
        "report",
        help="Generate master cross-experiment report from completed results",
    )
    report_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing experiment result folders (default: ~/.lasagna/research/results/)",
    )
    report_parser.add_argument(
        "--output",
        default=None,
        help="Output Markdown file path (default: <results-dir>/master_report.md)",
    )
    report_parser.add_argument(
        "--ir-metric",
        default="ndcg@10",
        help="Primary IR metric for quality comparisons (default: ndcg@10)",
    )

    # plan-ingestions
    plan_parser = subparsers.add_parser(
        "plan-ingestions",
        help="Analyze YAML experiments and show ingestion signature grouping",
    )
    plan_parser.add_argument(
        "--dir",
        default=None,
        help=f"Directory with experiment YAML files (default: {_EXPERIMENTS_DIR})",
    )

    # run-paper
    paper_parser = subparsers.add_parser(
        "run-paper",
        help="Run all experiment phases with structured output for paper artifacts",
    )
    paper_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for structured results (e.g., ~/.lasagna/research/paper/)",
    )
    paper_parser.add_argument(
        "--phases",
        default="A,B,C,D,E",
        help="Comma-separated phases to run (default: A,B,C,D,E)",
    )
    paper_parser.add_argument(
        "--dir",
        default=None,
        help=f"YAML experiments directory (default: {_EXPERIMENTS_DIR})",
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "register":
        cmd_register(args)
    elif args.command == "run-suite":
        cmd_run_suite(args)
    elif args.command == "supervise":
        cmd_supervise(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "plan-ingestions":
        cmd_plan_ingestions(args)
    elif args.command == "run-paper":
        cmd_run_paper(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
