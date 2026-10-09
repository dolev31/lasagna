"""Dataset manifest for reproducibility tracking.

Tracks dataset composition, version, checksums, and sample IDs per split
to enable leakage auditing and reproducible builds.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass
class BenchmarkSourceSpec:
    """Records how a benchmark was sampled."""

    name: str
    split: str
    n_samples: int
    track: str  # "retrieval", "e2e_generation", or "hallucination"
    domain: str = ""


@dataclass
class SplitStats:
    """Statistics for a single split."""

    count: int
    domains: Dict[str, int] = field(default_factory=dict)
    difficulties: Dict[str, int] = field(default_factory=dict)
    query_types: Dict[str, int] = field(default_factory=dict)
    sample_ids: List[str] = field(default_factory=list)


@dataclass
class CorpusStats:
    """Statistics for the document corpus."""

    doc_count: int = 0
    total_chars: int = 0
    benchmarks: Dict[str, int] = field(default_factory=dict)  # benchmark -> doc count
    distractor_count: int = 0


@dataclass
class DatasetManifest:
    """Complete manifest for the private evaluation dataset."""

    version: str = "1.0.0"
    created_at: str = ""
    seed: int = 42
    benchmarks: List[BenchmarkSourceSpec] = field(default_factory=list)
    splits: Dict[str, SplitStats] = field(default_factory=dict)
    corpus: CorpusStats = field(default_factory=CorpusStats)
    checksums: Dict[str, str] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save manifest to JSON."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        data = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> DatasetManifest:
        """Load manifest from JSON."""
        with open(path) as f:
            data = json.load(f)
        manifest = cls(
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", ""),
            seed=data.get("seed", 42),
            checksums=data.get("checksums", {}),
        )
        for b in data.get("benchmarks", []):
            manifest.benchmarks.append(BenchmarkSourceSpec(**b))
        for split_name, stats in data.get("splits", {}).items():
            manifest.splits[split_name] = SplitStats(**stats)
        corpus_data = data.get("corpus", {})
        manifest.corpus = CorpusStats(**corpus_data)
        return manifest

    def verify_checksums(self, dataset_dir: str) -> Dict[str, bool]:
        """Verify file checksums against stored values."""
        results = {}
        for filename, expected_hash in self.checksums.items():
            filepath = os.path.join(dataset_dir, filename)
            if not os.path.exists(filepath):
                results[filename] = False
                continue
            actual_hash = _compute_file_hash(filepath)
            results[filename] = actual_hash == expected_hash
        return results

    def check_leakage(self) -> List[str]:
        """Check for sample ID overlap between splits within the same track.

        Cross-track overlap (e.g., e2e_generation vs hallucination sharing
        the same queries) is expected and intentional — not flagged.
        """
        violations = []
        split_names = list(self.splits.keys())
        for i, name_a in enumerate(split_names):
            track_a = name_a.rsplit("_", 1)[0]  # e.g., "retrieval" from "retrieval_train"
            ids_a = set(self.splits[name_a].sample_ids)
            for name_b in split_names[i + 1 :]:
                track_b = name_b.rsplit("_", 1)[0]
                # Only check within the same track (e.g., retrieval_train vs retrieval_eval)
                if track_a != track_b:
                    continue
                ids_b = set(self.splits[name_b].sample_ids)
                overlap = ids_a & ids_b
                if overlap:
                    violations.append(
                        f"Leakage: {len(overlap)} samples shared between "
                        f"{name_a} and {name_b}: {sorted(overlap)[:5]}..."
                    )
        return violations

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Dataset Manifest v{self.version}",
            f"Created: {self.created_at}",
            f"Seed: {self.seed}",
            f"Benchmarks: {len(self.benchmarks)}",
            "",
        ]
        for split_name, stats in sorted(self.splits.items()):
            lines.append(f"  {split_name}: {stats.count} samples")
            if stats.domains:
                domain_str = ", ".join(f"{k}={v}" for k, v in sorted(stats.domains.items()))
                lines.append(f"    domains: {domain_str}")
            if stats.difficulties:
                diff_str = ", ".join(f"{k}={v}" for k, v in sorted(stats.difficulties.items()))
                lines.append(f"    difficulties: {diff_str}")
        if self.corpus.doc_count:
            lines.append(
                f"\nCorpus: {self.corpus.doc_count} docs, "
                f"{self.corpus.total_chars:,} chars, "
                f"{self.corpus.distractor_count} distractors"
            )
        leakage = self.check_leakage()
        if leakage:
            lines.append(f"\nWARNING: {len(leakage)} leakage violations found!")
            for v in leakage:
                lines.append(f"  {v}")
        else:
            lines.append("\nNo leakage detected.")
        return "\n".join(lines)


def _compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def compute_jsonl_hash(filepath: str) -> str:
    """Compute hash for a JSONL file (used in manifest checksums)."""
    return _compute_file_hash(filepath)
