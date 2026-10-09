"""Stratified dataset splitter — no sklearn dependency.

Uses deterministic hash-based assignment to create reproducible
train/eval/test splits with stratification by (domain, difficulty).
Rare strata are merged to avoid splits with too few samples.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


class StratifiedSplitter:
    """Creates reproducible train/eval/test splits with stratification."""

    def split(
        self,
        samples: List[Dict[str, Any]],
        ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        stratify_by: Sequence[str] = ("domain", "difficulty"),
        seed: int = 42,
        min_stratum_size: int = 5,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Stratified split with rare-stratum merging.

        Args:
            samples: List of sample dicts (must have 'sample_id' key).
            ratios: (train, eval, test) ratios summing to 1.0.
            stratify_by: Keys to stratify on.
            seed: Random seed for deterministic hashing.
            min_stratum_size: Strata smaller than this are merged.

        Returns:
            (train, eval, test) lists of sample dicts.
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

        # Group samples by stratum key
        strata: Dict[str, List[Dict]] = defaultdict(list)
        for sample in samples:
            key = _stratum_key(sample, stratify_by)
            strata[key].append(sample)

        # Merge rare strata into nearest bucket
        strata = _merge_rare_strata(strata, min_stratum_size)

        # Split each stratum independently
        train, eval_, test = [], [], []
        for _key, stratum_samples in strata.items():
            t, e, ts = _split_stratum(stratum_samples, ratios, seed)
            train.extend(t)
            eval_.extend(e)
            test.extend(ts)

        return train, eval_, test


def _stratum_key(sample: Dict[str, Any], stratify_by: Sequence[str]) -> str:
    """Create a stratum key from sample fields."""
    parts = []
    for key in stratify_by:
        val = sample.get(key, "unknown")
        parts.append(f"{key}={val}")
    return "|".join(parts)


def _merge_rare_strata(strata: Dict[str, List[Dict]], min_size: int) -> Dict[str, List[Dict]]:
    """Merge strata smaller than min_size into a catch-all bucket.

    Groups rare strata by domain (first stratification key) when possible,
    falling back to a global '_merged' bucket.
    """
    merged: Dict[str, List[Dict]] = {}
    rare_by_domain: Dict[str, List[Dict]] = defaultdict(list)

    for key, samples in strata.items():
        if len(samples) >= min_size:
            merged[key] = samples
        else:
            # Extract domain from key like "domain=biomedical|difficulty=easy"
            domain = "unknown"
            for part in key.split("|"):
                if part.startswith("domain="):
                    domain = part.split("=", 1)[1]
                    break
            rare_by_domain[domain].extend(samples)

    # Add merged rare strata
    for domain, samples in rare_by_domain.items():
        merged_key = f"domain={domain}|difficulty=_merged"
        if merged_key in merged:
            merged[merged_key].extend(samples)
        else:
            merged[merged_key] = samples

    return merged


def _split_stratum(
    samples: List[Dict],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split a single stratum using deterministic hashing.

    Each sample is assigned to train/eval/test based on:
      hash(sample_id + str(seed)) mod 1000
    mapped to cumulative ratio buckets.

    Includes a minimum-per-split guard: if a stratum is too small for
    every split to receive at least one sample, samples are distributed
    round-robin by hash order to ensure no split is empty.
    """
    train_bound = int(ratios[0] * 1000)
    eval_bound = train_bound + int(ratios[1] * 1000)

    train, eval_, test = [], [], []
    for sample in samples:
        sample_id = sample.get("sample_id", "")
        h = hashlib.sha256(f"{sample_id}:{seed}".encode()).hexdigest()
        bucket = int(h[:8], 16) % 1000

        if bucket < train_bound:
            train.append(sample)
        elif bucket < eval_bound:
            eval_.append(sample)
        else:
            test.append(sample)

    # Post-hoc guarantee: if any non-zero-ratio split ended up empty,
    # move one sample from the largest split.  This handles small strata
    # (e.g. 5 samples with 60/20/20 ratios) where hash bucketing can
    # leave eval or test empty.
    splits = [train, eval_, test]
    for i, ratio in enumerate(ratios):
        if ratio > 0 and len(splits[i]) == 0:
            donor = max(range(3), key=lambda j: len(splits[j]))
            if len(splits[donor]) > 1:
                splits[i].append(splits[donor].pop())

    return train, eval_, test
