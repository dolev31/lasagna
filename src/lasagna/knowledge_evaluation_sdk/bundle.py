"""The results bundle, and the key that decides what may be compared to what.

Publishing numbers to a shared leaderboard has one hard problem, and it is not
storage. It is that two rows are only comparable when everything except the
thing under test was held fixed — and until now nothing recorded what was held
fixed, so `compare()` unioned two result sets and printed them side by side
whatever they were measuring.

The comparability key is a hash over the held-fixed configuration:

    metrics_version, benchmark, split, n_samples, sampling seed,
    sample_id_digest, corpus_fingerprint, corpus_provenance, resolved_mode,
    embedder + dimension, chunker, retriever top_k, vector backend, judge model

It **excludes the dimension under test**, which is recorded separately as
``varied_dimensions``. That exclusion is the whole design: the point of the
comparison is that one thing differs, so a key covering everything would make
every row incomparable with every other, and a key covering nothing would make
all of them comparable regardless of what they measured.

``metrics_version`` is in the key deliberately. If the definition of ndcg@10
changes, old numbers must fall out of comparison rather than be silently mixed
with new ones — the quiet failure that nothing else in this design catches.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

#: Bump when a metric definition changes. Numbers computed under different
#: values are not comparable and must not share a leaderboard group.
#:
#: 1.1 — two changes, both of which move NDCG:
#:   * NDCG gain is linear (`rel`), matching `trec_eval` and therefore BEIR.
#:     It was `2^rel - 1`, identical on binary judgements and different on the
#:     graded ones (nfcorpus, trec-covid, dbpedia-entity).
#:   * The IR metrics are computed over `k` *documents*. The harness requested
#:     `k` chunks and then de-duplicated to parent documents, so a "top-10" list
#:     was 4-7 documents and every cutoff metric was computed over a list
#:     shorter than its own name claimed.
#: Numbers produced before this are not comparable with numbers produced after,
#: which is exactly what this field exists to enforce.
#:
#: 1.2 — the held-fixed block described the machine, not the run.
#:   `_system_under_test` read the process-global `rag_config`, which no run
#:   uses: every override path in the harness is session-scoped and
#:   `from_profile` stores the profile on the pipeline instance. A `speed`-profile
#:   run (all-MiniLM-L6-v2 @ 384) therefore published
#:   `embedder_model = BAAI/bge-m3, embedder_dim = 1024` — whatever the
#:   operator's settings file happened to say. Five of the fourteen key fields
#:   come from that block, so two runs on *different embedding models* hashed
#:   identically, passed `results validate`, and were ranked in one leaderboard
#:   table. Keys minted before 1.2 cannot be trusted to mean what they say, so
#:   they must not compare equal to keys minted after.
METRICS_VERSION = "1.2"

#: Bundle schema version. Consumers accept >=1.0,<2.0.
SCHEMA_VERSION = "1.0"

#: Fields whose agreement is required before two results may be compared.
#: Order is irrelevant (the payload is sorted); membership is not.
HELD_FIXED_FIELDS = (
    "metrics_version",
    "benchmark",
    "split",
    "n_samples",
    "sampling_seed",
    "sample_id_digest",
    "corpus_fingerprint",
    # "native" vs "pooled_gold". In the key, not beside it: a corpus pooled
    # from the samples' own gold context holds no hard negatives, so recall
    # against it reads higher than against a real corpus of the same size.
    # Documentation alone would not stop the two appearing in one leaderboard
    # row; a different comparability key does.
    "corpus_provenance",
    "resolved_mode",
    "embedder_model",
    "embedder_dim",
    "chunker",
    "retriever_top_k",
    "vector_backend",
    "judge_model",
)

#: The stages a sweep varies. Excluded from the key, recorded alongside it.
VARIABLE_DIMENSIONS = (
    "reranker",
    "embedder",
    "chunker",
    "query_enhancer",
    "retriever",
    "agentic_pipeline",
)

#: Override keys that are machinery, not a research dimension.
#:
#: A denylist rather than a second allowlist beside ``VARIABLE_DIMENSIONS``.
#: Sweeps come in two shapes: component swaps (``reranker: bge``) and config
#: flags (``enable_reranking: true``, ``hybrid_alpha: 0.3``), and the flags are
#: open-ended. An allowlist would silently report "varied nothing" for any knob
#: nobody thought to add to it — which is exactly the failure being fixed here,
#: where a factorial arm and its baseline both rendered ``_baseline_``. Hiding a
#: dimension that *was* varied is the expensive direction of this error.
_DERIVED_OVERRIDE_KEYS = frozenset(
    {
        # Auto-derived from embedding_model so each model gets its own
        # collection; nobody chose it and it varies as a consequence, not as a
        # treatment.
        "collection_name",
        # Consumed by _generate_pipeline_configs itself.
        "name",
        "profile",
        # Steers the harness, never the pipeline (experiment._HARNESS_KEYS).
        "allow_inert_overrides",
        "save_trajectories",
    }
)


#: Values assumed when a field is absent, so that adding a field to the key does
#: not orphan every bundle published before it existed.
#:
#: A default belongs here only when absence *proves* the value, never when it
#: merely suggests one. ``corpus_provenance`` qualifies: pooled-gold corpora did
#: not exist before the field did, so any bundle without it was necessarily
#: measured against a native corpus. Guessing ``embedder_model`` from absence
#: would be the opposite -- unknowable -- which is why the rule below still
#: records everything else as ``None``.
#:
#: Without this, adding ``corpus_provenance`` silently rewrote the key for every
#: run: ``lasagna results validate`` reported honest pre-release bundles as
#: INVALID with the message hub/README documents as *"held_fixed was edited to
#: fake comparability"*, and the leaderboard split two commensurable runs into
#: two tables that each looked complete.
KEY_DEFAULTS: Dict[str, Any] = {
    "corpus_provenance": "native",
}


def comparability_key(held_fixed: Dict[str, Any]) -> str:
    """Stable hash of the configuration held fixed for a comparison.

    Missing fields are recorded as ``None`` rather than omitted, so a bundle
    that failed to capture a field produces a *different* key from one that
    captured it — silently dropping it would let an unknown embedding model
    compare equal to a known one. The exceptions are in :data:`KEY_DEFAULTS`,
    where absence is proof of a value rather than a gap.
    """
    # Absent and explicitly-null must resolve the same way.
    #
    # This was `held_fixed.get(field, KEY_DEFAULTS.get(field))`, and
    # `dict.get(k, default)` returns None for a key that is *present* with a
    # None value -- so the default only applied when the key was missing
    # entirely. `held_fixed_from` always sets every field, to None when the
    # result row lacks it, so the one path that matters never got the default
    # and the same native run produced three different keys depending on how it
    # was serialised.
    payload = {}
    for field in HELD_FIXED_FIELDS:
        value = held_fixed.get(field)
        payload[field] = KEY_DEFAULTS[field] if value is None and field in KEY_DEFAULTS else value

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def held_fixed_from(result: Dict[str, Any], run_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the held-fixed block from a task result and its run metadata."""
    system = run_meta.get("system", {}) or {}
    return {
        "metrics_version": METRICS_VERSION,
        "benchmark": result.get("benchmark_name"),
        "split": result.get("split"),
        "n_samples": result.get("n_samples"),
        "sampling_seed": result.get("sampling_seed"),
        "sample_id_digest": result.get("sample_id_digest"),
        "corpus_fingerprint": result.get("corpus_fingerprint"),
        "corpus_provenance": result.get("corpus_provenance"),
        "resolved_mode": result.get("resolved_mode"),
        "embedder_model": system.get("embedder_model"),
        "embedder_dim": system.get("embedder_dim"),
        "chunker": system.get("chunker"),
        "retriever_top_k": system.get("retriever_top_k"),
        "vector_backend": system.get("vector_backend"),
        "judge_model": run_meta.get("judge_model"),
    }


def varied_dimensions(
    config_overrides: Dict[str, Any],
    across: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """What this config changed, so a reader knows what is under test.

    Component slots and config flags both count: a sweep over ``reranker`` and
    a sweep over ``enable_reranking`` are the same experiment expressed two
    ways, and a reader of the leaderboard needs to see either one.

    ``across`` is every config's overrides in the same run. When given, a key
    is only "varied" if it actually differs somewhere in the run. Without it,
    a factorial's ``held_fixed`` settings — which are applied to all four cells
    precisely so they *cannot* vary — were reported as varied by every cell, so
    all four arms of a 2x2 rendered an identical Varied column and the design
    became unreadable in the one table meant to display it.
    """
    candidates = {k for k in config_overrides if k not in _DERIVED_OVERRIDE_KEYS}
    # One arm has nothing to differ from, so "what varies across the run" is
    # vacuously empty and would hide the override the run exists to report.
    # Fall back to what this config applied.
    if across is None or len(across) < 2:
        return sorted(candidates)

    def _values(key: str) -> set:
        return {json.dumps(cfg.get(key), sort_keys=True, default=str) for cfg in across}

    return sorted(k for k in candidates if len(_values(k)) > 1)


def varied_levels(
    config_overrides: Dict[str, Any],
    across: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The values this config set for the dimensions that vary.

    Names alone cannot tell a 2x2's four cells apart — every cell varies the
    same two flags — so the levels are what makes the design readable.
    """
    return {k: config_overrides.get(k) for k in varied_dimensions(config_overrides, across)}


def compare_keys(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Field-level differences between two held-fixed blocks.

    Returned rather than raised. The first thing a researcher does is compare
    their run against a published row that differs somewhere, and a bare
    refusal teaches them nothing about which knob to turn.
    """
    return [
        f"{field}: {a.get(field)!r} vs {b.get(field)!r}"
        for field in HELD_FIXED_FIELDS
        if a.get(field) != b.get(field)
    ]


def build_bundle(
    results: Any,
    *,
    submitter: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble a self-contained, publishable record of one run.

    Everything a reader needs to decide whether these numbers may be compared
    with theirs, without access to the machine that produced them.
    """
    run_meta = dict(getattr(results, "experiment_config", {}) or {})
    rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results._results]

    # Every distinct config's overrides, so "varied" can mean "differs somewhere
    # in this run" rather than "present in this config" -- see varied_dimensions.
    seen_names: List[str] = []
    overrides_by_name: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = row.get("config_name", "")
        if name not in overrides_by_name:
            seen_names.append(name)
            overrides_by_name[name] = row.get("config_overrides") or {}
    across = [overrides_by_name[n] for n in seen_names]

    configs: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = row.get("config_name", "")
        if name in configs:
            continue
        held = held_fixed_from(row, run_meta)
        own = overrides_by_name.get(name, {})
        configs[name] = {
            "name": name,
            "comparability_key": comparability_key(held),
            "held_fixed": held,
            # Derived from the overrides this config actually applied. An empty
            # list means the config varied nothing -- a baseline -- which reads
            # the same as "nobody recorded it" unless it is computed here.
            "varied_dimensions": varied_dimensions(own, across),
            # The levels, so a 2x2's four cells are distinguishable: they all
            # vary the same two flags and differ only in the values.
            "varied_levels": varied_levels(own, across),
        }

    integrity = _integrity(rows, run_meta)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "metrics_version": METRICS_VERSION,
        "experiment": run_meta,
        "submitter": submitter or {},
        "configs": list(configs.values()),
        "results": rows,
        "integrity": integrity,
    }
    # The id covers the content, so an edited bundle no longer matches its id.
    bundle["run_id"] = _content_hash(bundle)
    return bundle


def json_schema() -> Dict[str, Any]:
    """The published JSON Schema for a bundle, built from the constants above.

    Generated rather than checked in by hand for one reason: the required
    ``held_fixed`` fields are ``HELD_FIXED_FIELDS``, and a schema listing a
    different set would reject bundles this package produces — or worse, accept
    ones missing a field the comparability key depends on. Hand-maintaining two
    copies of that tuple guarantees they diverge.

    `lasagna results schema` writes it; a test asserts the checked-in copy
    matches.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://dolev31.github.io/lasagna/schema/run-{SCHEMA_VERSION}.json",
        "title": "Lasagna results bundle",
        "description": (
            "One evaluation run, self-contained. Carries enough provenance for a "
            "reader to decide whether these numbers may be compared with theirs."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "metrics_version",
            "run_id",
            "experiment",
            "configs",
            "results",
            "integrity",
        ],
        "properties": {
            # Major version only: 1.1 adds fields a 1.0 reader can ignore.
            "schema_version": {"type": "string", "pattern": r"^1\."},
            "metrics_version": {
                "type": "string",
                "description": (
                    "Metric definitions. Numbers computed under different values "
                    "are not comparable."
                ),
            },
            "run_id": {
                "type": "string",
                "description": (
                    "Hash of the bundle contents. Recomputed on validation, so an "
                    "edited number no longer matches."
                ),
            },
            "submitter": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "affiliation": {"type": "string"},
                },
            },
            "experiment": {
                "type": "object",
                "required": ["environment"],
                "properties": {
                    "environment": {
                        "type": "object",
                        "required": ["lasagna_version", "python", "platform"],
                        "description": (
                            "No paths, hostnames or credential names: bundles are published."
                        ),
                    },
                    "variance_source": {"enum": ["run", "sample"]},
                    "determinism": {
                        "const": "seeded",
                        "description": (
                            "Never 'reproducible': LLM judges and GPU kernels defeat that claim."
                        ),
                    },
                },
            },
            "configs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "comparability_key", "held_fixed"],
                    "properties": {
                        "comparability_key": {
                            "type": "string",
                            "description": "Hash of held_fixed. Recomputed on validation.",
                        },
                        "held_fixed": {
                            "type": "object",
                            # Fields with a documented default in KEY_DEFAULTS
                            # are optional: a bundle published before the field
                            # existed is still valid, and the key computes the
                            # same value for it either way. Everything else is
                            # required, because a key hashing a field the schema
                            # does not demand is a key nobody can recompute.
                            "required": [f for f in HELD_FIXED_FIELDS if f not in KEY_DEFAULTS],
                        },
                        "varied_levels": {
                            "type": "object",
                            "description": (
                                "The value each varied dimension took, so the "
                                "cells of a factorial are distinguishable."
                            ),
                        },
                        "varied_dimensions": {
                            "type": "array",
                            # No enum. `varied_dimensions()` reports component
                            # slots AND config flags, and the flags are
                            # open-ended -- a 2x2 varies `enable_hyde`, which no
                            # fixed list would contain. The enum here was
                            # `VARIABLE_DIMENSIONS`, the six component slots, so
                            # the published schema rejected every bundle a
                            # factorial or flag sweep produces:
                            # `'enable_hyde' is not one of [...]`.
                            "items": {"type": "string"},
                            "description": (
                                "What this config changed: component slots "
                                "(`reranker`) or config flags (`enable_hyde`). "
                                "Excluded from the comparability key by design."
                            ),
                        },
                    },
                },
            },
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["benchmark_name", "config_name", "metrics"],
                },
            },
            "integrity": {
                "type": "object",
                "required": ["warnings", "blocking"],
                "properties": {
                    "warnings": {"type": "array", "items": {"type": "string"}},
                    "blocking": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def _integrity(rows: List[Dict[str, Any]], run_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Machine-readable reasons a bundle should not be trusted or published.

    Codes rather than prose so a validator can reject on them without parsing
    English, and so the list is the same on every run.

    **warnings vs blocking.** A warning says "read this before trusting the
    number". A blocking code says "this number cannot mean what it claims", and
    `lasagna results validate` refuses it outright. ``blocking`` used to be
    initialised and never appended to, so every bundle passed and the hub's
    "the validator is the product" was an overstatement.

    The bar for blocking is deliberately narrow: the bundle is *internally
    incoherent*, or it asserts something the recorded data cannot support.
    Anything that is merely a weak experiment stays a warning -- a pilot run is
    allowed to be a pilot, it just may not be published as more than one.
    """
    warnings: List[str] = []
    blocking: List[str] = []

    if any(r.get("error") for r in rows):
        warnings.append("FAILED_TASKS")

    # A number computed against part of a corpus is not the benchmark's number,
    # and the truncation is invisible downstream -- retrieval metrics simply
    # come out lower. Corpus size is not in the comparability key either, so a
    # truncated run and a full one would otherwise group together.
    if any(r.get("corpus_truncated") for r in rows):
        blocking.append("TRUNCATED_CORPUS")

    # Mutable upstream. Without a pinned revision, "same benchmark" is an
    # assumption about someone else's repository rather than a fact about this
    # run, so cross-submitter comparison is not supported by the record.
    #
    # A *warning*, not blocking. Blocking it outright made every live run
    # unpublishable the moment the code landed, since nothing pins revisions
    # yet -- a rule nobody can satisfy is not a standard, it is an outage.
    # `--strict` promotes it, which is what the hub runs, so publishing still
    # requires a pin while local work is unaffected. `pin_dataset_revision` and
    # `LASAGNA_DATASET_REVISIONS` are how it gets satisfied.
    if any(r.get("resolved_mode") == "live" and not r.get("hf_revision") for r in rows):
        warnings.append("UNPINNED_DATASET")

    # Retrieval measured against a corpus pooled from the samples' own gold
    # context. A real result, on an easier task: the pool holds no hard
    # negatives, so recall reads higher than it would against a real corpus.
    # `corpus_provenance` is in the comparability key, so these can never be
    # grouped with native-corpus rows -- this warning is for the reader who
    # sees the number on its own.
    if any(r.get("corpus_provenance") == "pooled_gold" for r in rows):
        warnings.append("POOLED_GOLD_CORPUS")

    # There was an UNDERPOWERED check here reading `row["underpowered"]`, and
    # nothing anywhere wrote that key -- the read-with-no-producer bug this
    # whole release existed to fix, reintroduced by the release itself. It could
    # never fire.
    #
    # It is removed rather than wired up, because power is not a property of a
    # bundle. The minimum detectable effect is a property of a *comparison*: it
    # needs two configs and a designated baseline, and a bundle is a set of rows
    # with no baseline marked. Computing one here would mean picking a baseline
    # by guesswork and stamping a statistical claim on the result.
    #
    # `lasagna results power <run.json> <baseline> <candidate>` is the mechanism,
    # and it asks for the two configs precisely because they cannot be inferred.

    # The statistics need at least two runs to say anything about spread. A
    # single run carrying a significance claim is asserting more than it
    # measured.
    n_runs = run_meta.get("n_runs", 1)
    if n_runs < 2 and any(r.get("significance") for r in rows):
        blocking.append("SIGNIFICANCE_WITHOUT_REPLICATION")

    # An empty bundle validates trivially otherwise.
    if not rows:
        blocking.append("NO_RESULTS")

    if not any(r.get("corpus_fingerprint") for r in rows):
        warnings.append("NO_CORPUS_FINGERPRINT")

    if any(r.get("split") and r.get("split") != "test" for r in rows):
        warnings.append("NON_TEST_SPLIT")

    if run_meta.get("n_runs", 1) < 2:
        # Not blocking on its own: a single run is a legitimate pilot. It only
        # becomes a problem when a significance claim is attached to it.
        warnings.append("SINGLE_RUN")

    judge = run_meta.get("judge_model")
    if judge and _is_floating_alias(judge):
        warnings.append("UNRESOLVED_JUDGE_ALIAS")

    if run_meta.get("determinism") != "seeded":
        warnings.append("UNSEEDED")

    return {"warnings": sorted(set(warnings)), "blocking": sorted(set(blocking))}


def _is_floating_alias(model: str) -> bool:
    """Whether a model name will silently point at different weights over time.

    `gpt-4o-mini` is an alias; `gpt-4o-mini-2024-07-18` is a snapshot. Two
    researchers six months apart run different judges under the same string.
    """
    import re

    return not re.search(r"\d{4}-\d{2}-\d{2}|\d{8}|@\w+", model)


def _content_hash(bundle: Dict[str, Any]) -> str:
    payload = {k: v for k, v in bundle.items() if k != "run_id"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
