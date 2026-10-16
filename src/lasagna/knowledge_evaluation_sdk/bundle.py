"""The results bundle, and the key that decides what may be compared to what.

Publishing numbers to a shared leaderboard has one hard problem, and it is not
storage. It is that two rows are only comparable when everything except the
thing under test was held fixed — and until now nothing recorded what was held
fixed, so `compare()` unioned two result sets and printed them side by side
whatever they were measuring.

The comparability key is a hash over the held-fixed configuration:

    metrics_version, benchmark, split, n_samples, sampling seed,
    sample_id_digest, corpus_fingerprint, corpus_provenance, resolved_mode,
    embedder + dimension, chunker, retriever top_k, vector backend, judge model,
    BM25 stemming, and the stage-manifest digest of every stage the run does
    not vary

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
import re
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
#: 1.3: held-fixed fields come from what each task ran (TaskResult.system), and
#: BM25 stemming is part of the key.
#: 1.4: exact_match and token_f1 treat an answer as an abstention only when it
#: opens with one, not when it contains a hedge anywhere, and normalize in the
#: official SQuAD order (punctuation before articles). The key also covers the
#: stage-manifest digests of every stage the run does not vary
#: (``fixed_stages``): dense-only and hybrid runs used to share a key.
#: 1.5: exact_match and token_f1 score the answer with its citation markers
#: removed, as ALCE does ("Paris [1]" failed EM against "Paris"); NDCG gives a
#: negative relevance grade zero gain, as trec_eval does; the harness asks the
#: judge for factual_correctness too. New per-query metrics, absent from older
#: bundles rather than redefined: r_precision@k, complete_recall@k, rouge_l,
#: contains_answer, abstention_rate, answer_similarity, citation_recall,
#: citation_precision, and the *_per_query token and call aggregates.
#: 1.6: the retrieval track scores the ranked list. With context reordering on
#: (standard, balanced and max_quality turn it on; standard is the default
#: base_profile) it scored the order the generator reads instead: even ranks
#: first, odd ranks reversed at the end, so rank 2 fell below the cut to k.
#: Profiles and arms with reordering off (speed) score as before. The stage
#: manifest records which order each track used (``context_order``). In the
#: generation track the search reordered all ten passages and the harness kept
#: the first five, so the generator read ranks 1, 3, 5, 7 and 9; it now reads
#: ranks 1 to 5 reordered, and the judge sees them in ranked order (context
#: precision weighs each passage by its rank). Query fusion fuses each
#: variation's ranked list, where it read reordered positions as ranks.
METRICS_VERSION = "1.6"

#: Bundle schema version. Consumers accept >=1.0,<2.0.
SCHEMA_VERSION = "1.0"

#: What a run's ``determinism`` may say. ``seeded_per_task``: one task at a
#: time, each seeding the process-wide generators before it runs.
#: ``seeded_global_concurrent``: tasks interleaved and shared those generators.
#: ``seeded`` is what runs recorded before the two were told apart.
SEEDED_LABELS = ("seeded_per_task", "seeded_global_concurrent", "seeded")


class MetricsVersionError(ValueError):
    """A result's metric definitions are missing or are not this build's."""


def metrics_version_problem(recorded: Any, package_version: Any = None) -> Optional[str]:
    """Why a result computed under ``recorded`` cannot be bundled here, or None."""
    rerun = (
        "Run the experiment again with this version of Lasagna "
        "(lasagna benchmark experiment <sweep.yaml>) and bundle the new results.json"
    )
    if not recorded:
        return (
            "the results record no metrics_version, so it is not known which metric "
            f"definitions produced their numbers (results.json files from before 0.1.0 "
            f"do not have one). {rerun}."
        )
    if str(recorded) != METRICS_VERSION:
        made_by = f" by Lasagna {package_version}" if package_version else ""
        return (
            f"the results were computed{made_by} under metrics version {recorded}, and "
            f"this version of Lasagna computes {METRICS_VERSION}; the definitions differ, "
            f"so the numbers would be labelled as something they are not. {rerun}, or "
            f"bundle them with the version that produced them."
        )
    return None


def metrics_version_problems(bundle: Dict[str, Any]) -> List[str]:
    """What ``results validate`` refuses about a bundle's metrics version."""
    problems = []
    problem = metrics_version_problem(
        bundle.get("metrics_version"),
        ((bundle.get("experiment") or {}).get("environment") or {}).get("lasagna_version"),
    )
    if problem:
        problems.append(f"metrics_version: {problem}")
    for cfg in bundle.get("configs") or []:
        held = (cfg.get("held_fixed") or {}).get("metrics_version")
        if held != bundle.get("metrics_version"):
            problems.append(
                f"config {cfg.get('name')!r}: held_fixed.metrics_version {held!r} differs "
                f"from the bundle's {bundle.get('metrics_version')!r}"
            )
    return problems


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
    # Whether BM25 stemmed its terms. It depends on nltk being installed, so two
    # machines running the same config could differ here and nowhere else.
    "bm25_stemming",
    # {stage: digest} from the row's stage manifest, for every stage the run's
    # arms do not vary. The fields above say nothing about retrieval mode,
    # fusion weights, MMR or HyDE, so a dense-only and a hybrid sweep shared a
    # key and one leaderboard table. None when the row recorded no manifest.
    "fixed_stages",
    # The embedder answer_similarity scores with (RAG_EVAL_SIMILARITY_MODEL).
    "similarity_model",
)

#: Fields hashed only for blocks at or after the metrics version that added
#: them, so a bundle minted before still recomputes to the key it carries.
#: Later blocks hash the field whether or not it is present (absent is None).
_KEY_FIELDS_SINCE: Dict[str, tuple] = {
    "fixed_stages": (1, 4),
    "similarity_model": (1, 5),
}

#: The stages a sweep varies. Excluded from the key, recorded alongside it.
VARIABLE_DIMENSIONS = (
    "reranker",
    "embedder",
    "chunker",
    "query_enhancer",
    "retriever",
    "agentic_pipeline",
    "generator",
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
    version = _version_tuple(held_fixed.get("metrics_version"))
    payload = {}
    for field in HELD_FIXED_FIELDS:
        since = _KEY_FIELDS_SINCE.get(field)
        if since is not None and version < since:
            continue
        value = held_fixed.get(field)
        payload[field] = KEY_DEFAULTS[field] if value is None and field in KEY_DEFAULTS else value

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _versions_before(since: tuple) -> List[str]:
    """Every ``major.minor`` metrics version before ``since`` within its major."""
    major, minor = since
    return [f"{major}.{m}" for m in range(minor)]


def _version_tuple(version: Any) -> tuple:
    """``"1.10"`` as ``(1, 10)``, so it sorts after ``"1.4"``; unparseable is newest."""
    try:
        return tuple(int(part) for part in str(version).split("."))
    except ValueError:
        return (10**9,)


def varied_stages(
    config_overrides: Dict[str, Any],
    across: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """The pipeline stages a run varies: those of every override that differs
    between its arms, or of this config's own overrides when it is the only
    arm (the same fallback as :func:`varied_dimensions`).

    Taken across the run, not per arm, so every arm of a sweep leaves the same
    stages out of its key and the arms stay in one comparability group.
    """
    from .stages import stage_of

    if across is None or len(across) < 2:
        keys = {k for k in config_overrides if k not in _DERIVED_OVERRIDE_KEYS}
    else:
        candidates = {k for cfg in across for k in cfg if k not in _DERIVED_OVERRIDE_KEYS}
        keys = {
            k
            for k in candidates
            if len({json.dumps(cfg.get(k), sort_keys=True, default=str) for cfg in across}) > 1
        }
    return sorted({stage for stage in map(stage_of, keys) if stage is not None})


def held_fixed_from(
    result: Dict[str, Any],
    run_meta: Dict[str, Any],
    across: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble the held-fixed block from a task result and its run metadata.

    ``across`` is every config's overrides in the run, which decides the
    stages left out of ``fixed_stages``; see :func:`varied_stages`.
    """
    declared = run_meta.get("system", {}) or {}
    ran = result.get("system") or {}
    manifest = ran.get("stages") or None
    varied = set(varied_stages(result.get("config_overrides") or {}, across))

    def _field(name: str) -> Any:
        # What the task recorded running with wins over what the profile
        # declared; the declaration is the fallback for older results.
        value = ran.get(name)
        return value if value is not None else declared.get(name)

    return {
        # The run's own: stamping this build's made a result computed under
        # older definitions look like one computed under these.
        "metrics_version": run_meta.get("metrics_version"),
        "benchmark": result.get("benchmark_name"),
        "split": result.get("split"),
        "n_samples": result.get("n_samples"),
        "sampling_seed": result.get("sampling_seed"),
        "sample_id_digest": result.get("sample_id_digest"),
        "corpus_fingerprint": result.get("corpus_fingerprint"),
        "corpus_provenance": result.get("corpus_provenance"),
        "resolved_mode": result.get("resolved_mode"),
        "embedder_model": _field("embedder_model"),
        "embedder_dim": _field("embedder_dim"),
        "chunker": _field("chunker"),
        "retriever_top_k": _field("retriever_top_k"),
        "vector_backend": _field("vector_backend"),
        "judge_model": run_meta.get("judge_model"),
        "similarity_model": run_meta.get("similarity_model"),
        "bm25_stemming": _field("bm25_stemming"),
        "fixed_stages": (
            {
                stage: (entry or {}).get("digest")
                for stage, entry in sorted(manifest.items())
                if stage not in varied
            }
            if manifest
            else None
        ),
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
    diffs = []
    for field in HELD_FIXED_FIELDS:
        va, vb = a.get(field), b.get(field)
        if va == vb:
            continue
        if field == "fixed_stages" and isinstance(va, dict) and isinstance(vb, dict):
            # Per stage, so the reader knows which part of the pipeline to look
            # at; the manifests in each bundle's rows name the settings.
            for stage in sorted(set(va) | set(vb)):
                if va.get(stage) != vb.get(stage):
                    diffs.append(f"fixed_stages.{stage}: {va.get(stage)!r} vs {vb.get(stage)!r}")
            continue
        diffs.append(f"{field}: {va!r} vs {vb!r}")
    return diffs


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
    problem = metrics_version_problem(
        run_meta.get("metrics_version"), run_meta.get("package_version")
    )
    if problem:
        raise MetricsVersionError(f"Cannot bundle: {problem}")
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
        held = held_fixed_from(row, run_meta, across)
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
        "metrics_version": run_meta["metrics_version"],
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
                    "nothing_samples": {
                        "type": "boolean",
                        "description": (
                            "True when only retrieval ran and no task called an LLM (and "
                            "repeats, if any, were identical): one run is then the whole "
                            "answer, and SINGLE_RUN is not raised."
                        ),
                    },
                    "determinism": {
                        "enum": list(SEEDED_LABELS),
                        "description": (
                            "seeded_per_task: tasks ran one at a time, each seeding the "
                            "process-wide generators first. seeded_global_concurrent: "
                            "tasks interleaved and shared them. Never 'reproducible': "
                            "LLM judges and GPU kernels defeat that claim."
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
                            # same value for it either way. So are fields gated
                            # by metrics version (_KEY_FIELDS_SINCE), which an
                            # older bundle cannot carry and a newer one hashes
                            # as None when absent. Everything else is required,
                            # because a key hashing a field the schema does not
                            # demand is a key nobody can recompute.
                            "required": [
                                f
                                for f in HELD_FIXED_FIELDS
                                if f not in KEY_DEFAULTS and f not in _KEY_FIELDS_SINCE
                            ],
                            # A field gated by version is required from that
                            # version on, so a current bundle cannot omit it.
                            "allOf": [
                                {
                                    "if": {
                                        "required": ["metrics_version"],
                                        "properties": {
                                            "metrics_version": {
                                                "not": {"enum": _versions_before(since)}
                                            }
                                        },
                                    },
                                    "then": {"required": [field]},
                                }
                                for field, since in sorted(_KEY_FIELDS_SINCE.items())
                            ],
                            "properties": {
                                "fixed_stages": {
                                    "type": ["object", "null"],
                                    "description": (
                                        "Stage-manifest digest of every stage the run "
                                        "does not vary, from metrics_version 1.4. Null "
                                        "when the row recorded no manifest; such rows "
                                        "are compared only within their own bundle."
                                    ),
                                },
                            },
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
                    "properties": {
                        "per_sample_scores": {
                            "type": "array",
                            "description": (
                                "One row per query. Retrieval rows carry `_retrieved_ids` "
                                "(the ranked list the metrics were computed on) and "
                                "`_retrieved` ({doc_id, score, rank, chunk_rank}). `rank` "
                                "is authoritative: scores can tie, so re-sorting by score "
                                "does not rebuild the ranking (a keyword reranker with "
                                "ties on every scifact query rescored to NDCG@10 0.5272 "
                                "instead of 0.5709)."
                            ),
                        },
                    },
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


#: What each stand-in changes, and how to remove it.
FALLBACK_FIXES: Dict[str, str] = {
    "sentence_split_without_nltk": "sentences are split by punctuation; pip install 'lasagna-rag[eval]'",
    "sentence_split_without_punkt": (
        "sentences are split by punctuation; python -m nltk.downloader punkt_tab"
    ),
    "bm25_without_stemming": "BM25 matches unstemmed words; pip install 'lasagna-rag[eval]'",
    "hybrid_without_bm25": "hybrid search has no sparse side; pip install rank-bm25",
    "docling_requested_but_missing": "documents are parsed without docling; pip install docling",
}

#: Benchmark corpora are loaded as text, so document parsing never runs.
_NEVER_IN_A_BENCHMARK = frozenset({"docling_requested_but_missing"})
_NEEDS_BM25 = frozenset({"bm25_without_stemming", "hybrid_without_bm25"})


def _uses_hybrid(row: Dict[str, Any]) -> Optional[bool]:
    stages = (row.get("system") or {}).get("stages") or {}
    settings = (stages.get("retriever") or {}).get("settings")
    if settings is None:
        return None  # no manifest: cannot tell, so assume it might have
    return bool(settings.get("enable_hybrid_search"))


def _splits_sentences(row: Dict[str, Any]) -> Optional[bool]:
    """Whether the task's chunking ran the sentence splitter.

    Read off the chunker that ran (``stages.effective_chunking_strategy``):
    hierarchical cuts fixed-size parents and children; structure-aware chunks
    each section semantically; a structural strategy with its setting off
    runs semantic, which the declared name hid.
    """
    from .stages import effective_chunking_strategy

    stages = (row.get("system") or {}).get("stages") or {}
    s = (stages.get("chunker") or {}).get("settings")
    if s is None:
        return None
    return effective_chunking_strategy(s) in ("structure_aware", "semantic", "sentence_window")


def _could_have_used(rows: List[Dict[str, Any]], probe) -> bool:
    seen = [probe(r) for r in rows if not r.get("error")]
    return not seen or any(v is not False for v in seen)


def exercised_fallbacks(rows: List[Dict[str, Any]], fallbacks: Optional[List[str]]) -> List[str]:
    """The active fallbacks the run's tasks could have used.

    A task without a manifest counts as having used every stand-in: the
    question cannot be answered, so the reader is told.
    """
    out = [f for f in fallbacks or [] if f not in _NEVER_IN_A_BENCHMARK]
    if not _could_have_used(rows, _uses_hybrid):
        out = [f for f in out if f not in _NEEDS_BM25]
    if not _could_have_used(rows, _splits_sentences):
        out = [f for f in out if not f.startswith("sentence_split_without_")]
    return out


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

    # A missing optional package swaps in a stand-in (sentence splitting,
    # BM25 without stemming) that changes the numbers; a reader comparing
    # with a run on a fuller install needs to know. Only stand-ins this run
    # could have exercised count: docling, requested by the default settings,
    # never touches a benchmark corpus, and flagging it meant `--strict` could
    # not pass on the documented install.
    exercised = exercised_fallbacks(
        rows, (run_meta.get("environment") or {}).get("active_fallbacks")
    )
    if exercised:
        warnings.append("ACTIVE_FALLBACKS")

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

    if run_meta.get("n_runs", 1) < 2 and not run_meta.get("nothing_samples"):
        # Not blocking on its own: a single run is a legitimate pilot. It only
        # becomes a problem when a significance claim is attached to it. A run
        # in which nothing samples (retrieval only, no LLM call) is complete
        # with one run, as the docs recommend, so it carries no warning.
        warnings.append("SINGLE_RUN")

    judge = run_meta.get("judge_model")
    if judge and _is_floating_alias(judge):
        warnings.append("UNRESOLVED_JUDGE_ALIAS")

    if run_meta.get("determinism") not in SEEDED_LABELS:
        warnings.append("UNSEEDED")

    # Stage isolation: two arms may differ only in the stages they name. A
    # difference anywhere else means the comparison is not attributable, which
    # is the property the whole framework exists to provide, so it blocks.
    from .stages import isolation_report

    isolation = isolation_report(rows)
    if isolation["undeclared"]:
        blocking.append("UNDECLARED_STAGE_CHANGE")
    if isolation["inert"]:
        warnings.append("INERT_ARM")
    if isolation["unverified"] and rows:
        warnings.append("STAGE_ISOLATION_UNVERIFIED")

    return {
        "warnings": sorted(set(warnings)),
        "blocking": sorted(set(blocking)),
        "isolation": isolation,
        # Each exercised stand-in with what it changes and how to remove it, so
        # ACTIVE_FALLBACKS is something a reader can act on.
        "fallbacks": [f"{f}: {FALLBACK_FIXES.get(f, 'see lasagna doctor')}" for f in exercised],
    }


def _is_floating_alias(model: str) -> bool:
    """Whether a model name will silently point at different weights over time.

    `gpt-5.5` is an alias; `gpt-5.5-2026-04-23` is a snapshot. Two researchers
    six months apart run different judges under the same string.

    Some ids are snapshots without a date: every Claude id from the 4.6
    generation on (`claude-sonnet-5-5`), and OpenAI models whose only listed
    snapshot is the dateless id (`gpt-6-luna`). The capability table knows
    which; anything else without a date is still an alias.
    """
    import re

    from lasagna.providers.model_capabilities import model_capabilities

    if model_capabilities(model).pinned_snapshot:
        return False
    return not re.search(r"\d{4}-\d{2}-\d{2}|\d{8}|@\w+", model)


def aggregate_mismatches(rows: List[Dict[str, Any]], tol: float = 1e-6) -> List[str]:
    """Aggregates that are not the mean of their own row's per-query scores.

    The run id catches a bundle edited after it was built, but not one built
    from an edited results.json: a headline NDCG@10 of 0.665 over per-query
    scores averaging 0.609 validated.

    Each metric is checked against the per-query rows that carry it. An
    end-to-end task lists a retrieval row and a generation row per query, and
    requiring every row to carry the metric skipped every metric of every
    e2e task. Metrics no row carries, or with an unscored (NaN) query, are not
    second-guessed, since they are aggregated another way.
    """
    problems: List[str] = []
    for row in rows:
        samples = [s for s in (row.get("per_sample_scores") or []) if isinstance(s, dict)]
        if not samples:
            continue
        for name, value in (row.get("metrics") or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            scores = [s[name] for s in samples if name in s]
            if not scores or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or v != v for v in scores
            ):
                continue
            mean = sum(scores) / len(scores)
            if abs(mean - value) > tol:
                problems.append(
                    f"{row.get('config_name')!r} run {row.get('run_index')}: {name} is "
                    f"{value:.6g} but its {len(scores)} per-query scores average {mean:.6g}"
                )
    return problems


#: Binary IR metrics a validator can recompute from the exported ranked list.
_RECHECKABLE = re.compile(r"^(precision|hit_rate|mrr)@(\d+)$")


def ranked_list_mismatches(rows: List[Dict[str, Any]], tol: float = 1e-9) -> List[str]:
    """Per-query scores higher than their own exported ranked list supports.

    Each retrieval row exports ``_retrieved_ids`` (the ranked list) and
    ``_expected_docs`` (the judged documents), so precision, hit rate and MRR
    at k can be recomputed. That catches a score raised, or a list reordered,
    without the other, which recomputing the run id cannot: anyone who edits a
    number can recompute the id.

    One-sided on purpose. Grades are not exported, and ``_expected_docs`` can
    include documents judged non-relevant; counting all of them as relevant
    can only raise these three metrics, so an honest score never exceeds the
    recomputed one. Graded metrics (NDCG) and recall-based ones are not
    rechecked, for the same reason.
    """
    from .ir_metrics import hit_rate_at_k, mrr, precision_at_k

    recompute = {"precision": precision_at_k, "hit_rate": hit_rate_at_k, "mrr": mrr}
    problems: List[str] = []
    for row in rows:
        for i, sample in enumerate(row.get("per_sample_scores") or []):
            if not isinstance(sample, dict):
                continue
            ranked, expected = sample.get("_retrieved_ids"), sample.get("_expected_docs")
            if not isinstance(ranked, list) or not isinstance(expected, (list, dict)):
                continue
            relevance = {str(d): 1 for d in expected}
            for name, stored in sample.items():
                match = _RECHECKABLE.match(name)
                if not match or isinstance(stored, bool) or not isinstance(stored, (int, float)):
                    continue
                kind, k = match.group(1), int(match.group(2))
                bound = recompute[kind](relevance, [str(d) for d in ranked], k)
                if stored > bound + tol:
                    problems.append(
                        f"{row.get('config_name')!r} run {row.get('run_index')} query "
                        f"{sample.get('_sample_id', i)!r}: {name} is {stored:.6g} but its "
                        f"ranked list gives at most {bound:.6g}"
                    )
    return problems


def _content_hash(bundle: Dict[str, Any]) -> str:
    payload = {k: v for k, v in bundle.items() if k != "run_id"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
