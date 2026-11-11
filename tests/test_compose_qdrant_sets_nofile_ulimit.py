"""The compose stack's nofile policy: every store service states its decision.

The Qdrant story first: eight concurrent clients creating collections and
payload indexes exhausted the container's default 1024 FDs ("Too many open
files" in the Qdrant log) during stress testing of the collection-creation
race fix. The Qdrant service therefore raises ``nofile`` to 524288, the
value that run validated; ``test_qdrant_service_raises_nofile_ulimit`` pins
that requirement to the compose file so a regression fails here instead of
in production.

The same class of problem covers every store service in the stack, so the
rest of this file pins the whole policy, not just Qdrant:

* ``PINNED`` -- (compose file, service) pairs that must set a minimum soft
  ``nofile``, with the reason each value was chosen.
* ``PENDING_MEASUREMENT`` -- pairs whose limit is still being measured: the
  pair is known and the named burst/measure test decides the number. A pair
  moves to ``PINNED`` (or ``SAFE_UNSET``) once the run reports.
* ``SAFE_UNSET`` -- pairs deliberately left at the default, each with a
  one-line reason.

``test_every_store_service_has_an_explicit_nofile_decision`` enumerates every
service that runs its own image in ``docker-compose.yml`` and
``backends/*.yml``. A new backend file (or a new image service) that is not
listed in one of the three tables fails this test, so the decision cannot be
forgotten -- add the pair to the right table with its reason.
"""

from __future__ import annotations

import yaml

from tests._paths import DEPLOY_DIR

MIN_NOFILE = 524288

OPENSEARCH_DOCS_URL = (
    "https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/"
)

# (compose file relative to DEPLOY_DIR, service) -> (minimum soft nofile, why)
PINNED = {
    ("backends/qdrant.yml", "qdrant"): (
        524288,
        "validated by the 8-client collection+index burst "
        "(main 50/50 failed at the 1024 default, branch 50/50 clean at 524288)",
    ),
    ("backends/opensearch.yml", "opensearch"): (
        65536,
        "OpenSearch's own Docker documentation requires nofile 65536 "
        f"(and memlock unlimited) on every node: {OPENSEARCH_DOCS_URL}",
    ),
}

# (compose file, service) -> (burst/measure test relative to tests/, what the run decides)
# Pairs stay here only while unmeasured; once a run reports, move the pair to
# PINNED or SAFE_UNSET and record the measured numbers in the reason.
PENDING_MEASUREMENT = {}

# (compose file, service) -> one-line reason for leaving the default unset
SAFE_UNSET = {
    ("backends/milvus.yml", "milvus"): (
        "safe, because the 8-client burst measured 2026-11-03 ran 50/50 clean "
        "(zero failures) at the default soft nofile 1024 / hard 1048576, so the "
        "default holds and no pin is needed"
    ),
    ("backends/weaviate.yml", "weaviate"): (
        "safe, because the Weaviate container runs one server process and never "
        "fans out concurrent collection creators itself -- collections are "
        "created by the single RAG API process -- so there is no client-burst "
        "path inside the container"
    ),
    ("backends/pgvector.yml", "postgres"): (
        "safe, because the pgvector backend reuses the metadata Postgres with "
        "max_connections=200, which caps concurrent connections (and file "
        "descriptors) far below the default limit"
    ),
    ("docker-compose.yml", "postgres"): (
        "safe, because it is the metadata database behind a connection pool; "
        "its file-descriptor usage scales with the pool size, not with "
        "client bursts"
    ),
    ("docker-compose.yml", "redis"): (
        "safe, because Redis is single-threaded with roughly one file "
        "descriptor per connection and the stack's connection counts are "
        "small; there is no burst path"
    ),
}

# (compose file, service) -> why it is not a store service at all
IGNORED_NON_STORE = {
    ("docker-compose.yml", "jaeger"): "observability, not a store service",
}


def _soft_nofile(service: dict) -> int:
    ulimit = (service.get("ulimits") or {}).get("nofile")
    if isinstance(ulimit, dict):
        return int(ulimit.get("soft", 0))
    return int(ulimit or 0)


def _image_services(compose_path):
    """Services that run their own image (as opposed to the built app image)."""
    doc = yaml.safe_load(compose_path.read_text())
    return {
        name
        for name, svc in (doc.get("services") or {}).items()
        if isinstance(svc, dict) and "image" in svc
    }


def test_qdrant_service_raises_nofile_ulimit():
    services = yaml.safe_load((DEPLOY_DIR / "backends/qdrant.yml").read_text())["services"]
    assert _soft_nofile(services["qdrant"]) >= MIN_NOFILE


def test_pinned_services_meet_their_minimums():
    for (rel, service), (minimum, reason) in PINNED.items():
        services = yaml.safe_load((DEPLOY_DIR / rel).read_text())["services"]
        soft = _soft_nofile(services[service])
        assert soft >= minimum, (
            f"{rel}#{service}: soft nofile {soft} < pinned minimum {minimum} ({reason})"
        )


def test_opensearch_sets_memlock_unlimited():
    services = yaml.safe_load((DEPLOY_DIR / "backends/opensearch.yml").read_text())["services"]
    memlock = (services["opensearch"].get("ulimits") or {}).get("memlock")
    if isinstance(memlock, dict):
        soft, hard = memlock.get("soft"), memlock.get("hard")
    else:  # compose also allows the scalar form, e.g. `memlock: -1`
        soft = hard = memlock
    assert int(soft or 0) == -1, "opensearch memlock soft must be unlimited (-1)"
    assert int(hard or 0) == -1, "opensearch memlock hard must be unlimited (-1)"


def test_pending_measurements_have_their_measure_test():
    import tests as tests_pkg
    from pathlib import Path

    tests_dir = Path(tests_pkg.__file__).resolve().parent
    for (rel, service), (test_rel, _note) in PENDING_MEASUREMENT.items():
        assert (tests_dir / test_rel).is_file(), (
            f"{rel}#{service} is pending measurement but {test_rel} is missing"
        )


def test_every_store_service_has_an_explicit_nofile_decision():
    known = set(PINNED) | set(SAFE_UNSET) | set(PENDING_MEASUREMENT)
    unknown = []
    backends_dir = DEPLOY_DIR / "backends"
    for path in sorted(backends_dir.glob("*.yml")):
        for service in sorted(_image_services(path)):
            key = (f"backends/{path.name}", service)
            if key not in known:
                unknown.append(key)
    for service in sorted(_image_services(DEPLOY_DIR / "docker-compose.yml")):
        key = ("docker-compose.yml", service)
        if key not in known and key not in IGNORED_NON_STORE:
            unknown.append(key)
    assert not unknown, (
        "store services without an explicit nofile decision: "
        + ", ".join(f"{rel}#{service}" for rel, service in unknown)
        + " -- add each to PINNED, SAFE_UNSET or PENDING_MEASUREMENT with its reason"
    )
