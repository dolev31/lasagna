"""The compose stack does not publish Redis on the host unless asked to.

Redis is the Celery broker and has no password. Compose published it on
127.0.0.1:6379, so any process on the host could queue a task, among them
``knowledge.gc.orphan_scan(dry_run=False)``, which deletes what it calls
orphans. The services reach Redis on the compose network; ``expose-redis.yml``
publishes it on loopback for debugging, when added on purpose.
"""

from __future__ import annotations

import yaml

from tests._paths import DEPLOY_DIR


def _services(name):
    return yaml.safe_load((DEPLOY_DIR / name).read_text()).get("services") or {}


def test_redis_is_not_published_by_the_stack():
    services = _services("docker-compose.yml")
    assert "ports" not in services["redis"]
    for fragment in sorted((DEPLOY_DIR / "backends").glob("*.y*ml")):
        published = (_services(f"backends/{fragment.name}").get("redis") or {}).get("ports")
        assert not published, f"{fragment.name} publishes redis: {published}"


def test_the_services_reach_it_on_the_compose_network():
    common = yaml.safe_load((DEPLOY_DIR / "docker-compose.yml").read_text())["x-env"]
    assert common["RAG_REDIS_URL"] == "redis://redis:6379"
    assert common["RAG_CELERY_BROKER_URL"].startswith("redis://redis:6379/")


def test_the_debug_override_publishes_on_loopback_only():
    (port,) = _services("expose-redis.yml")["redis"]["ports"]
    assert port.startswith("127.0.0.1:") and port.endswith(":6379")
