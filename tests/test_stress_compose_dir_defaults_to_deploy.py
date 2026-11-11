"""The stress suite must default compose_dir to the real stack (#31)."""

from pathlib import Path

from tests._paths import DEPLOY_DIR
from tests.stress.config import StressConfig


def test_default_compose_dir_holds_docker_compose_yml(monkeypatch):
    monkeypatch.delenv("STRESS_COMPOSE_DIR", raising=False)
    cfg = StressConfig.from_env()
    compose = Path(cfg.compose_dir)
    assert compose == DEPLOY_DIR
    assert (compose / "docker-compose.yml").is_file()
