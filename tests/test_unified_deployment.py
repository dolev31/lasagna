"""Tests for the unified plug-and-play deployment system.

Four test layers:
  Layer 1 — Static compose validation (no Docker containers needed)
  Layer 2 — Registry empty-string fix (unit tests)
  Layer 3 — Live deployment E2E (requires running Docker stack)
  Layer 4 — Kustomize overlay validation (no cluster needed)

Run static tests only (instant):
  pytest tests/test_unified_deployment.py -v -k "not Live"

Run everything including live E2E:
  pytest tests/test_unified_deployment.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from tests._paths import DEPLOY_DIR  # noqa: E402 — resolved from the installed package

BACKENDS = ["milvus", "milvus-lite", "pgvector", "qdrant", "weaviate", "pinecone", "opensearch"]

# `make validate` also requires RAG_JWT_SECRET. Pass a throwaway value on the
# command line rather than relying on the ambient environment: the Makefile
# does `-include .env`, so a developer with a local deploy/.env would see these
# tests pass while CI and every fresh clone saw them fail.
_VALIDATION_JWT_SECRET = "test-secret-for-validation"

# The compose files deliberately declare every credential as required
# (`${VAR:?...}`) so a stack can never boot with a password published in the
# repo. That means `docker compose config` needs them supplied here.
_REQUIRED_COMPOSE_SECRETS = {
    "RAG_JWT_SECRET": _VALIDATION_JWT_SECRET,
    "POSTGRES_PASSWORD": "test-postgres-password",
}


def _compose_env() -> dict:
    """Environment for `docker compose` dry runs.

    Ambient values win so a developer can point the tests at a real stack, but
    a throwaway default is always present so nothing depends on local setup.
    """
    env = {**os.environ}
    for key, fallback in _REQUIRED_COMPOSE_SECRETS.items():
        env.setdefault(key, fallback)
    return env


BASE_URL = "http://localhost:8000/api/v1"
HEALTH_TIMEOUT = 120  # seconds to wait for stack health
POLL_INTERVAL = 2.0
DOC_READY_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_merged_config(backend: str) -> dict:
    """Run 'docker compose config' and parse the merged YAML.

    This is a dry-run — it validates the compose files without starting
    any containers.  Requires the ``docker`` CLI to be installed.
    """
    env = _compose_env()
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            f"backends/{backend}.yml",
            "config",
        ],
        cwd=str(DEPLOY_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"Compose config merge failed for {backend}:\n{result.stderr}"
    return yaml.safe_load(result.stdout)


def _docker_compose_available() -> bool:
    """Check whether the docker compose CLI is available."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _wait_for_health(timeout: int = HEALTH_TIMEOUT) -> None:
    """Poll the API health/live endpoint until it returns 200."""
    import requests as _requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _requests.get(f"{BASE_URL}/health/live", timeout=5)
            if r.status_code == 200:
                return
        except _requests.ConnectionError:
            pass
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"API did not become healthy within {timeout}s")


def _wait_for_document_ready(
    doc_id: str,
    headers: dict,
    timeout: int = DOC_READY_TIMEOUT,
) -> None:
    """Poll document status until 'ready' or timeout."""
    import requests as _requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _requests.get(
            f"{BASE_URL}/documents/{doc_id}/status",
            headers=headers,
            timeout=10,
        )
        if r.status_code == 200:
            status = r.json().get("status")
            if status == "ready":
                return
            if status == "error":
                raise AssertionError(f"Document {doc_id} failed: {r.json().get('error_message')}")
        time.sleep(1.0)
    raise TimeoutError(f"Document {doc_id} not ready within {timeout}s")


# Skip the entire module if docker compose CLI is not available
pytestmark = pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose CLI not available",
)


# ============================================================================
# Layer 1: Static Compose Validation (no containers)
# ============================================================================


class TestComposeFileStructure:
    """Validates that all required deployment files exist."""

    def test_base_compose_exists(self):
        assert (DEPLOY_DIR / "docker-compose.yml").is_file(), "Base docker-compose.yml not found"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_backend_fragment_exists(self, backend):
        path = DEPLOY_DIR / "backends" / f"{backend}.yml"
        assert path.is_file(), f"Backend fragment not found: {path}"

    def test_env_example_exists(self):
        env_path = DEPLOY_DIR / ".env.example"
        assert env_path.is_file(), ".env.example not found"
        content = env_path.read_text()
        assert "VECTOR_BACKEND" in content, ".env.example must document VECTOR_BACKEND"

    def test_makefile_exists(self):
        assert (DEPLOY_DIR / "Makefile").is_file(), "Makefile not found"

    def test_backends_directory_exists(self):
        assert (DEPLOY_DIR / "backends").is_dir(), "backends/ directory not found"


class TestComposeMerging:
    """Validates compose config merging for each backend (dry-run)."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_compose_config_valid(self, backend):
        """'docker compose config' must succeed for base + backend."""
        env = _compose_env()
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                f"backends/{backend}.yml",
                "config",
            ],
            cwd=str(DEPLOY_DIR),
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"Merge failed for {backend}:\n{result.stderr}"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_api_service_has_correct_backend(self, backend):
        """Merged config must set RAG_VECTOR_BACKEND correctly on api."""
        config = _get_merged_config(backend)
        api_env = config["services"]["api"]["environment"]
        expected = "milvus" if backend == "milvus-lite" else backend
        assert api_env["RAG_VECTOR_BACKEND"] == expected, (
            f"Expected RAG_VECTOR_BACKEND={expected}, got {api_env.get('RAG_VECTOR_BACKEND')}"
        )

    def test_pgvector_swaps_postgres_image(self):
        """pgvector backend must use pgvector/pgvector:pg16."""
        config = _get_merged_config("pgvector")
        image = config["services"]["postgres"]["image"]
        assert "pgvector" in image, f"Expected pgvector image, got {image}"

    def test_non_pgvector_keeps_default_postgres(self):
        """Non-pgvector backends must keep postgres:16-alpine."""
        for backend in ("milvus", "qdrant", "pinecone"):
            config = _get_merged_config(backend)
            image = config["services"]["postgres"]["image"]
            assert image == "postgres:16-alpine", (
                f"{backend}: expected postgres:16-alpine, got {image}"
            )

    def test_milvus_adds_milvus_service(self):
        config = _get_merged_config("milvus")
        assert "milvus" in config["services"], "milvus backend must add a milvus service"

    def test_milvus_service_has_healthcheck(self):
        config = _get_merged_config("milvus")
        assert "healthcheck" in config["services"]["milvus"], (
            "milvus service must have a healthcheck"
        )

    def test_pgvector_no_milvus_service(self):
        config = _get_merged_config("pgvector")
        assert "milvus" not in config["services"], (
            "pgvector backend must NOT include a milvus service"
        )

    def test_pinecone_no_extra_vector_services(self):
        config = _get_merged_config("pinecone")
        services = set(config["services"].keys())
        vector_services = {"milvus", "qdrant", "weaviate"}
        overlap = services & vector_services
        assert overlap == set(), f"pinecone backend has unexpected vector services: {overlap}"

    def test_qdrant_adds_qdrant_service(self):
        config = _get_merged_config("qdrant")
        assert "qdrant" in config["services"], "qdrant backend must add a qdrant service"
        assert "healthcheck" in config["services"]["qdrant"], (
            "qdrant service must have a healthcheck"
        )

    def test_weaviate_adds_weaviate_service(self):
        config = _get_merged_config("weaviate")
        assert "weaviate" in config["services"], "weaviate backend must add a weaviate service"

    def test_milvus_lite_single_worker(self):
        """milvus-lite must force --workers=1 (embedded single-process)."""
        config = _get_merged_config("milvus-lite")
        command = config["services"]["api"].get("command", "")
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        assert "workers=1" in cmd_str, (
            f"milvus-lite api command must include --workers=1, got: {cmd_str}"
        )

    def test_milvus_lite_embeds_mcp(self):
        config = _get_merged_config("milvus-lite")
        api_env = config["services"]["api"]["environment"]
        val = str(api_env.get("RAG_EMBED_MCP", "")).lower()
        assert val == "true", f"milvus-lite must set RAG_EMBED_MCP=true, got {val}"

    def test_milvus_lite_disables_celery(self):
        config = _get_merged_config("milvus-lite")
        api_env = config["services"]["api"]["environment"]
        val = str(api_env.get("RAG_CELERY_ENABLED", "")).lower()
        assert val == "false", f"milvus-lite must set RAG_CELERY_ENABLED=false, got {val}"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_all_backends_have_core_services(self, backend):
        """Every backend must include api, redis, and postgres."""
        config = _get_merged_config(backend)
        for svc in ("api", "redis", "postgres"):
            assert svc in config["services"], f"{backend}: missing core service '{svc}'"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_shared_env_vars_present(self, backend):
        """Every backend's api service must have all shared env vars."""
        config = _get_merged_config(backend)
        api_env = config["services"]["api"]["environment"]
        required_keys = [
            "RAG_DATABASE_BACKEND",
            "RAG_DATABASE_DSN",
            "RAG_REDIS_URL",
            "RAG_CACHE_BACKEND",
            "RAG_TENANT_ISOLATION",
            "RAG_ENABLE_PROMETHEUS",
            "RAG_CONNECTION_POOL_SIZE",
            "RAG_VECTOR_BACKEND",
        ]
        for key in required_keys:
            assert key in api_env, f"{backend}: missing env var '{key}'"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_no_backend_specific_env_pollution(self, backend):
        """Backend-specific env vars must NOT leak into unrelated backends."""
        config = _get_merged_config(backend)
        api_env = config["services"]["api"]["environment"]

        if backend != "pgvector":
            val = api_env.get("PGVECTOR_DSN", "")
            assert not val, f"PGVECTOR_DSN leaked into {backend}: {val}"

        if backend != "qdrant":
            val = api_env.get("QDRANT_URL", "")
            assert not val, f"QDRANT_URL leaked into {backend}: {val}"

        if backend != "weaviate":
            val = api_env.get("WEAVIATE_URL", "")
            assert not val, f"WEAVIATE_URL leaked into {backend}: {val}"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_postgres_has_healthcheck(self, backend):
        """Postgres must always have a healthcheck."""
        config = _get_merged_config(backend)
        assert "healthcheck" in config["services"]["postgres"]

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_redis_has_healthcheck(self, backend):
        """Redis must always have a healthcheck."""
        config = _get_merged_config(backend)
        assert "healthcheck" in config["services"]["redis"]


class TestComposeProfiles:
    """Validates that profile-gated services are correctly configured.

    Note: ``docker compose config`` strips profiles from the output.
    We read the raw base YAML file to check profile assignments.
    """

    @pytest.fixture(autouse=True)
    def _load_base_compose(self):
        with open(DEPLOY_DIR / "docker-compose.yml") as f:
            self._base = yaml.safe_load(f)

    def test_celery_worker_has_celery_profile(self):
        celery = self._base["services"].get("celery-worker", {})
        profiles = celery.get("profiles", [])
        assert "celery" in profiles, f"celery-worker must have 'celery' profile, got {profiles}"

    def test_mcp_has_mcp_profile(self):
        mcp = self._base["services"].get("mcp", {})
        profiles = mcp.get("profiles", [])
        assert "mcp" in profiles, f"mcp must have 'mcp' profile, got {profiles}"

    def test_jaeger_has_tracing_profile(self):
        jaeger = self._base["services"].get("jaeger", {})
        profiles = jaeger.get("profiles", [])
        assert "tracing" in profiles, f"jaeger must have 'tracing' profile, got {profiles}"

    def test_core_services_have_no_profile(self):
        """api, redis, postgres must NOT have profiles (always start)."""
        for svc in ("api", "redis", "postgres"):
            profiles = self._base["services"][svc].get("profiles", [])
            assert len(profiles) == 0, f"{svc} must not have profiles, got {profiles}"


class TestDockerfileValidation:
    """Validates the Dockerfile uses WEB_CONCURRENCY."""

    def test_uses_web_concurrency(self):
        """CMD must reference WEB_CONCURRENCY, not hardcode workers."""
        content = (DEPLOY_DIR / "Dockerfile").read_text()
        assert "WEB_CONCURRENCY" in content, "Dockerfile must use WEB_CONCURRENCY env var"
        # Must NOT have the old exec-form hardcoded --workers 4
        assert '"--workers", "4"' not in content, (
            "Dockerfile must not hardcode --workers 4 in exec-form"
        )

    def test_has_healthcheck(self):
        content = (DEPLOY_DIR / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content, "Dockerfile must include HEALTHCHECK"

    def test_exposes_port_8000(self):
        content = (DEPLOY_DIR / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content


class TestMakefileValidation:
    """Validates Makefile targets and validation logic."""

    def test_has_required_targets(self):
        content = (DEPLOY_DIR / "Makefile").read_text()
        required = [
            "up:",
            "down:",
            "down-v:",
            "build:",
            "logs:",
            "ps:",
            "health:",
            "validate:",
            "help:",
        ]
        for target in required:
            assert target in content, f"Makefile missing target '{target}'"

    def test_validate_catches_bad_backend(self):
        """make validate VECTOR_BACKEND=nonexistent must fail."""
        result = subprocess.run(
            [
                "make",
                "-C",
                str(DEPLOY_DIR),
                "validate",
                "VECTOR_BACKEND=nonexistent",
                f"RAG_JWT_SECRET={_VALIDATION_JWT_SECRET}",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "make validate should fail for nonexistent backend"
        assert "nonexistent" in (result.stdout + result.stderr), (
            "make validate must fail on the unknown backend specifically, not on a missing secret"
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_validate_accepts_valid_backends(self, backend):
        """make validate VECTOR_BACKEND={backend} must succeed (API-only profiles)."""
        result = _make_validate(f"VECTOR_BACKEND={backend}", "COMPOSE_PROFILES=")
        assert result.returncode == 0, f"make validate failed for {backend}: {result.stdout}"


def _make_validate(*overrides: str) -> subprocess.CompletedProcess:
    """`make validate` with throwaway secrets; ``overrides`` win over them."""
    return subprocess.run(
        [
            "make",
            "-C",
            str(DEPLOY_DIR),
            "validate",
            f"RAG_JWT_SECRET={_VALIDATION_JWT_SECRET}",
            f"POSTGRES_PASSWORD={_REQUIRED_COMPOSE_SECRETS['POSTGRES_PASSWORD']}",
            *overrides,
        ],
        capture_output=True,
        text=True,
    )


def _full_stack(backend: str) -> dict:
    """The merged config with the celery and mcp profiles active, as `make up` runs it."""
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            f"backends/{backend}.yml",
            "--profile",
            "celery",
            "--profile",
            "mcp",
            "config",
        ],
        cwd=str(DEPLOY_DIR),
        capture_output=True,
        text=True,
        env=_compose_env(),
    )
    assert result.returncode == 0, f"Compose config merge failed for {backend}:\n{result.stderr}"
    return yaml.safe_load(result.stdout)


#: Backends whose worker, beat and mcp can share the vector store with the api.
SERVER_BACKENDS = [b for b in BACKENDS if b != "milvus-lite"]
LONG_RUNNING = ("api", "celery-worker", "celery-beat", "mcp")


class TestTheFullStackRuns:
    """The compose stack with Celery and MCP has to work as one system.

    Before: the api stored an upload at a path inside its own container and
    handed it to the worker, which had no such file; the worker and mcp
    containers inherited the image's healthcheck on :8000, which they do not
    serve, so `up --wait` could never succeed; `.env.example` defaulted to
    milvus-lite with the celery and mcp profiles on, a combination that cannot
    work; the Milvus image tag did not exist; and the Dockerfile's unquoted
    `>=` silently dropped the version floors it named.
    """

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_uploads_are_on_one_volume_at_the_upload_dir(self, backend):
        services = _full_stack(backend)["services"]
        for name in ("api", "celery-worker", "mcp"):
            if backend == "milvus-lite" and name != "api":
                continue
            svc = services[name]
            mounts = {
                v["target"]: v.get("source")
                for v in svc.get("volumes", [])
                if v["type"] == "volume"
            }
            upload_dir = svc["environment"]["RAG_UPLOAD_DIR"]
            assert mounts.get(upload_dir) == "uploads-data", (
                f"{backend}/{name}: RAG_UPLOAD_DIR={upload_dir} but the uploads volume is "
                f"mounted at {mounts}"
            )

    @pytest.mark.parametrize("backend", SERVER_BACKENDS)
    def test_every_long_running_service_has_its_own_healthcheck(self, backend):
        services = _full_stack(backend)["services"]
        for name in LONG_RUNNING:
            check = services[name].get("healthcheck", {})
            assert check.get("test") and not check.get("disable"), f"{backend}/{name}: no check"
        for name in ("celery-worker", "celery-beat"):
            probe = " ".join(services[name]["healthcheck"]["test"])
            assert ":8000" not in probe, f"{name} does not serve :8000: {probe}"
        assert "/health" in " ".join(services["mcp"]["healthcheck"]["test"])
        assert "inspect ping" in " ".join(services["celery-worker"]["healthcheck"]["test"])

    @pytest.mark.parametrize("backend", SERVER_BACKENDS)
    def test_every_service_gets_the_same_vector_backend(self, backend):
        services = _full_stack(backend)["services"]
        seen = {
            name: services[name]["environment"].get("RAG_VECTOR_BACKEND") for name in LONG_RUNNING
        }
        assert len(set(seen.values())) == 1 and None not in seen.values(), seen

    def test_milvus_lite_refuses_to_run_the_worker_and_mcp(self):
        services = _full_stack("milvus-lite")["services"]
        for name in ("celery-worker", "celery-beat", "mcp"):
            command = " ".join(services[name]["command"])
            assert "exit 1" in command and "COMPOSE_PROFILES" in command, name

    @pytest.mark.parametrize("profiles", ["celery,mcp", "celery", "mcp", " celery , mcp "])
    def test_validate_refuses_milvus_lite_with_celery_or_mcp(self, profiles):
        result = _make_validate("VECTOR_BACKEND=milvus-lite", f"COMPOSE_PROFILES={profiles}")
        assert result.returncode != 0
        out = result.stdout + result.stderr
        assert "milvus-lite runs inside the api process" in out
        assert "COMPOSE_PROFILES=" in out

    @pytest.mark.parametrize("profiles", ["", "tracing"])
    def test_validate_allows_milvus_lite_without_them(self, profiles):
        result = _make_validate("VECTOR_BACKEND=milvus-lite", f"COMPOSE_PROFILES={profiles}")
        assert result.returncode == 0, result.stdout

    def test_validate_requires_the_postgres_password(self):
        result = _make_validate("VECTOR_BACKEND=qdrant", "POSTGRES_PASSWORD=")
        assert result.returncode != 0
        assert "POSTGRES_PASSWORD" in result.stdout + result.stderr

    def test_the_example_env_passes_validation(self):
        """Copying .env.example and filling in the secrets must give a valid stack."""
        values = dict(
            line.split("=", 1)
            for line in (DEPLOY_DIR / ".env.example").read_text().splitlines()
            if line and not line.startswith("#") and "=" in line
        )
        assert values["VECTOR_BACKEND"] != "milvus-lite"
        result = _make_validate(
            f"VECTOR_BACKEND={values['VECTOR_BACKEND']}",
            f"COMPOSE_PROFILES={values['COMPOSE_PROFILES']}",
        )
        assert result.returncode == 0, result.stdout

    def test_containers_read_the_env_file(self):
        """Compose reads .env only for ${...}; LLM keys there never reached a container."""
        base = yaml.safe_load((DEPLOY_DIR / "docker-compose.yml").read_text())
        for name in LONG_RUNNING:
            env_file = base["services"][name].get("env_file")
            assert env_file and env_file[0]["path"] == ".env", name
            assert env_file[0]["required"] is False, name

    def test_the_milvus_service_can_start(self):
        milvus = _full_stack("milvus")["services"]["milvus"]
        assert not milvus["image"].endswith("-latest"), milvus["image"]
        assert milvus["command"] == ["milvus", "run", "standalone"]
        etcd = next(v for v in milvus["volumes"] if v["target"] == "/milvus/configs/embedEtcd.yaml")
        assert Path(etcd["source"]).is_file(), etcd

    def test_pip_version_specs_are_quoted(self):
        """`pip install pkg>=1.0` in a shell is `pip install pkg` with stdout to "=1.0"."""
        import re

        text = (DEPLOY_DIR / "Dockerfile").read_text().replace("\\\n", " ")
        pip_lines = [line for line in text.splitlines() if "pip install" in line]
        assert pip_lines
        for line in pip_lines:
            # The shell treats a > or < outside quotes as a redirection.
            unquoted = re.sub(r'"[^"]*"', "", line)
            assert not re.search(r"[<>]", unquoted), f"unquoted version spec: {line.strip()}"

    def test_the_shutdown_drain_fits_inside_the_kill_timeouts(self):
        """Killed before the drain ends, documents stay "processing" for good."""
        import re

        api = _full_stack("qdrant")["services"]["api"]
        drain = float(api["environment"]["RAG_PROCESSING_TIMEOUT"])
        graceful = float(
            re.search(r"--graceful-timeout (\d+)", (DEPLOY_DIR / "Dockerfile").read_text()).group(1)
        )
        grace = float(str(api["stop_grace_period"]).rstrip("s"))
        assert drain < graceful < grace, (drain, graceful, grace)

    def test_gunicorn_preloads_and_passes_the_trusted_proxies(self):
        text = (DEPLOY_DIR / "Dockerfile").read_text()
        assert "--preload" in text
        assert "--forwarded-allow-ips" in text and "FORWARDED_ALLOW_IPS" in text
        assert "/data/uploads" in text, "the uploads volume would be root-owned"


# ============================================================================
# Layer 2: Registry Empty-String Fix (unit tests)
# ============================================================================


class TestRegistryEmptyStringFix:
    """Validates that empty-string env vars don't trigger false config matches."""

    def test_empty_pgvector_dsn_ignored(self, monkeypatch):
        """PGVECTOR_DSN='' must not produce a config with empty dsn."""
        monkeypatch.setenv("PGVECTOR_DSN", "")
        monkeypatch.delenv("RAG_DATABASE_DSN", raising=False)
        from lasagna.vector_backends.registry import (
            _get_default_pgvector_config,
        )

        config = _get_default_pgvector_config()
        assert config.get("dsn", "") != "", (
            "Empty PGVECTOR_DSN should fall through to a non-empty default"
        )

    def test_empty_qdrant_url_ignored(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "")
        from lasagna.vector_backends.registry import (
            _get_default_qdrant_config,
        )

        config = _get_default_qdrant_config()
        assert config.get("url", None) != "", "Empty QDRANT_URL should fall through"

    def test_empty_weaviate_url_ignored(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_URL", "")
        from lasagna.vector_backends.registry import (
            _get_default_weaviate_config,
        )

        config = _get_default_weaviate_config()
        assert config.get("embedded") is True or config.get("url", None) != "", (
            "Empty WEAVIATE_URL should fall through to embedded mode"
        )

    def test_empty_milvus_uri_ignored(self, monkeypatch):
        monkeypatch.setenv("MILVUS_URI", "")
        from lasagna.vector_backends.registry import (
            _get_default_milvus_config,
        )

        config = _get_default_milvus_config()
        assert config.get("uri", None) != "", "Empty MILVUS_URI should fall through to db_path"


# ============================================================================
# Layer 3: Live Deployment E2E (requires Docker)
# ============================================================================


def _api_reachable() -> bool:
    """Check if the API is already running on localhost:8000."""
    try:
        import requests as _requests

        r = _requests.get(f"{BASE_URL}/health/live", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def live_service():
    """Skip live tests if the API is not reachable.

    Live tests expect the stack to be started externally via:
        VECTOR_BACKEND=milvus-lite make -C deploy up
    This avoids starting/stopping Docker inside pytest.
    """
    if not _api_reachable():
        pytest.skip(
            "Live API not reachable at localhost:8000. "
            "Start a stack first: VECTOR_BACKEND=<backend> make -C deploy up"
        )
    return True


@pytest.fixture(scope="module")
def session_headers():
    return {"X-Session-ID": f"test-unified-{uuid4().hex[:8]}"}


class TestLiveServiceHealth:
    """E2E: Health endpoints must work for the running backend."""

    def test_liveness(self, live_service):
        import requests as _requests

        r = _requests.get(f"{BASE_URL}/health/live", timeout=10)
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_readiness_all_checks_up(self, live_service):
        import requests as _requests

        r = _requests.get(f"{BASE_URL}/health/ready", timeout=10)
        assert r.status_code == 200, f"Readiness returned {r.status_code}: {r.text}"
        data = r.json()
        for name, check in data.get("checks", {}).items():
            assert check["status"] == "up", (
                f"Health check '{name}' is {check['status']}: {check.get('detail')}"
            )

    def test_health_has_expected_checks(self, live_service):
        import requests as _requests

        r = _requests.get(f"{BASE_URL}/health", timeout=10)
        checks = r.json().get("checks", {})
        for expected in ("embedding_model", "vector_backend", "database"):
            assert expected in checks, f"Missing health check: {expected}"

    def test_database_check_is_up(self, live_service):
        """The database health check must report 'up'."""
        import requests as _requests

        r = _requests.get(f"{BASE_URL}/health", timeout=10)
        checks = r.json()["checks"]
        assert checks["database"]["status"] == "up", (
            f"Database is {checks['database']['status']}: {checks['database'].get('detail')}"
        )

    def test_stats_endpoint(self, live_service):
        import requests as _requests

        r = _requests.get(f"{BASE_URL}/stats", timeout=10)
        assert r.status_code in (200, 503)


class TestLiveDocumentCycle:
    """E2E: Upload → process → search → delete must work."""

    def test_full_cycle(self, live_service, session_headers, tmp_path):
        import requests as _requests

        # Upload a document
        test_file = tmp_path / "test_unified_deploy.txt"
        test_file.write_text(
            "The unified deployment system supports multiple vector backends "
            "including Milvus, PgVector, Qdrant, Weaviate, and Pinecone. "
            "It uses Docker Compose base files with backend-specific fragments."
        )
        with open(test_file, "rb") as f:
            r = _requests.post(
                f"{BASE_URL}/documents",
                files={"file": ("test_unified_deploy.txt", f, "text/plain")},
                headers=session_headers,
                timeout=60,
            )
        assert r.status_code == 201, f"Upload failed: {r.status_code} {r.text}"
        doc_id = r.json()["document_id"]

        try:
            # Wait for processing
            _wait_for_document_ready(doc_id, session_headers)

            # Search
            r = _requests.post(
                f"{BASE_URL}/search",
                json={"query": "vector backends deployment", "top_k": 5},
                headers=session_headers,
                timeout=30,
            )
            assert r.status_code == 200, f"Search failed: {r.status_code} {r.text}"
            results = r.json().get("results", [])
            assert len(results) > 0, "Search returned no results"

        finally:
            # Always clean up — delete the document
            _requests.delete(
                f"{BASE_URL}/documents/{doc_id}",
                headers=session_headers,
                timeout=15,
            )

    def test_document_list(self, live_service, session_headers):
        import requests as _requests

        r = _requests.get(
            f"{BASE_URL}/documents",
            headers=session_headers,
            timeout=10,
        )
        assert r.status_code == 200


class TestLiveProfileIsolation:
    """E2E: Only the selected backend's containers should be running."""

    def test_no_unexpected_vector_services(self, live_service):
        """Check running containers for unexpected vector services."""
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            cwd=str(DEPLOY_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("Cannot list containers")

        # Parse the JSON lines output
        output = result.stdout.strip()
        if not output:
            pytest.skip("No containers running")

        # docker compose ps --format json outputs one JSON object per line
        containers = []
        for line in output.splitlines():
            line = line.strip()
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        service_names = {c.get("Service", "") for c in containers}
        vector_services = {"milvus", "qdrant", "weaviate"}
        running_vector = service_names & vector_services

        # We can't know which backend is running from inside the test,
        # but we can verify at most ONE vector service is present
        assert len(running_vector) <= 1, (
            f"Multiple vector services running: {running_vector}. "
            f"Only one backend should be active at a time."
        )


# ============================================================================
# Layer 4: Kustomize Overlay Validation
# ============================================================================


def _kubectl_available() -> bool:
    try:
        result = subprocess.run(
            ["kubectl", "version", "--client"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


K8S_DIR = DEPLOY_DIR / "k8s"
K8S_OVERLAYS = ["milvus", "pgvector", "qdrant"]


class TestKustomizeStructure:
    """Validates Kustomize directory structure."""

    def test_base_kustomization_exists(self):
        path = K8S_DIR / "base" / "kustomization.yaml"
        assert path.is_file(), f"Missing: {path}"

    @pytest.mark.parametrize("overlay", K8S_OVERLAYS)
    def test_overlay_kustomization_exists(self, overlay):
        path = K8S_DIR / "overlays" / overlay / "kustomization.yaml"
        assert path.is_file(), f"Missing: {path}"

    def test_base_has_core_resources(self):
        """Base kustomization must reference core K8s resources."""
        content = (K8S_DIR / "base" / "kustomization.yaml").read_text()
        for resource in ("deployment-api", "service", "configmap"):
            assert resource in content, f"Base kustomization missing reference to {resource}"


@pytest.mark.skipif(not _kubectl_available(), reason="kubectl not installed")
class TestKustomizeRendering:
    """Validates that Kustomize overlays render correctly."""

    @pytest.mark.parametrize("overlay", K8S_OVERLAYS)
    def test_overlay_renders(self, overlay):
        """kubectl kustomize must succeed."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(K8S_DIR / "overlays" / overlay)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Kustomize failed for {overlay}:\n{result.stderr}"

    @pytest.mark.parametrize("overlay", K8S_OVERLAYS)
    def test_overlay_sets_correct_backend(self, overlay):
        """Rendered configmap must set RAG_VECTOR_BACKEND correctly."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(K8S_DIR / "overlays" / overlay)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"Kustomize render failed: {result.stderr}")

        rendered = result.stdout
        # The configmap should contain the backend name
        assert (
            f'RAG_VECTOR_BACKEND: "{overlay}"' in rendered
            or f"RAG_VECTOR_BACKEND: {overlay}" in rendered
            or f'RAG_VECTOR_BACKEND: "{overlay}"' in rendered
        ), f"Overlay {overlay} did not set RAG_VECTOR_BACKEND correctly"
