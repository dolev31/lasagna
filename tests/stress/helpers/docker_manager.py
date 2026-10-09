"""Docker Compose lifecycle manager for stress tests.

Handles starting/stopping compose stacks per vector backend,
gating on /health/ready (NOT /health/live), and detecting runtime
backend info from the health response.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from ..config import StressConfig


@dataclass
class BackendInfo:
    """Runtime backend information detected from /health/ready."""

    vector_backend: str
    database_backend: str
    embedding_model: str
    celery_enabled: bool
    worker_mode: str  # "uvicorn-single" | "gunicorn-multi"
    startup_time_s: float


class DockerComposeManager:
    """Manages Docker Compose lifecycle for stress tests.

    Uses /health/ready for startup gating (NOT /health/live which only
    checks process aliveness without dependency verification).

    Handles teardown of a prior backend before starting the next one
    to prevent cross-backend contamination.
    """

    def __init__(self, config: StressConfig) -> None:
        self._base_config = config
        self._backend_config: Optional[StressConfig] = None
        self._current_backend: Optional[str] = None
        self._backend_info: Optional[BackendInfo] = None

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        return self._backend_info

    @property
    def backend_config(self) -> StressConfig:
        """Return the backend-specific config (after start() is called)."""
        if self._backend_config is None:
            return self._base_config
        return self._backend_config

    def _compose_cmd(self, backend: str) -> List[str]:
        """Build the docker compose command with correct file fragments."""
        cmd = [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            f"backends/{backend}.yml",
        ]
        return cmd

    def _profile_args(self, config: StressConfig) -> List[str]:
        """Add --profile flags for celery, etc."""
        args = []
        for profile in config.compose_profiles:
            args.extend(["--profile", profile])
        return args

    async def start(self, backend: str) -> BackendInfo:
        """Start compose stack and wait for /health/ready.

        Optimized: if the same backend is already running and healthy,
        skips teardown/restart entirely (saves ~2-3 min per module).

        Args:
            backend: Vector backend name (pgvector, qdrant, milvus, etc.)

        Returns:
            BackendInfo with detected runtime configuration.

        Raises:
            TimeoutError: If /health/ready doesn't return 200 within ready_timeout.
        """
        # Resolve backend-specific config (local var survives stop() which clears self._backend_config)
        backend_config = self._base_config.for_backend(backend)
        self._backend_config = backend_config

        # Fast path: same backend already running and healthy → skip restart
        if self._current_backend == backend and self._backend_info is not None:
            if await self._is_healthy(backend_config):
                logger.info(f"Backend {backend} already running and healthy — reusing")
                return self._backend_info

        # Tear down prior backend if switching (prevents cross-backend contamination)
        if self._current_backend and self._current_backend != backend:
            logger.info(
                f"Tearing down prior backend={self._current_backend} "
                f"before starting backend={backend}"
            )
            await self.stop(keep_volumes=False)

        # Restore after stop() which clears self._backend_config
        self._backend_config = backend_config

        logger.info(f"Starting compose stack for backend={backend}")
        self._current_backend = backend

        # Clean down -v before starting to remove stale state.
        # Only needed on first start or backend switch (not on reuse).
        self._ensure_clean_state(backend)

        # Build compose up command with backend-specific profile args
        cmd = self._compose_cmd(backend) + self._profile_args(self._backend_config) + ["up", "-d"]

        t0 = time.monotonic()

        # Run docker compose up with retry (handles race conditions after down -v)
        last_err = None
        for attempt in range(3):
            if attempt > 0:
                logger.info(f"Retrying docker compose up (attempt {attempt + 1}/3)...")
                time.sleep(3)
                self._ensure_clean_state(backend)

            proc = subprocess.run(
                cmd,
                cwd=self._backend_config.compose_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode == 0:
                last_err = None
                break
            last_err = (
                f"docker compose up failed (rc={proc.returncode}):\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
            logger.warning(f"Attempt {attempt + 1} failed: {proc.stderr[:200]}")

        if last_err:
            raise RuntimeError(last_err)

        logger.info("Compose stack started, waiting for /health/ready...")

        # Poll /health/ready with fast backoff
        health_url = f"{self._backend_config.api_base_url}/api/v1/health/ready"
        await self._wait_for_ready(health_url, self._backend_config.ready_timeout)

        # Warm up all gunicorn workers
        await self._warmup_workers(self._backend_config)

        # Warm up upload pipeline (collection creation, chunker init, etc.)
        await self._warmup_pipeline(self._backend_config)

        startup_time = time.monotonic() - t0

        # Detect backend info from /health/ready
        self._backend_info = await self._detect_backend_info(startup_time, self._backend_config)
        logger.info(
            f"Backend ready: vector={self._backend_info.vector_backend}, "
            f"db={self._backend_info.database_backend}, "
            f"model={self._backend_info.embedding_model}, "
            f"startup={startup_time:.1f}s"
        )
        return self._backend_info

    async def _is_healthy(self, config: StressConfig) -> bool:
        """Quick health check — returns True if /health/ready returns 200 with all checks up."""
        url = f"{config.api_base_url}/api/v1/health/ready"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    checks = data.get("checks", {})
                    return all(c.get("status") == "up" for c in checks.values())
        except Exception:
            return False

    async def stop(self, keep_volumes: bool = False) -> None:
        """Stop the current compose stack."""
        if not self._current_backend:
            return

        logger.info(f"Stopping compose stack for backend={self._current_backend}")
        config = self._backend_config or self._base_config
        cmd = self._compose_cmd(self._current_backend) + self._profile_args(config)
        cmd += ["down"]
        if not keep_volumes:
            cmd.append("-v")

        proc = subprocess.run(
            cmd,
            cwd=config.compose_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            logger.warning(f"docker compose down returned rc={proc.returncode}: {proc.stderr}")

        self._current_backend = None
        self._backend_info = None
        self._backend_config = None

    def _ensure_clean_state(self, backend: str) -> None:
        """Ensure no stale containers or volumes from prior module's stack.

        Each test module has its own DockerComposeManager. The previous module's
        manager runs ``stop()`` in teardown, but there can be race conditions
        with volume cleanup. This explicit ``down -v`` guarantees a clean slate.
        """
        config = self._backend_config or self._base_config
        cmd = self._compose_cmd(backend) + self._profile_args(config) + ["down", "-v"]
        try:
            result = subprocess.run(
                cmd,
                cwd=config.compose_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                logger.debug(f"Pre-start cleanup completed for backend={backend}")
            else:
                logger.warning(
                    f"Pre-start cleanup failed (rc={result.returncode}): {result.stderr}"
                )
        except Exception as exc:
            logger.warning(f"Pre-start cleanup failed: {exc}")

    def _is_api_running(self, backend: str) -> bool:
        """Check if the API container is already running."""
        config = self._backend_config or self._base_config
        try:
            result = subprocess.run(
                self._compose_cmd(backend) + ["ps", "-q", "api"],
                cwd=config.compose_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _clear_prometheus_multiproc(self, backend: str) -> None:
        """Restart the API container to reset Prometheus multiprocess counters.

        When reusing a running stack, gunicorn workers' mmap'd counter files
        accumulate values from prior test runs. Simply deleting the files
        doesn't help because workers hold mmap handles. Restarting the API
        container triggers the CMD cleanup (rm -rf /tmp/prometheus_multiproc/*)
        and spawns fresh workers with fresh counter files.
        """
        config = self._backend_config or self._base_config
        try:
            result = subprocess.run(
                self._compose_cmd(backend) + ["restart", "api"],
                cwd=config.compose_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                logger.debug("Restarted API container (fresh Prometheus counters)")
            else:
                logger.warning(f"API restart failed: {result.stderr}")
        except Exception as exc:
            logger.warning(f"API restart failed: {exc}")

    async def _wait_for_ready(self, url: str, timeout: float) -> None:
        """Poll /health/ready until 200 or timeout.

        Uses fast backoff: 1s, 2s, 4s, 8s, then 8s intervals.
        """
        deadline = time.monotonic() + timeout
        delay = 1.0
        max_delay = 8.0

        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Verify all checks are "up"
                            checks = data.get("checks", {})
                            all_up = all(c.get("status") == "up" for c in checks.values())
                            if all_up:
                                return
                            logger.debug(
                                f"Health ready returned 200 but not all checks up: {checks}"
                            )
                        else:
                            logger.debug(f"Health ready returned {resp.status}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    logger.debug(f"Health ready poll failed: {exc}")

                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

        raise TimeoutError(
            f"/health/ready did not return 200 within {timeout}s. "
            f"Check docker compose logs for errors."
        )

    async def _warmup_workers(self, config: StressConfig) -> None:
        """Trigger lazy initialization of all gunicorn workers.

        Gunicorn workers load the RAG backend (embedding model, reranker) on first
        request, which takes ~10-15s. Without warmup, early test requests hitting
        uninitialized workers receive 500 errors.

        Strategy:
        1. Fire num_burst concurrent requests to trigger init in all workers
        2. Verify with repeated concurrent bursts (not sequential polls, which
           may hit the same worker) until all burst responses return 200
        """
        url = f"{config.api_base_url}/api/v1/health/ready"
        num_burst = 8  # More than typical WEB_CONCURRENCY (4) to hit all workers

        logger.info(f"Warming up workers with {num_burst}-request bursts...")

        async with aiohttp.ClientSession() as session:
            # Phase 1: Fire concurrent requests to trigger initialization
            tasks = [
                session.get(url, timeout=aiohttp.ClientTimeout(total=120)) for _ in range(num_burst)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            ready_count = sum(
                1 for r in responses if not isinstance(r, Exception) and r.status == 200
            )
            for r in responses:
                if not isinstance(r, Exception):
                    r.release()

            logger.debug(f"Warmup initial burst: {ready_count}/{num_burst} ready")

            # Phase 2: Verify ALL workers ready with concurrent bursts.
            # Sequential polling can hit the same worker repeatedly due to
            # gunicorn round-robin timing. Concurrent bursts ensure we touch
            # all workers.
            deadline = time.monotonic() + 60
            all_ready = False
            while not all_ready and time.monotonic() < deadline:
                await asyncio.sleep(1.0)
                burst = [
                    session.get(url, timeout=aiohttp.ClientTimeout(total=30))
                    for _ in range(num_burst)
                ]
                burst_results = await asyncio.gather(*burst, return_exceptions=True)
                ok_count = sum(
                    1 for r in burst_results if not isinstance(r, Exception) and r.status == 200
                )
                for r in burst_results:
                    if not isinstance(r, Exception):
                        r.release()
                all_ready = ok_count == num_burst
                if not all_ready:
                    logger.debug(f"Warmup burst: {ok_count}/{num_burst} ready, retrying...")

        logger.info(f"Worker warmup complete (all {num_burst} burst responses OK)")

    async def _warmup_pipeline(self, config: StressConfig) -> None:
        """Verify the full upload pipeline works before real tests begin.

        After workers pass health checks, the upload pipeline (collection creation,
        chunking, embedding) may still fail on first use. This method does a test
        upload+delete cycle with retries, which forces all workers to initialize
        the vector backend collection and chunking pipeline.
        """
        import tempfile
        import os

        base_url = config.api_base_url
        session_id = "__warmup_pipeline__"

        logger.info("Warming up upload pipeline...")

        # Create a small test file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="warmup_", delete=False)
        tmp.write("Pipeline warmup test document for stress test initialization.")
        tmp.close()

        doc_id = None
        async with aiohttp.ClientSession() as session:
            # Retry upload up to 10 times (collection creation race, model loading)
            for attempt in range(10):
                try:
                    data = aiohttp.FormData()
                    data.add_field(
                        "file",
                        open(tmp.name, "rb"),
                        filename="warmup.txt",
                        content_type="text/plain",
                    )
                    async with session.post(
                        f"{base_url}/api/v1/documents",
                        headers={"X-Session-ID": session_id},
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        if resp.status in (200, 201):
                            result = await resp.json()
                            doc_id = result.get("document_id")
                            logger.info(f"Pipeline warmup upload succeeded (attempt {attempt + 1})")
                            break
                        else:
                            body = await resp.text()
                            logger.debug(
                                f"Pipeline warmup upload attempt {attempt + 1}: "
                                f"status={resp.status} body={body[:200]}"
                            )
                except Exception as exc:
                    logger.debug(f"Pipeline warmup upload attempt {attempt + 1} failed: {exc}")

                await asyncio.sleep(2)

            # Clean up: delete the warmup document
            if doc_id:
                try:
                    async with session.delete(
                        f"{base_url}/api/v1/documents/{doc_id}",
                        headers={"X-Session-ID": session_id},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        logger.debug(f"Pipeline warmup cleanup: status={resp.status}")
                except Exception:
                    pass  # Best effort cleanup

        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        if doc_id is None:
            logger.warning(
                "Pipeline warmup upload failed after 10 attempts. "
                "Test uploads may fail on first use."
            )
        else:
            logger.info("Pipeline warmup complete — upload pipeline verified")

    async def _detect_backend_info(self, startup_time: float, config: StressConfig) -> BackendInfo:
        """Extract runtime info from /health/ready response.

        Uses checks.vector_backend.detail for backend name (NOT /stats.vector_backend
        which is unpopulated per rag_backend.get_stats()).
        """
        url = f"{config.api_base_url}/api/v1/health/ready"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

        checks: Dict = data.get("checks", {})

        vector_check = checks.get("vector_backend", {})
        embedding_check = checks.get("embedding_model", {})
        db_check = checks.get("database", {})

        return BackendInfo(
            vector_backend=vector_check.get("detail", "unknown"),
            database_backend=db_check.get("detail", "unknown"),
            embedding_model=embedding_check.get("detail", "unknown"),
            celery_enabled=config.celery_enabled,
            worker_mode=config.worker_mode,
            startup_time_s=round(startup_time, 2),
        )

    async def get_container_stats(self) -> Dict[str, Dict]:
        """Get docker stats for memory/CPU monitoring (best-effort).

        Returns dict of container_name -> {memory_mb, cpu_percent}.
        """
        if not self._current_backend:
            return {}

        try:
            proc = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return {}

            stats = {}
            for line in proc.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    name = parts[0]
                    mem_str = parts[1].split("/")[0].strip()
                    cpu_str = parts[2].replace("%", "").strip()
                    stats[name] = {
                        "memory_usage": mem_str,
                        "cpu_percent": float(cpu_str) if cpu_str else 0.0,
                    }
            return stats
        except Exception:
            return {}
