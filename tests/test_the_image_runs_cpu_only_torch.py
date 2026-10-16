"""The image installs torch's CPU-only build unless TORCH_VARIANT=cuda asks for CUDA.

The image was about 11.5 GB: torch from PyPI on Linux (arm64 and x86_64) pulls
the CUDA libraries, several GB the CPU-only containers never use. Each stage
now installs the CPU build from the PyTorch index, at the version the
constraints pin, before anything that depends on torch, and the constrained
install keeps it. ``TORCH_VARIANT=cuda`` keeps the PyPI build for GPU hosts.

The RUN steps are executed here with a stand-in ``pip`` and ``torch``, so the
test checks what the shell does with each value, not only the text.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests._paths import DEPLOY_DIR

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = (DEPLOY_DIR / "Dockerfile").read_text()
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _stages() -> dict:
    """Each stage's RUN and ARG instructions, continuation lines joined, by stage name."""
    joined = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    stages: dict = {}
    current = None
    for line in joined.splitlines():
        match = re.match(r"FROM \S+ AS (\S+)", line)
        if match:
            current = stages.setdefault(match[1], [])
        elif current is not None and line.startswith(("RUN ", "ARG ")):
            current.append(line)
    return stages


def _constraints() -> str:
    version = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)[1]
    return (REPO / "constraints" / f"{version}.txt").read_text()


def _pinned_torch() -> str:
    return re.search(r"^torch==\S+", _constraints(), re.M)[0]


def _torch_step(stage: str) -> str:
    steps = [s for s in _stages()[stage] if s.startswith("RUN ") and CPU_INDEX in s]
    assert len(steps) == 1, steps
    return steps[0][len("RUN ") :]


def _run(command: str, tmp_path: Path, **env) -> subprocess.CompletedProcess:
    """Run a RUN step with /bin/sh, as Docker does, with a pip that records its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    pip = bin_dir / "pip"
    pip.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path / "pip-calls"}\n')
    pip.chmod(0o755)
    (bin_dir / "python").unlink(missing_ok=True)
    (bin_dir / "python").symlink_to(sys.executable)
    (tmp_path / "constraints.txt").write_text(_constraints())
    command = command.replace("/tmp/constraints.txt", "constraints.txt")
    return subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), **env},
        capture_output=True,
        text=True,
    )


def _pip_calls(tmp_path: Path) -> list:
    calls = tmp_path / "pip-calls"
    return calls.read_text().splitlines() if calls.exists() else []


def test_cpu_is_the_default_variant():
    assert "ARG TORCH_VARIANT=cpu" in _stages()["production"]


@pytest.mark.parametrize("stage", ["models", "production"])
def test_each_stage_installs_torch_before_the_constrained_install(stage):
    steps = _stages()[stage]
    torch_at = next(i for i, s in enumerate(steps) if CPU_INDEX in s)
    constrained_at = next(i for i, s in enumerate(steps) if re.search(r"pip install .*-c ", s))
    assert torch_at < constrained_at, steps


@pytest.mark.parametrize("stage", ["models", "production"])
def test_the_cpu_build_is_the_pinned_version_without_its_dependencies(stage, tmp_path):
    run = _run(_torch_step(stage), tmp_path, TORCH_VARIANT="cpu")
    assert run.returncode == 0, run.stdout + run.stderr
    assert _pip_calls(tmp_path) == [
        f"install --no-cache-dir --no-deps --index-url {CPU_INDEX} {_pinned_torch()}"
    ]


def test_the_models_stage_uses_the_cpu_build_for_a_cuda_image(tmp_path):
    # It only downloads the models and loads them once, with no GPU at build time.
    run = _run(_torch_step("models"), tmp_path, TORCH_VARIANT="cuda")
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(_pip_calls(tmp_path)) == 1


def test_cuda_leaves_torch_to_the_constrained_install(tmp_path):
    run = _run(_torch_step("production"), tmp_path, TORCH_VARIANT="cuda")
    assert run.returncode == 0, run.stdout + run.stderr
    assert _pip_calls(tmp_path) == []


def test_an_unknown_variant_fails_the_build_and_names_the_choices(tmp_path):
    run = _run(_torch_step("production"), tmp_path, TORCH_VARIANT="gpu")
    assert run.returncode != 0
    assert "Unknown TORCH_VARIANT=gpu. Use cpu (the default) or cuda." in run.stdout
    assert _pip_calls(tmp_path) == []


def _fake_torch(tmp_path: Path, cuda) -> dict:
    package = tmp_path / "fake" / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        f"__version__ = '2.14.1'\nversion = SimpleNamespace(cuda={cuda!r})\n"
    )
    return {"PYTHONPATH": str(tmp_path / "fake")}


def _check_step() -> str:
    steps = [s for s in _stages()["production"] if "torch.version.cuda" in s]
    assert len(steps) == 1, steps
    production = _stages()["production"]
    assert production.index(steps[0]) > next(
        i for i, s in enumerate(production) if re.search(r"pip install .*-c ", s)
    )
    return steps[0][len("RUN ") :]


def test_a_cpu_image_whose_torch_was_replaced_by_a_cuda_build_fails(tmp_path):
    run = _run(_check_step(), tmp_path, TORCH_VARIANT="cpu", **_fake_torch(tmp_path, "13.0"))
    assert run.returncode != 0
    assert "replaced the CPU build of torch with 2.14.1 (CUDA 13.0)" in run.stderr


def test_a_cpu_image_with_the_cpu_build_passes_the_check(tmp_path):
    run = _run(_check_step(), tmp_path, TORCH_VARIANT="cpu", **_fake_torch(tmp_path, None))
    assert run.returncode == 0, run.stderr


def test_a_cuda_image_is_not_checked(tmp_path):
    run = _run(_check_step(), tmp_path, TORCH_VARIANT="cuda", **_fake_torch(tmp_path, "13.0"))
    assert run.returncode == 0, run.stderr


def test_compose_builds_with_the_variant_from_env():
    compose = yaml.safe_load((DEPLOY_DIR / "docker-compose.yml").read_text())
    assert compose["x-build"]["args"]["TORCH_VARIANT"] == "${TORCH_VARIANT:-cpu}"
    assert "TORCH_VARIANT=" in (DEPLOY_DIR / ".env.example").read_text()
