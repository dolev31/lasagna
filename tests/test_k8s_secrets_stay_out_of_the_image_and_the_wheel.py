"""An operator's Kubernetes secrets stay out of git, the image and the wheel.

deploy/README.md told operators to replace the CHANGE_ME values in
k8s/base/secrets.yaml in place. The image build sent all of src/lasagna/ to
Docker, and the wheel's package data (``deploy/**/*``) is matched against the
file system, not git, so the JWT secret, the admin token, the API keys and the
database password went into every image and every wheel built from that
checkout. Operators now copy overlays/local.example to overlays/local/, which
git ignores, the Docker build leaves out
(test_docker_build_context_has_no_secrets.py) and the wheel and sdist exclude.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from tests._paths import DEPLOY_DIR

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
K8S = DEPLOY_DIR / "k8s"
LOCAL = "src/lasagna/deploy/k8s/overlays/local"
OPERATOR_VALUE = "an-operators-real-jwt-secret"


def _kubernetes_section() -> str:
    text = (DEPLOY_DIR / "README.md").read_text()
    return text.split("## Kubernetes", 1)[1].split("\n## ", 1)[0]


def test_the_readme_says_to_copy_the_template_not_to_edit_the_shipped_files():
    section = _kubernetes_section()
    assert "cp -r k8s/overlays/local.example k8s/overlays/local" in section
    assert "kubectl apply -k k8s/overlays/local/" in section
    edit_in_place = re.search(r"[Rr]eplace[^.]*`base/secrets\.yaml`", section)
    assert edit_in_place is None, edit_in_place.group(0)


def test_git_ignores_the_local_overlay_and_tracks_the_template():
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", f"{LOCAL}/secrets.yaml"], cwd=REPO_ROOT, check=False
    )
    assert ignored.returncode == 0, f"git does not ignore {LOCAL}/"
    tracked = subprocess.run(
        ["git", "ls-files", "src/lasagna/deploy/k8s/overlays/local.example"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert len(tracked) == 2, tracked


def test_the_package_data_exclusion_matches_the_local_overlay_only():
    """setuptools applies these with fnmatch, where `*` spans directories."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        patterns = tomllib.load(f)["tool"]["setuptools"]["exclude-package-data"]["lasagna"]

    def excluded(path):
        return any(fnmatch.fnmatch(path, p) for p in patterns)

    assert excluded("deploy/k8s/overlays/local/secrets.yaml")
    assert excluded("deploy/k8s/overlays/local/prod/kustomization.yaml")
    assert not excluded("deploy/k8s/base/secrets.yaml")
    assert not excluded("deploy/k8s/overlays/local.example/secrets.yaml")


def test_a_wheel_and_sdist_built_from_a_configured_checkout_leave_it_out(tmp_path):
    setuptools = pytest.importorskip("setuptools")
    if int(setuptools.__version__.split(".")[0]) < 77:
        pytest.skip("pyproject.toml needs setuptools >= 77")
    tree = tmp_path / "checkout"
    tree.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
        shutil.copy(REPO_ROOT / name, tree / name)
    package = tree / "src" / "lasagna"
    shutil.copytree(K8S, package / "deploy" / "k8s", ignore=shutil.ignore_patterns("local"))
    (package / "__init__.py").write_text("")
    local = package / "deploy" / "k8s" / "overlays" / "local"
    local.mkdir()
    (local / "secrets.yaml").write_text(f"RAG_JWT_SECRET: {OPERATOR_VALUE}\n")

    built = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools import build_meta as b; b.build_wheel('out'); b.build_sdist('out')",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    (wheel,) = (tree / "out").glob("*.whl")
    (sdist,) = (tree / "out").glob("*.tar.gz")
    names = zipfile.ZipFile(wheel).namelist() + tarfile.open(sdist).getnames()
    assert any(n.endswith("deploy/k8s/base/secrets.yaml") for n in names), "template missing"
    shipped = [n for n in names if "overlays/local/" in n]
    assert not shipped, f"the operator's overlay ships: {shipped}"


def _kubectl() -> bool:
    try:
        return (
            subprocess.run(["kubectl", "version", "--client"], capture_output=True).returncode == 0
        )
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _kubectl(), reason="kubectl not installed")
def test_the_template_overlay_keeps_the_base_values_it_does_not_set():
    """A strategic-merge patch, not a secretGenerator merge, which drops stringData."""
    rendered = subprocess.run(
        ["kubectl", "kustomize", str(K8S / "overlays" / "local.example")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    docs = list(yaml.safe_load_all(rendered))
    (secret,) = [d for d in docs if d and d["kind"] == "Secret"]
    values = secret["stringData"]
    assert values["RAG_REDIS_URL"] == "redis://redis.lasagna:6379"
    for key in ("RAG_JWT_SECRET", "RAG_ADMIN_TOKEN", "RAG_API_KEYS", "PGVECTOR_DSN"):
        assert "CHANGE_ME" in values[key], key
    spec = importlib.util.spec_from_file_location("check_manifests", K8S / "check_manifests.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    assert check.problems(docs) == []
