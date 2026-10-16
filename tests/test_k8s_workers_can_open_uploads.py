"""The Kubernetes manifests let a Celery worker open what an api pod stored.

The base turned Celery on, and the api and worker Deployments mounted no
volume: the api stored each upload in its own container filesystem and handed
the path to a worker in another pod, so every upload failed there. Both now
mount one ReadWriteMany claim at ``LASAGNA_UPLOAD_DIR``, and
``k8s/check_manifests.py`` (``make validate-k8s``) refuses manifests that do not.
"""

from __future__ import annotations

import importlib.util
import subprocess

import pytest
import yaml

from tests._paths import DEPLOY_DIR

K8S = DEPLOY_DIR / "k8s"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_manifests", K8S / "check_manifests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base():
    resources = yaml.safe_load((K8S / "base" / "kustomization.yaml").read_text())["resources"]
    return [
        doc for name in resources for doc in yaml.safe_load_all((K8S / "base" / name).read_text())
    ]


def test_the_base_shares_the_uploads_between_the_api_and_the_worker(check):
    docs = _base()
    assert check.problems(docs) == []
    claim = next(d for d in docs if d["kind"] == "PersistentVolumeClaim")
    assert claim["spec"]["accessModes"] == ["ReadWriteMany"]


def test_a_worker_without_the_volume_is_refused(check):
    docs = _base()
    worker = next(d for d in docs if d["metadata"]["name"] == "lasagna-celery")
    del worker["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]

    (problem,) = check.problems(docs)
    assert "lasagna-celery mounts no PersistentVolumeClaim at /data/uploads" in problem


def test_a_read_write_once_claim_is_refused(check):
    docs = _base()
    claim = next(d for d in docs if d["kind"] == "PersistentVolumeClaim")
    claim["spec"]["accessModes"] = ["ReadWriteOnce"]

    (problem,) = check.problems(docs)
    assert "ReadWriteMany" in problem


def test_without_celery_nothing_needs_sharing(check):
    docs = [d for d in _base() if d["kind"] != "PersistentVolumeClaim"]
    next(d for d in docs if d["kind"] == "ConfigMap")["data"]["RAG_CELERY_ENABLED"] = "false"
    assert check.problems(docs) == []


def _kubectl() -> bool:
    try:
        return (
            subprocess.run(["kubectl", "version", "--client"], capture_output=True).returncode == 0
        )
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _kubectl(), reason="kubectl not installed")
@pytest.mark.parametrize("overlay", ["milvus", "pgvector", "pinecone", "qdrant", "weaviate"])
def test_every_overlay_passes_the_check(check, overlay):
    rendered = subprocess.run(
        ["kubectl", "kustomize", str(K8S / "overlays" / overlay)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert check.problems(yaml.safe_load_all(rendered)) == []
