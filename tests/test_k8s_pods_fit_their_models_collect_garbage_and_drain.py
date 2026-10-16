"""The Kubernetes defaults fit the models, run garbage collection and drain in order.

Four defects the manifests shipped with, none of which kubectl can see:

* Each of the image's 4 gunicorn workers loads the embedding model and the
  reranker itself, after the fork (``--preload`` shares imports, not models):
  about 16 GiB against the api's 6Gi limit, so the pod was killed. The Celery
  worker forked 4 processes, each with the embedding model, against 8Gi.
* The api autoscaler added pods at 80% of a 2Gi memory request, which a pod
  passes once its models load, so it sat at 10 replicas.
* No Celery beat ran, and with Celery on the api leaves garbage collection to
  beat: it never ran.
* ``RAG_PROCESSING_TIMEOUT`` was the code's 300 s, outliving gunicorn's 30 s
  graceful timeout and the pod's 45 s grace period.

``k8s/check_manifests.py`` (``make validate-k8s``) now refuses each of these.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
import shutil
import subprocess

import pytest
import yaml

from tests._paths import DEPLOY_DIR, PACKAGE_DIR

K8S = DEPLOY_DIR / "k8s"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_manifests", K8S / "check_manifests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shipped():
    resources = yaml.safe_load((K8S / "base" / "kustomization.yaml").read_text())["resources"]
    return [
        doc for name in resources for doc in yaml.safe_load_all((K8S / "base" / name).read_text())
    ]


@pytest.fixture
def docs(shipped):
    return copy.deepcopy(shipped)


def _named(docs, name, kind="Deployment"):
    return next(d for d in docs if d["kind"] == kind and d["metadata"]["name"] == name)


def _container(docs, name):
    return _named(docs, name)["spec"]["template"]["spec"]["containers"][0]


def _config(docs):
    return _named(docs, "lasagna-config", "ConfigMap")["data"]


def test_the_shipped_manifests_pass(check, docs):
    assert check.problems(docs) == []


def test_what_the_check_assumes_of_the_image_is_what_the_image_does(check):
    dockerfile = (DEPLOY_DIR / "Dockerfile").read_text()
    assert f"--graceful-timeout {check.IMAGE_GRACEFUL_TIMEOUT} " in dockerfile
    assert f'--workers "${{WEB_CONCURRENCY:-{check.IMAGE_WEB_CONCURRENCY}}}"' in dockerfile
    assert f'RAG_EMBEDDING_MODEL="{check.IMAGE_EMBEDDING_MODEL}"' in dockerfile
    assert f'RAG_RERANKER_MODEL="{check.IMAGE_RERANKER_MODEL}"' in dockerfile
    config = (PACKAGE_DIR / "core" / "config.py").read_text()
    default = re.search(r'"processing_timeout":\s*(\d+)', config)
    assert int(default[1]) == check.DEFAULT_PROCESSING_TIMEOUT


def test_the_image_default_of_four_workers_does_not_fit(check, docs):
    _container(docs, "lasagna-api").pop("env")
    found = check.problems(docs)
    assert any("runs 4 gunicorn workers" in p and "limit of 6Gi" in p for p in found), found


def test_a_memory_request_below_the_models_is_refused(check, docs):
    _container(docs, "lasagna-api")["resources"]["requests"]["memory"] = "2Gi"
    (problem,) = check.problems(docs)
    assert "lasagna-api requests 2Gi" in problem and "at least 4Gi" in problem


def test_a_larger_embedding_model_needs_more_memory(check, docs):
    _config(docs)["RAG_EMBEDDING_MODEL"] = "BAAI/bge-m3"
    found = check.problems(docs)
    assert any("lasagna-api requests 4Gi" in p for p in found), found
    assert any("lasagna-celery requests 5Gi" in p for p in found), found


def test_a_model_the_check_does_not_know_is_not_sized(check, docs):
    _config(docs)["RAG_EMBEDDING_MODEL"] = "example/unknown-model"
    assert check.problems(docs) == []


def test_four_celery_processes_do_not_fit(check, docs):
    command = _container(docs, "lasagna-celery")["command"]
    command[command.index("--concurrency=2")] = "--concurrency=4"
    found = check.problems(docs)
    assert any("runs 4 Celery worker processes" in p and "limit of 8Gi" in p for p in found)


def test_a_celery_worker_without_a_concurrency_is_refused(check, docs):
    container = _container(docs, "lasagna-celery")
    container["command"] = [w for w in container["command"] if "concurrency" not in w]
    (problem,) = check.problems(docs)
    assert "without --concurrency" in problem


def test_an_autoscaler_on_memory_the_models_alone_exceed_is_refused(check, docs):
    hpa = _named(docs, "lasagna-api", "HorizontalPodAutoscaler")
    hpa["spec"]["metrics"].append(
        {
            "type": "Resource",
            "resource": {
                "name": "memory",
                "target": {"type": "Utilization", "averageUtilization": 80},
            },
        }
    )
    (problem,) = check.problems(docs)
    assert "always over and stays at maxReplicas (10)" in problem


def test_without_celery_beat_garbage_collection_never_runs(check, docs):
    docs.remove(_named(docs, "lasagna-celery-beat"))
    (problem,) = check.problems(docs)
    assert "no Deployment runs `celery beat`" in problem


def test_beat_runs_exactly_one_pod_and_never_two_at_once(check, docs):
    beat = _named(docs, "lasagna-celery-beat")
    beat["spec"]["replicas"] = 2
    beat["spec"]["strategy"] = {"type": "RollingUpdate"}
    found = check.problems(docs)
    assert len(found) == 2, found
    assert "2 pods run celery beat" in found[0]
    assert "Recreate" in found[1]


def test_without_celery_no_beat_is_needed(check, docs):
    _config(docs)["RAG_CELERY_ENABLED"] = "false"
    docs.remove(_named(docs, "lasagna-celery-beat"))
    assert check.problems(docs) == []


def test_a_processing_timeout_longer_than_gunicorns_drain_is_refused(check, docs):
    del _config(docs)["RAG_PROCESSING_TIMEOUT"]
    (problem,) = check.problems(docs)
    assert "RAG_PROCESSING_TIMEOUT (300 s)" in problem and "below 30" in problem


def test_a_grace_period_shorter_than_gunicorns_drain_is_refused(check, docs):
    del _named(docs, "lasagna-api")["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]
    (problem,) = check.problems(docs)
    assert "terminationGracePeriodSeconds (30 s)" in problem and "above 30" in problem


def test_a_failed_render_is_refused_rather_than_passed(check):
    """`kubectl kustomize` that fails prints nothing; the check said "Manifests OK"."""
    (problem,) = check.problems([])
    assert "no lasagna-config ConfigMap" in problem


def _kubectl() -> bool:
    try:
        return (
            subprocess.run(["kubectl", "version", "--client"], capture_output=True).returncode == 0
        )
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _kubectl() or not shutil.which("make"), reason="needs kubectl and make")
def test_make_validate_k8s_fails_when_the_overlay_does_not_render(tmp_path):
    result = subprocess.run(
        ["make", "-s", "validate-k8s", "K8S_OVERLAY=does-not-exist"],
        cwd=DEPLOY_DIR,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")},
    )
    assert result.returncode != 0, result.stdout
    assert "Manifests OK" not in result.stdout
