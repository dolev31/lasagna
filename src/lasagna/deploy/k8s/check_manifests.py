"""Refuse Kubernetes manifests that kubectl accepts but that cannot work.

    kubectl kustomize k8s/overlays/local | python3 k8s/check_manifests.py

(``make validate-k8s`` from src/lasagna/deploy.) kubectl checks each object on
its own; this checks how they fit together:

* Uploads. With ``RAG_CELERY_ENABLED`` on, the api stores an upload and a
  worker in another pod opens it, so both Deployments mount one ReadWriteMany
  claim at ``LASAGNA_UPLOAD_DIR``; otherwise every upload fails on the worker.
* Memory. Each gunicorn worker (``WEB_CONCURRENCY``) and each Celery worker
  process (``--concurrency``) loads the models itself, on first use, after the
  fork: the image's ``--preload`` shares the imports, not the models. The
  pod's memory request and limit must hold that many copies, and an
  autoscaler's memory target must sit above them, or it is always exceeded and
  the Deployment stays at maxReplicas. A model not in ``MODEL_GIB`` is not
  sized.
* Garbage collection. With Celery on, the api leaves the periodic GC and
  orphan scans to Celery beat, so exactly one pod must run it.
* Shutdown. The api waits ``RAG_PROCESSING_TIMEOUT`` seconds for documents
  still processing, which must end inside gunicorn's ``--graceful-timeout``,
  which must end inside the pod's ``terminationGracePeriodSeconds``;
  otherwise the process is killed first and those documents stay
  "processing".
"""

from __future__ import annotations

import math
import re
import shlex
import sys
from typing import Any, Dict, Iterable, List, Optional

import yaml

CONFIG = "lasagna-config"
API = "lasagna-api"
WORKER = "lasagna-celery"
DEPLOYMENTS = (API, WORKER)
TRUTHY = {"1", "true", "yes", "on"}

#: What a container gets when it sets nothing: the Dockerfile's CMD and ENV,
#: and the code default of ``processing_timeout`` (core/config.py).
IMAGE_WEB_CONCURRENCY = 4
IMAGE_GRACEFUL_TIMEOUT = 30
IMAGE_EMBEDDING_MODEL = "BAAI/bge-m3"
IMAGE_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"
DEFAULT_PROCESSING_TIMEOUT = 300
#: Kubernetes' own default for terminationGracePeriodSeconds.
DEFAULT_GRACE_PERIOD = 30

#: GiB a process gains when it loads a model and runs it on the CPU (peak
#: resident size over the bare process, constraints/0.1.0.txt versions).
MODEL_GIB = {
    "BAAI/bge-m3": 2.2,
    "BAAI/bge-large-en-v1.5": 1.4,
    "BAAI/bge-base-en-v1.5": 0.5,
    "BAAI/bge-small-en-v1.5": 0.2,
    "sentence-transformers/all-MiniLM-L6-v2": 0.2,
    "all-MiniLM-L6-v2": 0.2,
    "mixedbread-ai/mxbai-rerank-large-v1": 1.6,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 0.3,
}
#: GiB one process holds before any model: Python, torch, the package and the
#: working memory of a request.
PROCESS_GIB = 1.0

_UNITS = {
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
    "k": 10**3,
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
}


def _gib(quantity: Any) -> Optional[float]:
    """A Kubernetes memory quantity ("6Gi", "512Mi", "2G", bytes) in GiB."""
    if quantity is None:
        return None
    match = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]*)\s*", str(quantity))
    if not match or (match[2] and match[2] not in _UNITS):
        return None
    return float(match[1]) * _UNITS.get(match[2], 1) / 2**30


def _setting(env: Dict[str, str], name: str, default: Any = None) -> Any:
    """A setting as the package reads it: ``LASAGNA_<NAME>``, then ``RAG_<NAME>``."""
    for key in (f"LASAGNA_{name}", f"RAG_{name}"):
        if env.get(key) not in (None, ""):
            return env[key]
    return default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def _pod(deployment: Dict[str, Any]) -> Dict[str, Any]:
    return deployment["spec"]["template"]["spec"]


def _container(deployment: Dict[str, Any]) -> Dict[str, Any]:
    return (_pod(deployment).get("containers") or [{}])[0]


def _env(container: Dict[str, Any], configmaps: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """The container's environment: its ConfigMaps, then its own ``env`` values."""
    env: Dict[str, str] = {}
    for source in container.get("envFrom") or []:
        env.update(configmaps.get((source.get("configMapRef") or {}).get("name"), {}))
    for item in container.get("env") or []:
        if "value" in item:
            env[item["name"]] = str(item["value"])
    return env


def _words(container: Dict[str, Any]) -> List[str]:
    """The container's command and args, a shell string split into words."""
    words: List[str] = []
    for word in (container.get("command") or []) + (container.get("args") or []):
        words += shlex.split(str(word)) if " " in str(word) else [str(word)]
    return words


def _option(words: List[str], *names: str) -> Optional[str]:
    for i, word in enumerate(words):
        for name in names:
            if word == name and i + 1 < len(words):
                return words[i + 1]
            if word.startswith(f"{name}="):
                return word.split("=", 1)[1]
    return None


def _int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _celery_processes(words: List[str]) -> Optional[int]:
    """The worker processes ``celery worker`` forks; a pool of threads is one."""
    if _option(words, "--pool", "-P") in ("solo", "threads", "eventlet", "gevent"):
        return 1
    return _int(_option(words, "--concurrency", "-c"))


def _runs_beat(deployment: Dict[str, Any]) -> bool:
    words = _words(_container(deployment))
    return "celery" in words and "beat" in words


def _uploads(docs, data, deployments) -> List[str]:
    if not _truthy(data.get("RAG_CELERY_ENABLED", "")):
        return []
    upload_dir = data.get("LASAGNA_UPLOAD_DIR") or data.get("RAG_UPLOAD_DIR")
    if not upload_dir:
        return [
            f"{CONFIG} turns Celery on but sets no LASAGNA_UPLOAD_DIR, so the api and the "
            "worker store and read uploads in their own filesystems."
        ]
    claimed = {
        name: _claims_at(deployments[name], upload_dir)
        for name in DEPLOYMENTS
        if name in deployments
    }
    out = [
        f"{name} mounts no PersistentVolumeClaim at {upload_dir} (LASAGNA_UPLOAD_DIR), so a "
        "worker cannot open what an api pod stored."
        for name, found in claimed.items()
        if not found
    ]
    shared = set.intersection(*(set(c) for c in claimed.values())) if claimed else set()
    if not out and not shared:
        out.append(f"{' and '.join(claimed)} mount different claims at {upload_dir}.")
    claims = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "PersistentVolumeClaim"}
    for name in sorted(shared):
        modes = (claims.get(name) or {}).get("spec", {}).get("accessModes") or []
        if "ReadWriteMany" not in modes:
            out.append(
                f"The claim {name} is {', '.join(modes) or 'not defined'}; pods on different "
                "nodes need ReadWriteMany."
            )
    return out


def _claims_at(deployment: Dict[str, Any], path: str) -> List[str]:
    spec = _pod(deployment)
    claims = {
        v["name"]: v["persistentVolumeClaim"]["claimName"]
        for v in spec.get("volumes") or []
        if "persistentVolumeClaim" in v
    }
    return [
        claims[m["name"]]
        for c in spec.get("containers") or []
        for m in c.get("volumeMounts") or []
        if m.get("mountPath", "").rstrip("/") == path.rstrip("/") and m["name"] in claims
    ]


def _footprint(name: str, deployment: Dict[str, Any], configmaps) -> Optional[Dict[str, Any]]:
    """How many processes load models in this pod, which models, and the GiB they hold."""
    container = _container(deployment)
    env = _env(container, configmaps)
    words = _words(container)
    models = [_setting(env, "EMBEDDING_MODEL", IMAGE_EMBEDDING_MODEL)]
    if name == API:
        if words and "gunicorn" not in words:
            return None
        processes = _int(
            _option(words, "--workers", "-w") or env.get("WEB_CONCURRENCY") or IMAGE_WEB_CONCURRENCY
        )
        what, knob = "gunicorn workers (WEB_CONCURRENCY)", "WEB_CONCURRENCY"
        if _truthy(_setting(env, "ENABLE_RERANKING", "true")):
            models.append(_setting(env, "RERANKER_MODEL", IMAGE_RERANKER_MODEL))
    else:
        # Indexing embeds; it does not rerank.
        processes = _celery_processes(words)
        what, knob = "Celery worker processes (--concurrency)", "--concurrency"
    if processes is None or any(m not in MODEL_GIB for m in models):
        return None
    each = PROCESS_GIB + sum(MODEL_GIB[m] for m in models)
    return {
        "processes": processes,
        "models": models,
        "each": each,
        "total": processes * each,
        "what": what,
        "knob": knob,
    }


def _memory(docs, deployments, configmaps) -> List[str]:
    out: List[str] = []
    worker = deployments.get(WORKER)
    if worker is not None and _celery_processes(_words(_container(worker))) is None:
        out.append(
            f"{WORKER} starts the Celery worker without --concurrency, so it forks one process "
            "per CPU of the node, not of the pod, and each loads the embedding model. Set "
            "--concurrency."
        )
    sized: Dict[str, Dict[str, Any]] = {}
    for name in DEPLOYMENTS:
        if name not in deployments:
            continue
        found = _footprint(name, deployments[name], configmaps)
        if found is None:
            continue
        resources = _container(deployments[name]).get("resources") or {}
        limit = (resources.get("limits") or {}).get("memory")
        request = (resources.get("requests") or {}).get("memory") or limit
        found["request"] = request
        sized[name] = found
        total, need = found["total"], math.ceil(found["total"])
        models = " and ".join(found["models"])
        if _gib(limit) is not None and _gib(limit) < total:
            out.append(
                f"{name} runs {found['processes']} {found['what']}, and each loads {models} "
                f"itself (about {found['each']:.1f} GiB), so the pod needs about {total:.1f} "
                f"GiB and its memory limit of {limit} kills it. Lower {found['knob']}, or raise "
                f"the limit to at least {need}Gi."
            )
        if _gib(request) is not None and _gib(request) < total:
            out.append(
                f"{name} requests {request} of memory but holds about {total:.1f} GiB once its "
                "models load, so it can be placed on a node without room for it and is "
                f"evicted first when the node runs short. Raise the request to at least {need}Gi."
            )
    for hpa in (d for d in docs if d.get("kind") == "HorizontalPodAutoscaler"):
        target = hpa["spec"].get("scaleTargetRef", {}).get("name")
        if target not in sized:
            continue
        found = sized[target]
        for metric in hpa["spec"].get("metrics") or []:
            resource = metric.get("resource") or {}
            if metric.get("type") != "Resource" or resource.get("name") != "memory":
                continue
            goal = resource.get("target") or {}
            if goal.get("type") == "AverageValue":
                threshold, said = _gib(goal.get("averageValue")), str(goal.get("averageValue"))
            else:
                percent = goal.get("averageUtilization") or 0
                request = _gib(found["request"])
                threshold = None if request is None else percent / 100 * request
                said = f"{percent}% of the {found['request']} request"
            if threshold is not None and threshold <= found["total"]:
                out.append(
                    f"The autoscaler {hpa['metadata']['name']} adds {target} pods when memory "
                    f"passes {said} ({threshold:.1f} GiB), but a pod holds about "
                    f"{found['total']:.1f} GiB once its models load, so it is always over and "
                    f"stays at maxReplicas ({hpa['spec'].get('maxReplicas')}). Scale on CPU."
                )
    return out


def _garbage_collection(docs, data, deployments) -> List[str]:
    if not _truthy(data.get("RAG_CELERY_ENABLED", "")):
        return []
    beats = [d for d in deployments.values() if _runs_beat(d)]
    if not beats:
        return [
            "Celery is on, so the api leaves garbage collection and orphan scans to Celery "
            "beat, but no Deployment runs `celery beat`: they never run. Add one with one "
            "replica (base/deployment-celery-beat.yaml)."
        ]
    out: List[str] = []
    replicas = sum(d["spec"].get("replicas", 1) for d in beats)
    if replicas != 1:
        out.append(
            f"{replicas} pods run celery beat ({', '.join(d['metadata']['name'] for d in beats)}); "
            "each schedules every task, so garbage collection runs that many times. Run one."
        )
    scaled = {d["metadata"]["name"] for d in beats}
    for hpa in (d for d in docs if d.get("kind") == "HorizontalPodAutoscaler"):
        if hpa["spec"].get("scaleTargetRef", {}).get("name") in scaled:
            out.append(
                f"The autoscaler {hpa['metadata']['name']} scales celery beat, which must run "
                "exactly one pod. Remove it."
            )
    for beat in beats:
        if (beat["spec"].get("strategy") or {}).get("type") != "Recreate":
            out.append(
                f"{beat['metadata']['name']} updates by rolling update, which starts the new "
                "celery beat before the old one stops, so for a while two schedule every task. "
                "Set its strategy type to Recreate."
            )
    return out


def _shutdown(deployments, configmaps) -> List[str]:
    api = deployments.get(API)
    if api is None:
        return []
    container = _container(api)
    words = _words(container)
    env = _env(container, configmaps)
    wait = _int(_setting(env, "PROCESSING_TIMEOUT", DEFAULT_PROCESSING_TIMEOUT))
    grace = _int(_pod(api).get("terminationGracePeriodSeconds", DEFAULT_GRACE_PERIOD))
    if wait is None or grace is None:
        return []
    if words and "gunicorn" not in words:
        stages = [("RAG_PROCESSING_TIMEOUT", wait), ("terminationGracePeriodSeconds", grace)]
    else:
        graceful = _int(_option(words, "--graceful-timeout") or IMAGE_GRACEFUL_TIMEOUT)
        stages = [
            ("RAG_PROCESSING_TIMEOUT", wait),
            ("gunicorn's --graceful-timeout", graceful),
            ("terminationGracePeriodSeconds", grace),
        ]
    out: List[str] = []
    for (inner, a), (outer, b) in zip(stages, stages[1:]):
        if a is None or b is None or a < b:
            continue
        fix = (
            f"Set terminationGracePeriodSeconds above {a}"
            if outer == "terminationGracePeriodSeconds"
            else f"Set {inner} below {b}"
        )
        out.append(
            f"{API} stops by waiting {inner} ({a} s) inside {outer} ({b} s), but the first "
            "does not end before the second, so the process is killed while documents are "
            f'still processing and they stay "processing". {fix}.'
        )
    return out


def problems(documents: Iterable[Dict[str, Any]]) -> List[str]:
    docs = [d for d in documents if d]
    configmaps = {
        d["metadata"]["name"]: {k: str(v) for k, v in (d.get("data") or {}).items()}
        for d in docs
        if d.get("kind") == "ConfigMap"
    }
    deployments = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "Deployment"}
    if CONFIG not in configmaps or not deployments:
        return [
            f"The input holds no {CONFIG} ConfigMap or no Deployment. Pipe in a rendered "
            "overlay: kubectl kustomize k8s/overlays/local | python3 k8s/check_manifests.py"
        ]
    data = configmaps[CONFIG]
    return (
        _uploads(docs, data, deployments)
        + _memory(docs, deployments, configmaps)
        + _garbage_collection(docs, data, deployments)
        + _shutdown(deployments, configmaps)
    )


def main() -> int:
    found = problems(yaml.safe_load_all(sys.stdin))
    for line in found:
        print(f"ERROR: {line}", file=sys.stderr)
    if not found:
        print("Manifests OK: uploads, memory, garbage collection and shutdown fit together.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
