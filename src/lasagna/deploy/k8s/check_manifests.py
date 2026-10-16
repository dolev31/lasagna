"""Refuse Kubernetes manifests in which the Celery worker cannot open an upload.

    kubectl kustomize k8s/overlays/pgvector | python3 k8s/check_manifests.py

(``make validate-k8s`` from src/lasagna/deploy.) The api stores an upload under
the upload directory and hands its path to a worker in another pod. With
``RAG_CELERY_ENABLED`` on, both Deployments must mount one ReadWriteMany claim
at ``LASAGNA_UPLOAD_DIR``; otherwise every upload fails on the worker.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List

import yaml

CONFIG = "rag-knowledge-config"
DEPLOYMENTS = ("rag-knowledge-api", "rag-knowledge-celery")
TRUTHY = {"1", "true", "yes", "on"}


def _claims_at(deployment: Dict[str, Any], path: str) -> List[str]:
    spec = deployment["spec"]["template"]["spec"]
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


def problems(documents: Iterable[Dict[str, Any]]) -> List[str]:
    docs = [d for d in documents if d]
    data: Dict[str, str] = {}
    for d in docs:
        if d.get("kind") == "ConfigMap" and d["metadata"]["name"] == CONFIG:
            data.update(d.get("data") or {})
    if str(data.get("RAG_CELERY_ENABLED", "")).strip().lower() not in TRUTHY:
        return []
    upload_dir = data.get("LASAGNA_UPLOAD_DIR") or data.get("RAG_UPLOAD_DIR")
    if not upload_dir:
        return [
            f"{CONFIG} turns Celery on but sets no LASAGNA_UPLOAD_DIR, so the api and the "
            "worker store and read uploads in their own filesystems."
        ]
    deployments = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "Deployment"}
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


def main() -> int:
    found = problems(yaml.safe_load_all(sys.stdin))
    for line in found:
        print(f"ERROR: {line}", file=sys.stderr)
    if not found:
        print("Manifests OK: the api and the worker share the uploads volume.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
