"""The deployment guide starts from a clone and explains the two stacks' embedders.

Its quick start began with `cd src/lasagna/deploy`, never saying the stack
builds from a clone of the repository; and the Kubernetes ConfigMap embeds
with another model than compose, which the guide listed without a reason.
"""

from __future__ import annotations

import re

from tests._paths import PACKAGE_DIR

DEPLOY = PACKAGE_DIR / "deploy"


def _quick_start() -> str:
    guide = (DEPLOY / "README.md").read_text()
    return guide[guide.index("## Quick start") : guide.index("## Files")]


def test_the_quick_start_clones_the_repository_first():
    steps = _quick_start()
    assert steps.index("git clone https://github.com/dolev31/lasagna.git") < steps.index(
        "cd lasagna/src/lasagna/deploy"
    )


def test_the_guide_says_why_the_stacks_embed_differently():
    configmap = (DEPLOY / "k8s" / "base" / "configmap.yaml").read_text()
    compose = (DEPLOY / "docker-compose.yml").read_text()
    k8s_model = re.search(r'RAG_EMBEDDING_MODEL: "([^"]+)"', configmap).group(1)
    compose_model = re.search(r"RAG_EMBEDDING_MODEL:-([^}]+)}", compose).group(1)
    guide = " ".join((DEPLOY / "README.md").read_text().split())
    why = guide[guide.index("Why k8s embeds with another model than compose") :][:1200]
    assert k8s_model in why and compose_model in why
    assert "uploading it again" in why and "RAG_EMBEDDING_MODEL" in why
