"""The k8s worker's liveness probe asks this pod's worker, not any worker.

It ran ``celery inspect ping`` without ``-d``, which broadcasts: a reply from
any worker in the cluster passed it, so a stuck worker stayed up as long as
another pod's answered. It now pings ``celery@$HOSTNAME``, the node name a
worker started without ``-n`` takes, through a shell so ``$HOSTNAME`` expands.
The probe also imports the package (about 5 s) before it waits for the reply,
so its timeout leaves room for both.
"""

from __future__ import annotations

import os
import subprocess

import pytest
import yaml

from tests._paths import DEPLOY_DIR


@pytest.fixture(scope="module")
def worker():
    deployment = yaml.safe_load(
        (DEPLOY_DIR / "k8s" / "base" / "deployment-celery.yaml").read_text()
    )
    return deployment["spec"]["template"]["spec"]["containers"][0]


def test_the_worker_keeps_the_default_node_name(worker):
    assert not any(
        w in ("-n", "--hostname") or w.startswith("--hostname=") for w in worker["command"]
    )


def test_the_probe_pings_the_node_named_after_this_pod(worker, tmp_path):
    command = worker["livenessProbe"]["exec"]["command"]
    assert command[:2] == ["sh", "-c"], "a list command does not expand $HOSTNAME"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    seen = tmp_path / "args"
    fake = bin_dir / "celery"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {seen}\n')
    fake.chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin", "HOSTNAME": "lasagna-celery-7d9f-x2k4q"}
    subprocess.run(command, env=env, check=True, timeout=10)
    args = seen.read_text().split()
    assert args[args.index("-d") + 1] == "celery@lasagna-celery-7d9f-x2k4q", args
    assert args[args.index("inspect") + 1] == "ping"


def test_the_probe_has_time_to_import_the_package_and_wait_for_the_reply(worker):
    probe = worker["livenessProbe"]
    words = probe["exec"]["command"][-1].split()
    ping_timeout = int(words[words.index("--timeout") + 1])
    assert probe["timeoutSeconds"] >= ping_timeout + 5
