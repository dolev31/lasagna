"""`lasagna doctor` never waits on the network, and its redaction names a temp dir once.

Two things from the last audit. A fresh-HOME offline `doctor` was reported to take
1:39; no check in it opens a connection, and this pins that: every socket
connect and DNS lookup would hang, and `doctor` must finish without touching one.
And `redact_paths` printed `<path>/private<tmp>/...` for a macOS temp dir when
TMPDIR was `/tmp`, because the temp root was replaced inside `/private/tmp/...`
and the leftover was then read as an absolute path.
"""

from __future__ import annotations

import socket
import time

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.core import provenance


class _Hung(Exception):
    pass


def test_doctor_opens_no_connection_and_finishes_quickly(monkeypatch):
    touched = []

    def hang(*args, **kwargs):
        touched.append(args[:2])
        raise _Hung("doctor tried to reach the network")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", hang)
    monkeypatch.setattr(socket, "getaddrinfo", hang)
    started = time.monotonic()
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert touched == [], touched
    assert result.exit_code in (0, 1), result.output
    assert "model_cache" in result.output
    assert time.monotonic() - started < 60


@pytest.mark.parametrize("tmpdir", ["/tmp", "/private/tmp"])
def test_a_temp_dir_prints_one_placeholder(monkeypatch, tmpdir):
    monkeypatch.setattr("tempfile.gettempdir", lambda: tmpdir)
    for path in ("/private/tmp/work/run.db", "/tmp/work/run.db"):
        shown = provenance.redact_paths(path)
        assert "private" not in shown, shown
        assert shown.count("<tmp>") == 1, shown
        assert shown.endswith("/work/run.db"), shown


def test_a_macos_per_user_temp_dir_prints_one_placeholder(monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: "/tmp")
    shown = provenance.redact_paths("/private/var/folders/xg/ab12/T/tmpx1/data")
    assert shown == "<tmp>/tmpx1/data", shown


def test_a_directory_that_only_starts_like_tmp_is_not_changed(monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: "/tmp")
    assert "<tmp>" not in provenance.redact_paths("/tmpfoo/x")
    assert "<tmp>" not in provenance.redact_paths("/a/tmp/x")
