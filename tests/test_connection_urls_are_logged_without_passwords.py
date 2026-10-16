"""Connection URLs reach the logs with their password replaced by ``***``.

The Redis cache and the Redis session store logged ``redis://:<password>@host``
at INFO on every start, after the broker and database lines had been fixed;
the Milvus, Qdrant and Weaviate URLs and the SDK's server URL were logged the
same way. Every logger call that formats a URL, URI or DSN goes through
``redact_url``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

import lasagna

PACKAGE = Path(lasagna.__file__).parent
SECRET = "s3cret-pw"  # pragma: allowlist secret
LEVELS = {"trace", "debug", "info", "success", "warning", "error", "critical", "exception"}
WORDS = ("url", "uri", "dsn", "broker")


@pytest.fixture
def logs():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    yield lines
    logger.remove(sink)


@pytest.fixture
def fake_redis(monkeypatch):
    client = SimpleNamespace(ping=lambda: True)
    module = SimpleNamespace(Redis=SimpleNamespace(from_url=lambda url, **kw: client))
    monkeypatch.setitem(sys.modules, "redis", module)


def test_the_redis_cache_logs_its_url_without_the_password(fake_redis, logs):
    from lasagna.infrastructure.cache.backend import RedisCacheBackend

    RedisCacheBackend(f"redis://:{SECRET}@cache:6379/0")

    said = "".join(logs)
    assert "cache:6379" in said and SECRET not in said


def test_the_redis_session_store_logs_its_url_without_the_password(fake_redis, logs):
    from lasagna.infrastructure.session_store import RedisSessionStore

    RedisSessionStore(f"redis://app:{SECRET}@sessions:6379/1")

    said = "".join(logs)
    assert "sessions:6379" in said and SECRET not in said


@pytest.mark.parametrize(
    "dsn, shown",
    [
        (
            f"host=db user=rag password={SECRET} dbname=rag",
            "host=db user=rag password=*** dbname=rag",
        ),
        (f"host=db password='{SECRET} x' port=5432", "host=db password=*** port=5432"),
        ("/data/uploads/a.pdf", "/data/uploads/a.pdf"),
        ("s3://bucket/key.pdf", "s3://bucket/key.pdf"),
    ],
)
def test_a_key_value_dsn_and_a_plain_location_are_shown_safely(dsn, shown):
    from lasagna.core.provenance import redact_url

    assert redact_url(dsn) == shown


def _unredacted(node: ast.Call) -> list:
    """The URL-like values a logger call formats without redact_url."""
    values = []
    for arg in list(node.args) + [k.value for k in node.keywords]:
        parts = arg.values if isinstance(arg, ast.JoinedStr) else [arg]
        for part in parts:
            expr = part.value if isinstance(part, ast.FormattedValue) else part
            if isinstance(expr, ast.Constant):
                continue
            text = ast.unparse(expr)
            if any(w in text.lower() for w in WORDS) and "redact" not in text:
                values.append(text)
    return values


def test_no_logger_call_formats_a_url_without_redacting_it():
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LEVELS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                for value in _unredacted(node):
                    found.append(f"{path.relative_to(PACKAGE)}:{node.lineno}: {value}")
    assert not found, "log lines that can carry a password:\n" + "\n".join(found)
