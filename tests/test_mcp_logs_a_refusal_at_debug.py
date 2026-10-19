"""An MCP tool's refusal is logged at DEBUG, not as an ERROR with a traceback.

`patch_rag_config` with `{"chunk_size": 999999}` answered correctly, and the
server's stderr also got "ERROR | patch_rag_config failed: ..." and a traceback
ending in `SessionSettingRefused`. A refusal is the tool's answer, and its
text names the fix. The same went for `DocumentsLeftBehind`.
"""

from __future__ import annotations

import pytest
from loguru import logger

from lasagna.api._profile_guard import DocumentsLeftBehind, EmbeddingSizeUnverified
from lasagna.api._profile_switch import ProfileCannotLoad
from lasagna.core.session_config._session_keys import SessionSettingRefused
from lasagna.mcp import server
from lasagna.sdk.exceptions import ConfigurationError


@pytest.fixture
def records():
    seen = []
    handler = logger.add(lambda m: seen.append(m.record), level="DEBUG", diagnose=False)
    logger.enable("lasagna")
    yield seen
    logger.remove(handler)


@pytest.mark.parametrize(
    "exc",
    [
        SessionSettingRefused("chunk_size 999999 is above the largest value, 8192."),
        DocumentsLeftBehind("3 of 3 documents are in another collection."),
        EmbeddingSizeUnverified("The store could not be read."),
        ProfileCannotLoad("The profile's model is not downloaded."),
    ],
    ids=lambda e: type(e).__name__,
)
def test_a_refusal_is_logged_at_debug_without_a_traceback(records, exc):
    server._log_tool_failure("patch_rag_config failed", exc)
    assert [(r["level"].name, r["exception"]) for r in records] == [("DEBUG", None)]


def test_a_failure_is_still_an_error_with_its_traceback(records):
    for exc in (RuntimeError("boom"), ConfigurationError("Could not list profiles: boom")):
        records.clear()
        try:
            raise exc
        except Exception as caught:
            server._log_tool_failure("list_profiles failed", caught)
        assert [r["level"].name for r in records] == ["ERROR"]
        assert records[0]["exception"] is not None
