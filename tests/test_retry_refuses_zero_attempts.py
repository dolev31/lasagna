"""A retry count that cannot make one call is refused when it is written.

``with_retry(max_attempts=0)`` used to build a wrapper that never called the
function and then executed ``raise None``, so the first call failed with
"exceptions must derive from BaseException", far from the mistake.
"""

from __future__ import annotations

import pytest

from lasagna.sdk.utils.retry import RetryConfig, with_retry, with_retry_sync


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_with_retry_refuses_a_count_below_one(bad):
    with pytest.raises(ValueError, match="max_attempts"):
        with_retry(max_attempts=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_with_retry_sync_refuses_a_count_below_one(bad):
    with pytest.raises(ValueError, match="max_attempts"):
        with_retry_sync(max_attempts=bad)


def test_retry_config_refuses_a_count_below_one():
    with pytest.raises(ValueError, match="max_attempts"):
        RetryConfig(max_attempts=0)


async def test_one_attempt_calls_the_function_once():
    calls = []

    @with_retry(max_attempts=1, delay=0)
    async def once():
        calls.append(1)
        return "ok"

    assert await once() == "ok"
    assert calls == [1]
