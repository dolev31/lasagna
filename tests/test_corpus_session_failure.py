"""A failed corpus ingest must fail its siblings, not hang them.

Tasks sharing a corpus elect one ingester; the rest park on an `asyncio.Event`
they captured by reference. Two failure paths — no corpus available, and more
than half the qrels documents missing — deleted the session slot and returned
*without setting the event*. Every waiting task then awaited an Event nobody
would ever set.

The shape of the bug matters: a sweep over one benchmark and six configs did
not report six failures, it produced no output at all and never returned. The
success and exception paths in the same function both call `set()`, so the two
early returns were the odd ones out.

A second defect sat behind it: waiters did `_corpus_sessions[cache_key]` after
waking, and a failed ingest removes that key — so fixing only the hang would
have turned it into a `KeyError` in every sibling.
"""

from __future__ import annotations

import asyncio

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp


@pytest.fixture(autouse=True)
def _clean_sessions():
    exp._corpus_sessions.clear()
    yield
    exp._corpus_sessions.clear()


def _slot(**kw):
    slot = {
        "session_id": None,
        "corpus_to_valid": {},
        "valid_to_corpus": {},
        "ref_count": 2,
        "ready": asyncio.Event(),
        "_ingester_claimed": True,
    }
    slot.update(kw)
    return slot


class TestWaitersAreAlwaysWoken:
    """Every path that removes a slot must set its event first."""

    def test_a_waiter_does_not_hang_when_the_ingest_fails(self):
        """The whole defect, expressed as a timeout."""

        async def _run():
            slot = _slot()
            exp._corpus_sessions["k"] = slot

            async def _waiter():
                await slot["ready"].wait()
                return slot.get("error")

            async def _ingester_fails():
                await asyncio.sleep(0)
                async with exp._corpus_cache_lock:
                    removed = exp._corpus_sessions.pop("k", None)
                if removed is not None:
                    removed["error"] = "no corpus"
                    removed["ready"].set()

            waiter = asyncio.create_task(_waiter())
            await _ingester_fails()
            return await asyncio.wait_for(waiter, timeout=2.0)

        assert asyncio.run(_run()) == "no corpus"

    def test_the_slot_object_survives_removal_from_the_registry(self):
        """Waiters hold the slot by reference, so popping it is safe —
        provided they read the object and not the dict."""
        slot = _slot()
        exp._corpus_sessions["k"] = slot
        removed = exp._corpus_sessions.pop("k", None)
        removed["ready"].set()
        assert removed is slot
        assert slot["ready"].is_set()
        assert "k" not in exp._corpus_sessions

    @pytest.mark.parametrize(
        "marker",
        [
            "if corpus is None:",
            "if pct > 50:",
        ],
    )
    def test_both_early_returns_wake_waiters_before_returning(self, marker):
        """Read from the source, because these are the two paths that were wrong.

        A future edit that reintroduces `del _corpus_sessions[key]; return`
        would restore the hang, and no unit test can observe a hang that only
        manifests with real ingestion.
        """
        import inspect

        source = inspect.getsource(exp._ensure_corpus_ingested)
        assert marker in source, f"{marker} no longer present; update this test"

        after = source.split(marker, 1)[1][:900]
        assert '["ready"].set()' in after, (
            f"the branch at {marker!r} removes the session slot without setting "
            f"its ready event, so every waiting task hangs forever"
        )

    def test_no_removal_path_leaves_the_event_unset(self):
        """Every `pop`/`del` of a session slot must be near a `set()`."""
        import inspect
        import re

        source = inspect.getsource(exp._ensure_corpus_ingested)
        for match in re.finditer(r"_corpus_sessions\.pop\(|del _corpus_sessions\[", source):
            window = source[match.start() : match.start() + 500]
            assert '["ready"].set()' in window, (
                f"a session slot is removed at offset {match.start()} without waking its waiters"
            )


class TestTheAbortSaysHowToFixIt:
    def test_the_qrels_abort_names_the_setting(self):
        """ "Results would be unreliable" without a remedy sends the user
        looking through source. The cap is the remedy."""
        import inspect

        source = inspect.getsource(exp._ensure_corpus_ingested)
        abort = source.split("if pct > 50:", 1)[1][:700]
        assert "max_corpus_docs" in abort


class TestNoLimitMeansNoLimit:
    """`max_corpus_docs: 0` is documented as "no limit" in two places.

    The harness omitted the kwarg entirely when it was 0, so the adapter's own
    signature default of 100_000 applied — capping 9 of the 13 BEIR corpora and
    aborting 7 of them, on the setting documented as uncapped.
    """

    def test_zero_is_passed_through_rather_than_omitted(self):
        import inspect

        source = inspect.getsource(exp._ensure_corpus_ingested)
        assert 'load_kwargs = {"max_docs"' in source, (
            "max_docs must always be passed; omitting it lets the adapter "
            "default silently reimpose a cap"
        )
        assert "if max_corpus_docs > 0:\n            load_kwargs" not in source

    def test_the_adapters_treat_zero_as_unlimited(self):
        """The pass-through only works if 0 means unlimited downstream."""
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        adapter = benchmark_registry.get("smoke")
        full = adapter.load_corpus(max_docs=0)
        assert full is not None and len(full) > 0
        capped = adapter.load_corpus(max_docs=1)
        assert len(capped) == 1
        assert len(full) > len(capped)

    def test_every_corpus_loop_guards_the_zero_case(self):
        """Checked per adapter, because they are not uniform.

        Passing `max_docs=0` through exposed that `private_eval` broke on
        `len(corpus) >= max_docs` with no guard — true on the first document, so
        "no limit" produced a one-document corpus. `smoke` accepted the argument
        and ignored it entirely. Assuming the adapters agreed is what made the
        pass-through look safe.
        """
        import inspect
        import re

        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        checked = 0
        for name in sorted(benchmark_registry.names()):
            adapter = benchmark_registry.get(name)
            loader = type(adapter).load_corpus
            try:
                source = inspect.getsource(loader)
            except (OSError, TypeError):  # pragma: no cover - builtins
                continue
            if "max_docs" not in source:
                continue
            checked += 1
            for cmp_line in re.findall(r".*>=\s*max_docs.*", source):
                assert "max_docs and" in cmp_line, (
                    f"{name}: `{cmp_line.strip()}` is true for max_docs=0, so "
                    f"'no limit' truncates to a single document"
                )
        assert checked >= 2, "expected at least the BEIR and private_eval loaders"


class TestTruncationReachesTheBundle:
    """`TRUNCATED_CORPUS` is one of only two blocking integrity codes.

    The adapter set `corpus_truncated` on *itself*; the runner copied it from
    the *task*; nothing bridged the two. So the flag was always False on every
    result, the blocking check could never fire, and a truncated BEIR bundle
    passed `lasagna results validate` clean. The only test that exercised the
    code fed `_integrity()` a hand-built dict, so the missing wire was invisible.
    """

    def test_the_flag_is_published_into_the_session_slot(self):
        import inspect

        from lasagna.knowledge_evaluation_sdk import experiment as e

        source = inspect.getsource(e._ensure_corpus_ingested)
        assert 'slot["corpus_truncated"]' in source, (
            "the ingester must publish the adapter's truncation flag, or nothing "
            "downstream can see it"
        )

    def test_it_reaches_the_task(self):
        import inspect

        from lasagna.knowledge_evaluation_sdk import experiment as e

        assert "task.corpus_truncated" in inspect.getsource(e._eval_live)

    def test_the_flag_is_set_from_the_break_not_from_metadata(self):
        """Declared counts are hand-maintained and were wrong by 47x for
        nfcorpus, which set the flag on runs where nothing was truncated."""
        import inspect

        from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.beir import (
            BEIRBenchmarkAdapter,
        )

        source = inspect.getsource(BEIRBenchmarkAdapter.load_corpus)
        assert "self.corpus_truncated = truncated" in source
        # The metadata comparison must no longer set the flag.
        before_loop = source.split("corpus: Dict[str, str] = {}")[0]
        assert "self.corpus_truncated = True" not in before_loop

    def test_a_full_load_does_not_flag_truncation(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        adapter = benchmark_registry.get("smoke")
        adapter.load_corpus(max_docs=0)
        assert adapter.corpus_truncated is False
