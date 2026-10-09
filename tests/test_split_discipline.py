"""A test-set number must not be a train-set number wearing its label.

`load_benchmark` has had split discipline since it was written: it refuses to
substitute a split unless asked, caches under the split actually loaded, and
records the substitution. The BEIR path bypassed all of it.

`_load_with_fallback` walked `[requested, "test", "validation", "train"]` and
returned the first that loaded, swallowing the error. So a benchmark that does
not publish the requested split silently returned **train** judgements — cached
under the requested split's key, stamped `split: "test"` on the result, with no
warning and nothing in provenance. Every later run read the poisoned cache.

Training-set retrieval scores reported as test-set scores is the worst thing
this package can do quietly, and nothing in the suite touched the code path.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk import datasets as ds


class _FakeDataset:
    """Stands in for a `datasets.Dataset` without touching the network."""

    def __init__(self, split: str) -> None:
        self.split = split


def _loader(available):
    """A `load_dataset` stub that only knows about `available` splits."""

    def _load_dataset(hf_name, split=None, **kwargs):
        if split not in available:
            raise ValueError(f"Unknown split {split!r} for {hf_name!r}")
        return _FakeDataset(split)

    return _load_dataset


@pytest.fixture
def fake_hf(monkeypatch):
    """Patch the lazily-imported `datasets.load_dataset`."""

    def _install(available):
        import datasets as hf

        monkeypatch.setattr(hf, "load_dataset", _loader(available), raising=False)

    return _install


class TestItReportsTheSplitItActuallyLoaded:
    """The return value changed from a dataset to `(dataset, split)`.

    A caller that cannot see which split it got cannot key a cache on it, which
    is precisely how the substitution became invisible.
    """

    def test_the_requested_split_is_returned_with_the_data(self, fake_hf):
        fake_hf({"test"})
        dataset, split = ds._load_with_fallback("x/y", splits=["test", "train"])
        assert split == "test"
        assert dataset.split == "test"

    def test_a_substitution_reports_the_substitute_not_the_request(self, fake_hf):
        fake_hf({"train"})
        _, split = ds._load_with_fallback("x/y", splits=["test", "train"])
        assert split == "train", "the caller must be told what it actually got"


class TestSubstitutionIsRefusableAndLoud:
    def test_it_refuses_when_fallback_is_disabled(self, fake_hf):
        """The discipline the non-BEIR path always had, now reachable here."""
        fake_hf({"train"})
        with pytest.raises(ValueError, match="No usable split"):
            ds._load_with_fallback("x/y", splits=["test", "train"], allow_split_fallback=False)

    def test_refusing_does_not_break_the_happy_path(self, fake_hf):
        fake_hf({"test"})
        _, split = ds._load_with_fallback(
            "x/y", splits=["test", "train"], allow_split_fallback=False
        )
        assert split == "test"

    def test_a_substitution_warns(self, fake_hf):
        """Silent substitution is the whole defect. A log line is the minimum."""
        from loguru import logger

        fake_hf({"train"})
        captured: list = []
        sink = logger.add(captured.append, level="WARNING")
        try:
            ds._load_with_fallback("x/y", splits=["test", "train"])
        finally:
            logger.remove(sink)

        joined = " ".join(captured)
        assert "test" in joined and "train" in joined
        assert "NOT comparable" in joined

    def test_the_error_says_what_it_tried(self, fake_hf):
        fake_hf(set())
        with pytest.raises(ValueError) as exc:
            ds._load_with_fallback("x/y", splits=["test", "validation"])
        assert "test" in str(exc.value) and "validation" in str(exc.value)


class TestTheBeirPathHonoursIt:
    """The qrels *are* the split; substituting them is the damaging case.

    Queries are not split-sensitive for BEIR — the `queries` config holds every
    query and the qrels decide which are in play — so that fallback stays on.
    """

    def test_qrels_are_loaded_with_the_callers_flag(self, monkeypatch, tmp_path):
        seen = {}

        def _spy(hf_name, config=None, splits=None, revision="", allow_split_fallback=True):
            seen[config or "qrels"] = allow_split_fallback
            return _FakeDataset(splits[0]), splits[0]

        monkeypatch.setattr(ds, "_load_with_fallback", _spy)
        monkeypatch.setattr(ds, "_convert_beir_with_qrels", lambda q, r: [])
        # A path that does not exist, so the cache misses and the loads run.
        monkeypatch.setattr(ds, "_cache_path", lambda *a, **k: tmp_path / "miss.json")

        try:
            ds.load_benchmark("beir/scifact", split="test", allow_split_fallback=False)
        except Exception:
            # Anything downstream of the two loads is irrelevant here.
            pass

        assert seen.get("qrels") is False, "qrels must obey allow_split_fallback"
        assert seen.get("queries") is True, "queries are not split-sensitive for BEIR"


class TestTheCacheIsKeyedOnWhatWasLoaded:
    def test_a_substituted_split_rekeys_the_cache(self):
        """Otherwise the substitution persists invisibly across runs.

        `actual_split` was bound once and never updated on the BEIR branch, so
        the re-key could not fire and a substituted result was written under the
        requested split's filename.
        """
        import inspect

        source = inspect.getsource(ds.load_benchmark)
        assert "actual_split, cache_key_split" not in source  # guard against drift
        assert "if actual_split != split:" in source
        # The BEIR branch must assign it, or the re-key is unreachable there.
        beir_branch = source.split('if fmt == "beir"')[1].split("else:")[0]
        assert "actual_split" in beir_branch, (
            "the BEIR branch must record the split it loaded, or the cache "
            "re-key below cannot fire for BEIR"
        )
