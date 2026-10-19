"""The first PDF is not charged for Docling's one-time model load.

A new Docling converter loads (and on the first run downloads) its models
inside the first ``convert()``. That time counted against ``docling_timeout``,
so on a slow or busy machine the first PDF failed with "did not convert within
300s" while later ones took seconds. The models now load when the converter is
built, before the timer starts.
"""

import sys
import time
import types

import pytest

from lasagna.processing import docling_processor as dp


class _LazyConverter:
    """Loads its models on the first convert(), like Docling's."""

    def __init__(self, load_seconds: float):
        self.load_seconds = load_seconds
        self.loaded = False

    def initialize_pipeline(self, fmt):
        time.sleep(self.load_seconds)
        self.loaded = True

    def convert(self, path):
        if not self.loaded:
            self.initialize_pipeline(None)
        return "converted"


@pytest.fixture
def fake_docling(monkeypatch):
    base = types.ModuleType("docling.datamodel.base_models")
    base.InputFormat = types.SimpleNamespace(PDF="pdf")
    for name, mod in (
        ("docling", types.ModuleType("docling")),
        ("docling.datamodel", types.ModuleType("docling.datamodel")),
        ("docling.datamodel.base_models", base),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _processor(converter):
    proc = dp.DoclingProcessor(types.SimpleNamespace())
    proc._build_converter = lambda: converter
    return proc


def test_the_model_load_happens_before_the_timer_starts(fake_docling):
    converter = _LazyConverter(load_seconds=0.6)
    proc = _processor(converter)
    result = dp._convert_with_timeout(proc._get_converter(), "test_document.pdf", 0.3)
    assert result == "converted"


def test_a_failed_model_load_is_logged_and_convert_reports_it(fake_docling):
    class Broken(_LazyConverter):
        def initialize_pipeline(self, fmt):
            raise OSError("no network")

        def convert(self, path):
            raise OSError("no network")

    proc = _processor(Broken(0))
    converter = proc._get_converter()
    with pytest.raises(OSError, match="no network"):
        dp._convert_with_timeout(converter, "a.pdf", 5)


def test_a_converter_without_initialize_pipeline_still_works():
    class Plain:
        def convert(self, path):
            return "ok"

    proc = _processor(Plain())
    assert dp._convert_with_timeout(proc._get_converter(), "a.pdf", 5) == "ok"


def test_a_timeout_says_what_to_run_for_the_first_download():
    class Stuck:
        def convert(self, path):
            time.sleep(2)

    with pytest.raises(TimeoutError) as err:
        dp._convert_with_timeout(Stuck(), "test_document.pdf", 0.1)
    message = str(err.value)
    assert "docling-tools models download" in message
    assert "DOCLING_ARTIFACTS_PATH" in message
