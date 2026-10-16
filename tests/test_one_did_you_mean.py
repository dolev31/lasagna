"""Every "did you mean" suggestion comes from one helper.

The data-file reader and the synthesizer each carried their own copy of the
matching, so a change to how names are suggested (the cutoff, the quoting)
reached the CLI and the sweep parser but not the errors for a data file.
"""

from __future__ import annotations

import inspect

import pytest

from lasagna.core._text import did_you_mean


@pytest.mark.parametrize(
    "module",
    [
        "lasagna.knowledge_evaluation_sdk.benchmarks.adapters.local_files",
        "lasagna.knowledge_evaluation_sdk.synthesize",
    ],
)
def test_no_module_matches_names_on_its_own(module):
    import importlib

    source = inspect.getsource(importlib.import_module(module))
    assert "get_close_matches" not in source
    assert "did_you_mean" in source


def test_a_message_can_quote_the_suggestion_its_own_way():
    assert did_you_mean("anwser", ["answer", "question"]) == " (did you mean 'answer'?)"
    assert did_you_mean("anwser", ["answer"], quote='"') == ' (did you mean "answer"?)'
    assert did_you_mean("zzz", ["answer"], quote='"') == ""
