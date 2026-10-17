"""Every model a session can name loads through the server's download policy.

A server refuses a model it would have to download inside a request
(``lasagna.core.model_downloads.check``): the embedder, the cross-encoder and
the chunk tokenizer did, while ``monot5_model``, ``splade_model``,
``colbert_model`` and ``flashrank_model``, all session settings, went straight
to ``from_pretrained`` or FlashRank's ``Ranker``. One session naming any
repository started a download inside a request, or, offline, failed with a
loader's traceback instead of the fix.
"""

from __future__ import annotations

import ast
import sys
from types import ModuleType, SimpleNamespace

import pytest

from lasagna.core import model_downloads
from lasagna.core.model_downloads import ModelNotDownloaded, refuse_downloads_in_requests
from tests._paths import PACKAGE_DIR

ABSENT = "someone/a-model-nobody-downloaded"


@pytest.fixture
def no_downloads(monkeypatch):
    """A server refusing downloads, with every loader failing the test if it is reached."""
    import transformers

    def reached(*a, **k):
        raise AssertionError("the loader ran; it would have downloaded the model")

    for name in ("AutoTokenizer", "AutoModel", "AutoModelForSeq2SeqLM", "AutoModelForMaskedLM"):
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", reached)
    monkeypatch.setattr(model_downloads, "_allowed", lambda: False)
    with refuse_downloads_in_requests():
        yield


def test_monot5(no_downloads):
    from lasagna.retrieval.search.rerankers import _MonoT5

    with pytest.raises(ModelNotDownloaded, match=ABSENT):
        _MonoT5(ABSENT)


def test_splade_and_it_loads_once_the_model_is_there(no_downloads, monkeypatch):
    from lasagna.retrieval.search.hybrid import SPLADESparseEncoder

    encoder = SPLADESparseEncoder(ABSENT)
    with pytest.raises(ModelNotDownloaded, match=ABSENT):
        encoder._ensure_loaded()
    # Not remembered as a failure: it loads once it is on disk.
    assert encoder._failed is None


def test_colbert(no_downloads):
    from lasagna.retrieval.search.ranker import ColBERTReranker

    with pytest.raises(ModelNotDownloaded, match=ABSENT):
        ColBERTReranker(config=SimpleNamespace(colbert_model=ABSENT))._ensure_model()


def test_flashrank(no_downloads, monkeypatch, tmp_path):
    fake = ModuleType("flashrank")

    def ranker(**kw):
        raise AssertionError("FlashRank's Ranker ran; it would have downloaded the model")

    fake.Ranker = ranker
    monkeypatch.setitem(sys.modules, "flashrank", fake)
    monkeypatch.setenv("HOME", str(tmp_path))
    from lasagna.retrieval.search.rerankers import FlashRankReranker

    monkeypatch.setattr(FlashRankReranker, "_rankers", {})
    with pytest.raises(ModelNotDownloaded, match="ms-marco-MultiBERT-L-12"):
        FlashRankReranker._ranker("ms-marco-MultiBERT-L-12")


def test_a_flashrank_model_in_its_cache_loads(monkeypatch, tmp_path):
    fake = ModuleType("flashrank")
    fake.Ranker = lambda **kw: ("loaded", kw["model_name"])
    monkeypatch.setitem(sys.modules, "flashrank", fake)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".cache" / "flashrank" / "ms-marco-MultiBERT-L-12").mkdir(parents=True)
    from lasagna.retrieval.search.rerankers import FlashRankReranker

    monkeypatch.setattr(FlashRankReranker, "_rankers", {})
    monkeypatch.setattr(model_downloads, "_allowed", lambda: False)
    with refuse_downloads_in_requests():
        assert FlashRankReranker._ranker("ms-marco-MultiBERT-L-12")[0] == "loaded"


#: Loader calls whose function does not call ``model_downloads.check`` itself,
#: and where the check is made instead.
CHECKED_ELSEWHERE = {
    ("infrastructure/embedding/manager.py", "_load_flag_embedding"): (
        "_ensure_loaded checks before _load_model calls it"
    ),
    ("infrastructure/embedding/manager.py", "_load_sentence_transformer"): (
        "_ensure_loaded checks before _load_model calls it"
    ),
    ("retrieval/search/ranker.py", "_refuse_unscorable_reranker"): "_model_for checks first",
}

LOADERS = {"from_pretrained", "SentenceTransformer", "CrossEncoder", "Ranker", "BGEM3FlagModel"}


def _loader_calls():
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) in LOADERS
            ]
            if not calls:
                continue
            checked = any(
                isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "check"
                for n in ast.walk(fn)
            )
            yield str(path.relative_to(PACKAGE_DIR)), fn.name, checked


def test_every_loader_call_is_checked_or_says_where_it_is():
    unchecked = [
        (where, fn)
        for where, fn, checked in _loader_calls()
        if not checked and (where, fn) not in CHECKED_ELSEWHERE
    ]
    assert not unchecked, f"model loaded without model_downloads.check: {unchecked}"


def test_the_list_of_exceptions_is_current():
    found = {(where, fn) for where, fn, checked in _loader_calls() if not checked}
    assert set(CHECKED_ELSEWHERE) <= found, set(CHECKED_ELSEWHERE) - found
