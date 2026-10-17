"""The monoT5 reranker loads its tokenizer on a plain install.

monoT5's checkpoint (castorini/monot5-base-msmarco-10k) ships its tokenizer as
a SentencePiece model, ``spiece.model``, and no ``tokenizer.json``. Reading it
needs the ``sentencepiece`` package, which no dependency declared, so on every
install ``reranker: monot5`` failed with "Cannot load monoT5 model ... Error
parsing line b'\\x0e' in .../spiece.model" (transformers fell back to reading
the file as a tiktoken vocabulary). The real-model test skips unless the 990 MB
checkpoint is cached, so nothing caught it. This builds a tiny SentencePiece
tokenizer the same way and loads it the way the reranker does.
"""

from __future__ import annotations

import json
import re

from tests._paths import PACKAGE_DIR

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REPO = PACKAGE_DIR.parents[1]


def _names(requirements):
    return {re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0].lower() for r in requirements}


def test_sentencepiece_is_a_core_dependency_and_pinned():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert "sentencepiece" in _names(project["dependencies"])
    pins = (REPO / "constraints" / "0.1.0.txt").read_text().splitlines()
    assert any(line.startswith("sentencepiece==") for line in pins)


def test_a_sentencepiece_only_t5_tokenizer_loads(tmp_path):
    import sentencepiece as spm
    from transformers import AutoTokenizer

    text = tmp_path / "corpus.txt"
    text.write_text(
        "\n".join(
            f"Query: question {i} Document: passage {i} about Paris and France Relevant: true"
            for i in range(200)
        )
    )
    spm.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(tmp_path / "spiece"),
        vocab_size=60,
        hard_vocab_limit=False,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
    )
    model = tmp_path / "tiny-t5"
    model.mkdir()
    (tmp_path / "spiece.model").rename(model / "spiece.model")
    (model / "config.json").write_text(json.dumps({"model_type": "t5"}))
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "T5Tokenizer", "extra_ids": 0})
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model))

    ids = tokenizer("Query: is it Document: Paris Relevant:")["input_ids"]
    assert len(ids) > 3 and ids[-1] == tokenizer.eos_token_id
