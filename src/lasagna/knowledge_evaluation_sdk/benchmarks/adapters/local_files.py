"""Benchmarks from your own files: a JSONL or CSV file, or a BEIR folder.

``benchmarks: [{path: data/my.jsonl}]`` in a sweep, or
``lasagna benchmark run ./my.jsonl``, measures a pipeline on questions that are
not in the catalogue, with no adapter to write. The formats are documented in
``docs/own-data.md``; ``lasagna benchmark validate-data`` checks a file.

Two properties matter more than convenience.

**The name carries the content.** A file registers as
``local/<stem>@<12 hex>``: the first twelve hex digits of the sha256 of its
bytes (for a BEIR folder, of the files read from it). The benchmark name is part
of the checkpoint key, the corpus key and the comparability key, so an edited
file never resumes an old checkpoint, never reuses the old corpus and never
shares a leaderboard group with the file it replaced. ``hf_revision`` holds the
full digest, so the dataset counts as pinned.

**A bad row is refused before anything loads**, with the file, row and field
named. Skipping it would change the number of questions without saying so.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from lasagna.core._text import did_you_mean

from .._base import (
    BaseBenchmarkAdapter,
    BenchmarkCategory,
    BenchmarkEvalMode,
    BenchmarkInfo,
    BenchmarkSample,
)

#: Separator between passages in a CSV ``contexts`` cell.
CSV_CONTEXT_SEPARATOR = "|||"

#: File suffixes read as data. Anything else is refused with the formats named.
DATA_SUFFIXES = (".jsonl", ".csv")

_JSONL_FIELDS = (
    "id",
    "question",
    "answer",
    "answers",
    "context",
    "contexts",
    "expected_docs",
    "metadata",
)
_CSV_COLUMNS = ("id", "question", "answer", "contexts")

#: Errors listed in one message; the rest are counted.
_MAX_ERRORS_SHOWN = 20

#: The content-hash suffix, stripped before a name is hashed again.
_HASH_SUFFIX = re.compile(r"@[0-9a-f]{12}$")

_TRACKS = ("retrieval", "generation")


class DataFileError(ValueError):
    """The data cannot be used as a benchmark. ``errors`` lists every problem."""

    def __init__(self, path: Path, errors: Sequence[str]):
        self.path = Path(path)
        self.errors = list(errors)
        shown = self.errors[:_MAX_ERRORS_SHOWN]
        more = len(self.errors) - len(shown)
        super().__init__("\n".join(shown + ([f"... and {more} more"] if more > 0 else [])))


# ---------------------------------------------------------------------------
# Parsed data
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    """One question, validated."""

    query_id: str
    question: str
    answers: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    unanswerable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: ``{doc_id: grade}``. The row's own contexts for JSONL and CSV (none for an
    #: unanswerable row: its passages are what it is about, not what answers
    #: it); the qrels for BEIR.
    expected_docs: Dict[str, int] = field(default_factory=dict)

    def full_metadata(self) -> Dict[str, Any]:
        return {**self.metadata, "unanswerable": self.unanswerable}

    def sample(self) -> BenchmarkSample:
        """A fresh sample: callers may edit it without touching this row."""
        return BenchmarkSample(
            query_id=self.query_id,
            question=self.question,
            ground_truth=self.answers[0] if self.answers else "",
            ground_truth_answers=list(self.answers),
            expected_docs=dict(self.expected_docs),
            context="\n\n".join(self.contexts),
            metadata=self.full_metadata(),
        )


@dataclass
class LocalData:
    """A validated data file or BEIR folder."""

    path: Path
    #: ``"jsonl"``, ``"csv"`` or ``"beir"``.
    kind: str
    #: sha256 of the file's bytes, or of the files a BEIR folder is read from.
    sha256: str
    #: ``{split: rows}``. One split, ``test``, except in a BEIR folder.
    splits: Dict[str, List[_Row]]
    #: ``{doc_id: text}``: the BEIR corpus, or every row's contexts pooled.
    #: Empty when there is nothing to retrieve.
    corpus: Dict[str, str]

    @property
    def label(self) -> str:
        """How messages name it: the file or folder name."""
        return self.path.name

    def benchmark_name(self, name: str = "") -> str:
        """``local/<stem>@<hash>``, or ``<name>@<hash>`` when the sweep names it."""
        stem = self.path.name if self.kind == "beir" else self.path.stem
        base = _HASH_SUFFIX.sub("", name) if name else f"local/{stem}"
        return f"{base}@{self.sha256[:12]}"

    def rows(self) -> List[_Row]:
        return [row for rows in self.splits.values() for row in rows]

    def track_support(self) -> Dict[str, Optional[str]]:
        """``{track: None}`` when the data supports it, else why it does not."""
        rows = self.rows()
        support: Dict[str, Optional[str]] = {}
        if not self.corpus:
            support["retrieval"] = "no row has contexts, so there is nothing to retrieve"
        elif not any(r.expected_docs for r in rows):
            support["retrieval"] = (
                "every row with contexts is unanswerable, so no passage is relevant to any question"
            )
        else:
            support["retrieval"] = None
        if self.kind == "beir":
            support["generation"] = "a BEIR folder has relevance judgments but no answers"
        elif not any(r.answers or r.unanswerable for r in rows):
            support["generation"] = 'no row has an "answer"'
        else:
            support["generation"] = None
        return support

    def supported_tracks(self) -> List[str]:
        return [t for t, why in self.track_support().items() if why is None]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_local_data(path: Any) -> LocalData:
    """Read and validate a JSONL or CSV file or a BEIR folder.

    Raises:
        DataFileError: naming every problem found, each with its file and row.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise DataFileError(p, [f"{_shown(p)}: no such file or folder"])
    if p.is_dir():
        return _read_beir(p)
    suffix = p.suffix.lower()
    if suffix not in DATA_SUFFIXES:
        raise DataFileError(
            p,
            [
                f"{p.name}: cannot read {suffix or 'a file with no extension'} as benchmark "
                f"data. Use a .jsonl file (one JSON object per line), a .csv file with a "
                f"header, or a BEIR folder (corpus.jsonl, queries.jsonl, qrels/test.tsv)."
            ],
        )
    raw = p.read_bytes()
    text = _decode(p.name, raw)
    errors: List[str] = []
    if suffix == ".jsonl":
        rows = _parse_jsonl(p.name, text, errors)
    else:
        rows = _parse_csv(p.name, text, errors)
    if not errors and not rows:
        errors.append(f"{p.name}: the file has no rows")
    _check_unique_ids(p.name, rows, errors)
    corpus = _pool_contexts(p.name, rows, errors)
    if errors:
        raise DataFileError(p, errors)
    return LocalData(
        path=p,
        kind=suffix.lstrip("."),
        sha256=hashlib.sha256(raw).hexdigest(),
        splits={"test": [row for _, row in rows]},
        corpus=corpus,
    )


def _shown(path: Path) -> str:
    """``path`` relative to the working directory when it is under it."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _decode(label: str, raw: bytes) -> str:
    try:
        # utf-8-sig: a spreadsheet's CSV export often starts with a byte-order mark.
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataFileError(
            Path(label), [f"{label}: not UTF-8 text (byte {exc.start}); save it as UTF-8"]
        ) from exc


def _json_type(value: Any) -> str:
    """The JSON name of a value's type, for messages about JSON data."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


#: (line or row number, row) pairs, in file order.
_Numbered = List[Tuple[int, _Row]]


def _parse_jsonl(label: str, text: str, errors: List[str]) -> _Numbered:
    rows: _Numbered = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{label} row {lineno}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{where}: not valid JSON ({exc.msg} at column {exc.colno})")
            continue
        if not isinstance(record, dict):
            errors.append(f"{where}: must be a JSON object, got {_json_type(record)}")
            continue
        row = _row_from_record(record, where, f"row{lineno}", errors)
        if row is not None:
            rows.append((lineno, row))
    return rows


def _row_from_record(
    record: Dict[str, Any], where: str, default_id: str, errors: List[str]
) -> Optional[_Row]:
    """One JSONL object as a row, or None with the problems added to ``errors``."""
    before = len(errors)

    def bad(message: str) -> None:
        errors.append(f"{where}: {message}")

    for key in sorted(set(record) - set(_JSONL_FIELDS)):
        hint = did_you_mean(key, _JSONL_FIELDS, quote='"')
        bad(f'unknown field "{key}"{hint}; put fields of your own under "metadata"')

    question = _text(record, "question", bad, required=True)
    query_id = _identifier(record, bad) or default_id

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        bad(f'field "metadata" must be an object, got {_json_type(metadata)}')
        metadata = {}
    unanswerable = metadata.get("unanswerable", False)
    if not isinstance(unanswerable, bool):
        bad(f'field "metadata.unanswerable" must be true or false, got {_json_type(unanswerable)}')
        unanswerable = False

    if "answer" in record and "answers" in record:
        bad('give "answer" or "answers", not both')
    answers: List[str] = []
    if "answer" in record:
        if unanswerable and record["answer"] == "":
            pass  # SQuAD v2 writes an unanswerable question's answer as ""
        elif record["answer"] == "":
            bad(
                'field "answer" is empty; leave it out, or mark a question with no answer '
                'with "metadata": {"unanswerable": true}'
            )
        else:
            answer = _text(record, "answer", bad)
            answers = [answer] if answer else []
    elif "answers" in record:
        answers = _text_list(record, "answers", bad, allow_empty=False)
    if unanswerable and answers:
        bad('"metadata.unanswerable" is true, so the row cannot also have an answer')

    if "context" in record and "contexts" in record:
        bad('give "context" or "contexts", not both')
    contexts: List[str] = []
    if "context" in record:
        context = _text(record, "context", bad)
        contexts = [context] if context else []
    elif "contexts" in record:
        contexts = _text_list(record, "contexts", bad, allow_empty=True)

    doc_ids: List[str] = []
    if "expected_docs" in record:
        doc_ids = _text_list(record, "expected_docs", bad, allow_empty=False)
        if doc_ids and not contexts:
            bad(
                '"expected_docs" needs "contexts": contexts[i] is the text of '
                "expected_docs[i]. For judgments over a separate corpus, use a BEIR folder."
            )
        elif doc_ids and len(doc_ids) != len(contexts):
            bad(
                f'"expected_docs" names {len(doc_ids)} documents but "contexts" has '
                f"{len(contexts)} passages; they pair up in order"
            )
        elif len(set(doc_ids)) != len(doc_ids):
            bad('"expected_docs" names the same document twice')

    if len(errors) > before or question is None:
        return None
    return _Row(
        query_id=query_id,
        question=question,
        answers=answers,
        contexts=contexts,
        unanswerable=unanswerable,
        metadata={k: v for k, v in metadata.items() if k != "unanswerable"},
        # Filled with document ids by `_pool_contexts`; the names are kept here
        # until then.
        expected_docs={d: 1 for d in doc_ids},
    )


def _text(
    record: Dict[str, Any], key: str, bad: Callable[[str], None], required: bool = False
) -> Optional[str]:
    if key not in record:
        if required:
            bad(f'field "{key}" is missing')
        return None
    value = record[key]
    if not isinstance(value, str):
        bad(f'field "{key}" must be a string, got {_json_type(value)}')
        return None
    if not value.strip():
        bad(f'field "{key}" is empty')
        return None
    return value.strip()


def _text_list(
    record: Dict[str, Any], key: str, bad: Callable[[str], None], allow_empty: bool
) -> List[str]:
    value = record[key]
    if not isinstance(value, list):
        bad(f'field "{key}" must be a list of strings, got {_json_type(value)}')
        return []
    if not value and not allow_empty:
        bad(f'field "{key}" is an empty list')
        return []
    out: List[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            bad(f'field "{key}" item {index} must be a string, got {_json_type(item)}')
        elif not item.strip():
            bad(f'field "{key}" item {index} is empty')
        else:
            out.append(item.strip())
    return out if len(out) == len(value) else []


def _identifier(record: Dict[str, Any], bad: Callable[[str], None]) -> Optional[str]:
    if "id" not in record:
        return None
    value = record["id"]
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        bad(f'field "id" must be a string or a number, got {_json_type(value)}')
        return None
    text = str(value).strip()
    if not text:
        bad('field "id" is empty')
        return None
    return text


def _parse_csv(label: str, text: str, errors: List[str]) -> _Numbered:
    try:
        records = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        errors.append(f"{label}: not readable as CSV ({exc})")
        return []
    if not records:
        return []
    header = [cell.strip() for cell in records[0]]
    if "question" not in header:
        errors.append(
            f"{label} row 1: the first row must be a header naming the columns "
            f"({', '.join(_CSV_COLUMNS)}); it has {', '.join(map(repr, header)) or 'nothing'}"
        )
        return []
    for column in header:
        if column not in _CSV_COLUMNS:
            hint = did_you_mean(column, _CSV_COLUMNS, quote='"')
            errors.append(
                f'{label} row 1: unknown column "{column}"{hint}; '
                f"the columns are {', '.join(_CSV_COLUMNS)}"
            )
    if len(set(header)) != len(header):
        errors.append(f"{label} row 1: a column is named twice")
    if errors:
        return []

    rows: _Numbered = []
    # Numbered as a spreadsheet numbers them: the header is row 1.
    for number, cells in enumerate(records[1:], start=2):
        if not any(cell.strip() for cell in cells):
            continue
        where = f"{label} row {number}"
        if len(cells) != len(header):
            errors.append(f"{where}: has {len(cells)} cells, the header has {len(header)}")
            continue
        record = dict(zip(header, cells))
        question = record["question"].strip()
        if not question:
            errors.append(f'{where}: column "question" is empty')
            continue
        contexts: List[str] = []
        if record.get("contexts", "").strip():
            contexts = [c.strip() for c in record["contexts"].split(CSV_CONTEXT_SEPARATOR)]
            if not all(contexts):
                errors.append(
                    f'{where}: column "contexts" has an empty passage between '
                    f'"{CSV_CONTEXT_SEPARATOR}" separators'
                )
                continue
        answer = record.get("answer", "").strip()
        rows.append(
            (
                number,
                _Row(
                    query_id=record.get("id", "").strip() or f"row{number}",
                    question=question,
                    answers=[answer] if answer else [],
                    contexts=contexts,
                ),
            )
        )
    return rows


def _check_unique_ids(label: str, rows: _Numbered, errors: List[str]) -> None:
    first: Dict[str, int] = {}
    for number, row in rows:
        if row.query_id in first:
            errors.append(
                f'{label} row {number}: id "{row.query_id}" is also used by row '
                f"{first[row.query_id]}"
            )
        else:
            first[row.query_id] = number


def _passage_id(text: str) -> str:
    """A passage named by its text, so one shared by several rows is one document."""
    return "ctx-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def _pool_contexts(label: str, rows: _Numbered, errors: List[str]) -> Dict[str, str]:
    """Every row's contexts as one corpus, and each row's own as its relevant set.

    Pooled across the whole file rather than the sampled rows, so ``n_samples``
    changes how many questions are asked, never how many passages there are to
    search. A passage that several rows share is one document, relevant to
    each of them; otherwise a retriever returning one copy would be marked
    wrong for the other row.
    """
    corpus: Dict[str, str] = {}
    defined_by: Dict[str, int] = {}
    for number, row in rows:
        named = list(row.expected_docs)
        doc_ids = named or [_passage_id(text) for text in row.contexts]
        for doc_id, text in zip(doc_ids, row.contexts):
            if doc_id in corpus and corpus[doc_id] != text:
                errors.append(
                    f'{label} row {number}: document "{doc_id}" has different text than '
                    f"in row {defined_by[doc_id]}"
                )
                continue
            corpus.setdefault(doc_id, text)
            defined_by.setdefault(doc_id, number)
        row.expected_docs = {} if row.unanswerable else {d: 1 for d in doc_ids}
    return corpus


def _read_jsonl_records(
    folder: Path, rel: str, errors: List[str]
) -> List[Tuple[int, Dict[str, Any]]]:
    label = f"{folder.name}/{rel}"
    text = _decode(label, (folder / rel).read_bytes())
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label} row {lineno}: not valid JSON ({exc.msg} at column {exc.colno})")
            continue
        if not isinstance(record, dict):
            errors.append(f"{label} row {lineno}: must be a JSON object, got {_json_type(record)}")
            continue
        records.append((lineno, record))
    return records


def _beir_id(record: Dict[str, Any], where: str, errors: List[str]) -> Optional[str]:
    value = record.get("_id")
    if value is None:
        errors.append(f'{where}: field "_id" is missing')
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        errors.append(f'{where}: field "_id" must be a non-empty string or number')
        return None
    return str(value).strip()


def _examples(ids: Iterable[str], k: int = 3) -> str:
    return ", ".join(f'"{i}"' for i in sorted(ids)[:k])


def _read_beir(folder: Path) -> LocalData:
    """A BEIR folder: ``corpus.jsonl``, ``queries.jsonl`` and ``qrels/<split>.tsv``.

    The layout BEIR's own downloads use, joined by the same converter as the
    catalogue's BEIR benchmarks, so a folder and its catalogue twin load the
    same queries with the same judgments.
    """
    label = folder.name
    qrels_files = sorted((folder / "qrels").glob("*.tsv")) if (folder / "qrels").is_dir() else []
    missing = [
        name for name in ("corpus.jsonl", "queries.jsonl") if not (folder / name).is_file()
    ] + ([] if qrels_files else ["qrels/test.tsv"])
    if missing:
        raise DataFileError(
            folder,
            [
                f"{label}: a BEIR folder needs corpus.jsonl, queries.jsonl and "
                f"qrels/<split>.tsv; missing {', '.join(missing)}"
            ],
        )

    errors: List[str] = []
    corpus: Dict[str, str] = {}
    for lineno, record in _read_jsonl_records(folder, "corpus.jsonl", errors):
        where = f"{label}/corpus.jsonl row {lineno}"
        doc_id = _beir_id(record, where, errors)
        title, body = record.get("title") or "", record.get("text") or ""
        if not isinstance(title, str) or not isinstance(body, str):
            errors.append(f'{where}: fields "title" and "text" must be strings')
            continue
        if doc_id is None:
            continue
        # Joined as the catalogue's BEIR adapter joins them.
        text = f"{title}\n\n{body}".strip() if title else body.strip()
        if not text:
            errors.append(f'{where}: document "{doc_id}" has no text and no title')
        elif doc_id in corpus:
            errors.append(f'{where}: document id "{doc_id}" is used twice')
        else:
            corpus[doc_id] = text

    queries: Dict[str, str] = {}
    for lineno, record in _read_jsonl_records(folder, "queries.jsonl", errors):
        where = f"{label}/queries.jsonl row {lineno}"
        query_id = _beir_id(record, where, errors)
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f'{where}: field "text" must be a non-empty string')
        elif query_id is not None and query_id in queries:
            errors.append(f'{where}: query id "{query_id}" is used twice')
        elif query_id is not None:
            queries[query_id] = text.strip()

    hasher = hashlib.sha256()
    read = [folder / "corpus.jsonl", folder / "queries.jsonl", *qrels_files]
    for file in read:
        hasher.update(file.relative_to(folder).as_posix().encode() + b"\0")
        hasher.update(hashlib.sha256(file.read_bytes()).digest())

    from ...datasets import _convert_beir_with_qrels

    splits: Dict[str, List[_Row]] = {}
    for qrels_file in qrels_files:
        judgments = _read_qrels(folder, qrels_file, errors)
        rel = f"{label}/qrels/{qrels_file.name}"
        unknown_docs = {j["corpus-id"] for j in judgments} - set(corpus)
        if unknown_docs:
            errors.append(
                f"{rel}: {len(unknown_docs)} judged document id(s) are not in corpus.jsonl, "
                f"for example {_examples(unknown_docs)}"
            )
        unknown_queries = {j["query-id"] for j in judgments} - set(queries)
        if unknown_queries:
            errors.append(
                f"{rel}: {len(unknown_queries)} judged query id(s) are not in queries.jsonl, "
                f"for example {_examples(unknown_queries)}"
            )
        joined = _convert_beir_with_qrels(
            [{"_id": q, "text": t} for q, t in queries.items()], judgments
        )
        if not joined:
            errors.append(
                f"{rel}: no judgment has a score above 0, so no query has a relevant document"
            )
        splits[qrels_file.stem] = [
            _Row(query_id=s["query_id"], question=s["question"], expected_docs=s["expected_docs"])
            for s in joined
        ]

    if errors:
        raise DataFileError(folder, errors)
    return LocalData(
        path=folder, kind="beir", sha256=hasher.hexdigest(), splits=splits, corpus=corpus
    )


def _read_qrels(folder: Path, qrels_file: Path, errors: List[str]) -> List[Dict[str, Any]]:
    label = f"{folder.name}/qrels/{qrels_file.name}"
    text = _decode(label, qrels_file.read_bytes())
    judgments = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split("\t")]
        if lineno == 1 and cells[:2] == ["query-id", "corpus-id"]:
            continue  # BEIR's header
        if len(cells) != 3:
            errors.append(
                f"{label} row {lineno}: needs three tab-separated columns "
                f"(query-id, corpus-id, score), got {len(cells)}"
            )
            continue
        try:
            score = int(cells[2])
        except ValueError:
            errors.append(f'{label} row {lineno}: score must be an integer, got "{cells[2]}"')
            continue
        judgments.append({"query-id": cells[0], "corpus-id": cells[1], "score": score})
    return judgments


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class LocalFileAdapter(BaseBenchmarkAdapter):
    """A JSONL or CSV file, registered under a name carrying its content hash.

    A file with contexts is measured live against them, pooled across every row
    into one corpus: the base class's pooled-gold path, so the result carries
    ``corpus_provenance: pooled_gold`` and the bundle warns that a pool of gold
    passages is an easier search than a real corpus.
    """

    def __init__(self, data: LocalData, name: str):
        self._data = data
        self._name = name
        self.hf_revision = f"sha256:{data.sha256}"
        # Per instance: only a file whose rows carry contexts has a pool.
        self.pools_gold_context = bool(data.corpus) and data.kind != "beir"

    @property
    def data(self) -> LocalData:
        return self._data

    def info(self) -> BenchmarkInfo:
        tracks = self._data.supported_tracks()
        if tracks == ["retrieval"]:
            mode = BenchmarkEvalMode.RETRIEVAL
        elif tracks == ["generation"]:
            mode = BenchmarkEvalMode.GENERATION
        else:
            mode = BenchmarkEvalMode.END_TO_END
        return BenchmarkInfo(
            name=self._name,
            description=f"Your data, read from {self._data.path}",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=mode,
            num_queries=len(self._data.rows()),
            num_documents=len(self._data.corpus) or None,
            license="your own data",
            tags=["local", self._data.kind],
        )

    def available_splits(self) -> List[str]:
        return sorted(self._data.splits)

    def available_filters(self) -> Dict[str, List[str]]:
        """Metadata keys with plain values, such as ``question_type``."""
        values: Dict[str, set] = {}
        for row in self._data.rows():
            for key, value in row.full_metadata().items():
                if isinstance(value, (str, int, bool)):
                    values.setdefault(key, set()).add(str(value))
        return {key: sorted(v) for key, v in sorted(values.items())}

    def load(
        self,
        split: str = "test",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        if split not in self._data.splits:
            raise ValueError(
                f"{self._data.label} has no split {split!r}; it has "
                f"{', '.join(map(repr, self.available_splits()))}. Set `split:` on the "
                f"benchmark entry, or --split."
            )
        rows = list(self._data.splits[split])
        if filters:
            rows = self._filter(rows, filters)
        if n_samples is not None and n_samples < len(rows):
            # The seeded draw every adapter uses, so sampling_hash means the same.
            rows = random.Random(seed).sample(rows, n_samples)
        return [row.sample() for row in rows]

    def _filter(self, rows: List[_Row], filters: Dict[str, Any]) -> List[_Row]:
        available = self.available_filters()
        for key, wanted in filters.items():
            if key not in available:
                raise ValueError(
                    f"{self._data.label}: cannot filter on {key!r}, no row has it in "
                    f"metadata. Filterable: {', '.join(available) or 'nothing'}."
                )
            accepted = {str(w) for w in (wanted if isinstance(wanted, list) else [wanted])}
            rows = [r for r in rows if str(r.full_metadata().get(key)) in accepted]
        return rows

    def build_pooled_gold_corpus(self, samples: List[BenchmarkSample]) -> Optional[Dict[str, str]]:
        """Every row's contexts, not only the sampled rows'. See ``_pool_contexts``."""
        return dict(self._data.corpus) or None

    def pooled_gold_expected_docs(
        self, samples: List[BenchmarkSample]
    ) -> Dict[str, Dict[str, int]]:
        by_id = {row.query_id: row for row in self._data.rows()}
        return {
            s.query_id: dict(by_id[s.query_id].expected_docs)
            for s in samples
            if s.query_id in by_id and by_id[s.query_id].expected_docs
        }


class LocalBeirAdapter(LocalFileAdapter):
    """A BEIR folder: a real corpus, so retrieval is measured against it."""

    def load_corpus(
        self,
        cache_dir: Optional[str] = None,
        max_docs: int = 100_000,
    ) -> Optional[Dict[str, str]]:
        items = list(self._data.corpus.items())
        # Set from what was cut, as the catalogue's BEIR adapter does.
        self.corpus_truncated = bool(max_docs) and len(items) > max_docs
        return dict(items[:max_docs] if self.corpus_truncated else items)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def resolve_data_path(path: str, base_dir: str = "") -> Path:
    """``path`` against the sweep file's folder, as component references are."""
    p = Path(path).expanduser()
    if not p.is_absolute() and base_dir:
        p = Path(base_dir) / p
    return p.resolve()


def is_local_data_path(value: str) -> bool:
    """Whether a benchmark argument names your own data rather than a catalogue entry.

    A registered name wins, so a folder that happens to be called ``smoke``
    does not shadow the benchmark.
    """
    from .._registry import benchmark_registry

    if value in benchmark_registry:
        return False
    p = Path(value).expanduser()
    return p.suffix.lower() in DATA_SUFFIXES or p.exists()


def register_local_benchmark(path: str, *, base_dir: str = "", name: str = "") -> LocalData:
    """Read, validate and register the data at ``path``; returns it.

    Registered as ``data.benchmark_name(name)`` in the process-wide benchmark
    registry, where every task, corpus and checkpoint looks it up.
    """
    from .._registry import benchmark_registry

    data = read_local_data(resolve_data_path(path, base_dir))
    adapter_cls = LocalBeirAdapter if data.kind == "beir" else LocalFileAdapter
    bench_name = data.benchmark_name(name)
    benchmark_registry.register_adapter(bench_name, adapter_cls(data, bench_name))
    return data


def prepare_local_benchmarks(specs: Iterable[Any], base_dir: str, tracks: Sequence[str]) -> None:
    """Register every benchmark entry given as a ``path`` and name it.

    Called at the top of ``Experiment.generate_tasks``. Sets each such entry's
    ``name`` to the registered, content-hashed name, so the tasks, the results
    and the run metadata all carry it. Calling it again re-reads the file: an
    edit in between gives a new name, which is the point.

    Raises ``ValueError`` when the data supports none of the requested tracks,
    before any model loads.
    """
    for spec in specs:
        if not getattr(spec, "path", ""):
            continue
        data = register_local_benchmark(spec.path, base_dir=base_dir, name=spec.name)
        spec.name = data.benchmark_name(spec.name)
        support = data.track_support()
        if not any(support.get(t) is None for t in tracks):
            raise ValueError(_unsupported_tracks_message(data, list(tracks), support))


def _unsupported_tracks_message(
    data: LocalData, tracks: List[str], support: Dict[str, Optional[str]]
) -> str:
    reasons = "; ".join(f"{support[t]}" for t in tracks)
    fixes = []
    if "retrieval" in tracks:
        fixes.append('Add "contexts" to the rows, or use a BEIR folder, to measure retrieval.')
    if "generation" in tracks:
        fixes.append(
            "A BEIR folder measures retrieval only."
            if data.kind == "beir"
            else 'Add an "answer" to the rows to measure generation.'
        )
    for track in data.supported_tracks():
        fixes.append(
            f"Or run the track it supports: --tracks {track} (tracks: [{track}] in a sweep)."
        )
        if track == "generation" and not data.corpus:
            fixes.append("With no contexts, the closed_book generator answers without retrieval.")
    return (
        f"{data.label} cannot be scored on {' or '.join(tracks)}: {reasons}. "
        f"{' '.join(fixes)} `lasagna benchmark validate-data {_shown(data.path)}` shows what "
        f"the data supports."
    )
