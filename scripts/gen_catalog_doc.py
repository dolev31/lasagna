"""Write the catalog pages in docs/catalog/ from the component and benchmark registries.

    python scripts/gen_catalog_doc.py           # rewrite the pages
    python scripts/gen_catalog_doc.py --check   # exit 1 if a page is out of date

Three pages come out of it:

* ``docs/catalog/stages.md``: the eight stages and the contract each one asks of a
  component, read from the abstract base classes in ``lasagna.core.base_components``.
* ``docs/catalog/components.md``: every component the package registers, by stage,
  with its description and paper from the registry.
* ``docs/catalog/benchmarks.md``: the 31 public benchmarks, with their task, size,
  license and whether they ship a retrievable corpus, from the benchmark registry.

Nothing is downloaded and no model is loaded: building the registries imports
code and reads metadata. The one thing the registries do not record is what a
component needs before it runs (an extra, a package, a key, an LLM); that is
``NEEDS`` below, and the script stops when a registered component has no entry,
so a new component cannot reach the catalog without its requirement being
stated. ``tests/test_catalog_doc_current.py`` runs the check.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "catalog"

REPO_URL = "https://github.com/dolev31/lasagna"

# ---------------------------------------------------------------------------
# What the registries do not record
# ---------------------------------------------------------------------------

#: Stages in the order a document travels through the pipeline.
STAGE_ORDER = [
    "document_processor",
    "chunker",
    "embedder",
    "retriever",
    "query_enhancer",
    "reranker",
    "agentic_pipeline",
    "generator",
]

#: title, base class, the method a component implements, when it runs, the
#: sweep key that lists arms of this stage (None: only through config_overrides),
#: what swapping it changes, and the one-line function form of ``@lasagna.<stage>``.
STAGES: Dict[str, dict] = {
    "document_processor": dict(
        title="Document processor",
        base="BaseDocumentProcessor",
        method="extract",
        when="ingest",
        sweep_key=None,
        changes="how files are parsed into text",
        function="`f(file_path) -> str`",
    ),
    "chunker": dict(
        title="Chunker",
        base="BaseChunker",
        method="chunk",
        when="ingest",
        sweep_key="chunkers",
        changes="how documents are split",
        function="`f(text) -> list[str]`",
    ),
    "embedder": dict(
        title="Embedder",
        base="BaseEmbedder",
        method="encode",
        when="query time",
        sweep_key="embedders",
        changes="what vectors are stored",
        function="`f(texts) -> one vector per text`",
    ),
    "retriever": dict(
        title="Retriever",
        base="BaseRetriever",
        method="retrieve",
        when="query time",
        sweep_key=None,
        changes="how candidates are fetched",
        function="`f(query, top_k) -> list[str]` or documents",
    ),
    "query_enhancer": dict(
        title="Query enhancer",
        base="BaseQueryEnhancer",
        method="enhance",
        when="query time",
        sweep_key="query_enhancers",
        changes="how the query is rewritten before retrieval",
        function="`f(query) -> list[str]`",
    ),
    "reranker": dict(
        title="Reranker",
        base="BaseReranker",
        method="rerank",
        when="query time",
        sweep_key="rerankers",
        changes="the ordering of retrieved candidates",
        function="`f(query, text) -> float`, or `f(query, documents, top_k) -> documents`",
    ),
    "agentic_pipeline": dict(
        title="Agentic pipeline",
        base="BaseAgenticPipeline",
        method="run",
        when="query time",
        sweep_key=None,
        changes="the multi-step retrieval loop",
        function="none: a loop is a class",
    ),
    "generator": dict(
        title="Generator",
        base="BaseGenerator",
        method="generate",
        when="answer time",
        sweep_key="generators",
        changes="how the answer is written from the retrieved passages",
        function="`f(query, contexts) -> str` (the answer)",
    ),
}

LLM = "An LLM"
MODEL = "Downloads a model on first use"
NOTHING = "Nothing extra"

#: What each built-in needs beyond ``pip install lasagna-rag``. Read from the
#: component's code: the extra or package its import names, the key its
#: constructor checks, whether it calls the run's LLM, whether it loads a model.
NEEDS: Dict[Tuple[str, str], str] = {
    ("document_processor", "default"): NOTHING,
    ("document_processor", "docling"): "`pip install 'lasagna-rag[docling]'`",
    ("document_processor", "markitdown"): "`pip install 'lasagna-rag[markitdown]'`",
    ("document_processor", "pymupdf4llm"): "`pip install 'lasagna-rag[pymupdf4llm]'` (AGPL-3.0)",
    ("chunker", "fixed"): NOTHING,
    ("chunker", "recursive"): NOTHING,
    ("chunker", "markdown_header"): NOTHING,
    ("chunker", "sentence_window"): NOTHING,
    ("chunker", "semantic"): "The embedding model",
    ("chunker", "structure_aware"): "The embedding model",
    ("chunker", "hierarchical"): NOTHING,
    ("chunker", "proposition"): LLM,
    ("chunker", "contextual"): LLM,
    ("embedder", "default"): MODEL,
    ("embedder", "sentence_transformers"): MODEL,
    ("embedder", "bge_m3"): "Downloads BAAI/bge-m3 on first use; `[flag-embedding]` adds its sparse and multi-vector outputs",
    ("embedder", "openai"): "`openai` package and `OPENAI_API_KEY`",
    ("embedder", "cohere"): "`pip install cohere` and `COHERE_API_KEY` (or `CO_API_KEY`)",
    ("embedder", "voyage"): "`pip install voyageai` and `VOYAGE_API_KEY`",
    ("retriever", "dense"): NOTHING,
    ("retriever", "bm25"): NOTHING,
    ("retriever", "hybrid_rrf"): NOTHING,
    ("retriever", "hybrid_convex"): NOTHING,
    ("retriever", "splade"): MODEL,
    ("query_enhancer", "hyde"): LLM,
    ("query_enhancer", "query_expansion"): LLM,
    ("query_enhancer", "query_decomposition"): LLM,
    ("query_enhancer", "combined"): LLM,
    ("query_enhancer", "step_back"): LLM,
    ("query_enhancer", "query2doc"): LLM,
    ("query_enhancer", "rewrite"): LLM,
    ("query_enhancer", "multi_query"): LLM,
    ("reranker", "cross_encoder"): MODEL,
    ("reranker", "combined"): MODEL,
    ("reranker", "ensemble"): MODEL,
    ("reranker", "mmr"): NOTHING,
    ("reranker", "colbert"): MODEL,
    ("reranker", "monot5"): MODEL,
    ("reranker", "llm_listwise"): LLM,
    ("reranker", "llm_pointwise"): LLM,
    ("reranker", "flashrank"): "`pip install flashrank`",
    ("reranker", "cohere"): "`pip install cohere` and `COHERE_API_KEY` (or `CO_API_KEY`)",
    ("agentic_pipeline", "default"): LLM,
    ("agentic_pipeline", "legacy_react"): LLM,
    ("agentic_pipeline", "self_ask"): LLM,
    ("agentic_pipeline", "ircot"): LLM,
    ("agentic_pipeline", "flare"): LLM,
    ("agentic_pipeline", "adaptive"): LLM,
    ("agentic_pipeline", "self_rag"): LLM,
    ("generator", "concise"): LLM,
    ("generator", "closed_book"): LLM,
    ("generator", "long_context"): LLM,
    ("generator", "cited"): LLM,
    ("generator", "chain_of_thought"): LLM,
    ("generator", "chain_of_note"): LLM,
}

#: Benchmark categories and modes in words a reader uses.
CATEGORY = {
    "single_hop": "Single-hop QA",
    "multi_hop": "Multi-hop QA",
    "fact_verification": "Fact verification",
    "passage_ranking": "Passage ranking",
}
MODE = {
    "retrieval": "Retrieval",
    "generation": "Generation",
    "end_to_end": "Retrieval and generation",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _paper(source: Optional[str]) -> str:
    """A link for ``arxiv:``/``doi:`` ids and URLs; plain text for a citation."""
    if not source:
        return ""
    if source.startswith("arxiv:"):
        ident = source.split(":", 1)[1]
        return f"[arXiv:{ident}](https://arxiv.org/abs/{ident})"
    if source.startswith("doi:"):
        ident = source.split(":", 1)[1]
        return f"[doi:{ident}](https://doi.org/{ident})"
    if source.startswith("http"):
        host = re.sub(r"^https?://(www\.)?", "", source).split("/")[0]
        return f"[{host}]({source})"
    return _cell(source)


def _count(n: Optional[int]) -> str:
    if n is None:
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 10_000:
        return f"{round(n / 1000)}K"
    if n >= 1000:
        return f"{n / 1000:.1f}K".replace(".0K", "K")
    return str(n)


def _signature(base: type, method: str) -> str:
    sig = inspect.signature(getattr(base, method))
    params = [p for name, p in sig.parameters.items() if name != "self"]
    parts: List[str] = []
    for p in params:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.kind is inspect.Parameter.KEYWORD_ONLY and "*" not in parts:
            parts.append("*")
            parts.append(_param(p))
        else:
            parts.append(_param(p))
    prefix = "async def " if inspect.iscoroutinefunction(getattr(base, method)) else "def "
    ret = sig.return_annotation
    out = prefix + f"{method}({', '.join(parts)})"
    if ret is not inspect.Signature.empty:
        out += f" -> {_annotation(ret)}"
    return out


def _annotation(a) -> str:
    if isinstance(a, str):
        return a.replace("typing.", "").replace("'", "")
    text = getattr(a, "__name__", None) if isinstance(a, type) else None
    return text or str(a).replace("typing.", "")


def _param(p: inspect.Parameter) -> str:
    text = p.name
    if p.annotation is not inspect.Parameter.empty:
        text += f": {_annotation(p.annotation)}"
    if p.default is not inspect.Parameter.empty:
        text += f" = {p.default!r}"
    return text


def _first_line(doc: Optional[str]) -> str:
    return (inspect.cleandoc(doc or "").splitlines() or [""])[0].strip()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

NOTICE = (
    "Generated by `scripts/gen_catalog_doc.py` from the component and benchmark\n"
    "registries. Do not edit by hand: run the script."
)


def _stages_page(components: List[dict]) -> str:
    from lasagna.core import base_components

    by_stage: Dict[str, int] = {}
    for c in components:
        by_stage[c["stage"]] = by_stage.get(c["stage"], 0) + 1

    lines = [
        "# The eight stages",
        "",
        NOTICE,
        "",
        "A RAG pipeline in Lasagna is eight stages behind typed contracts. A component",
        "fills one stage, and the stages around it cannot tell which one is there. That",
        "is what lets a sweep change one stage and hold the rest fixed.",
        "",
        "```text",
        "ingest, chunk, embed, retrieve, enhance the query, rerank, agentic loop, generate",
        "```",
        "",
        "| Stage | Runs at | Swap it to change | Built in | Sweep key |",
        "|---|---|---|--:|---|",
    ]
    for stage in STAGE_ORDER:
        s = STAGES[stage]
        key = f"`{s['sweep_key']}:`" if s["sweep_key"] else "`config_overrides`"
        lines.append(
            f"| [{s['title']}](#{_anchor(s['title'])}) | {s['when']} | {s['changes']} "
            f"| [{by_stage.get(stage, 0)}](components.md#{_anchor(s['title'])}) | {key} |"
        )
    lines += [
        "",
        "Every stage can be named in a `config_overrides` entry with its stage name, for",
        "example `{name: ircot, agentic_pipeline: ircot}`. Five stages also have a list key",
        "that sweeps several components at once. A document processor does not sweep:",
        "benchmark corpora are loaded as text, so check a processor on your own files.",
        "",
        "Registering a component is one decorator. See [Write a component](../guides/write-a-component.md).",
        "",
    ]
    for stage in STAGE_ORDER:
        s = STAGES[stage]
        base = getattr(base_components, s["base"])
        lines += [
            f"## {s['title']}",
            "",
            f"{_first_line(base.__doc__)}",
            "",
            f"Base class: `{s['base']}` in `lasagna.core.base_components`. "
            f"It runs at {s['when']}. Swap it to change {s['changes']}.",
            "",
            "```python",
            _signature(base, s["method"]),
            "```",
            "",
            f"Function form with `@lasagna.{stage}`: {s['function']}.",
            "",
        ]
        if by_stage.get(stage):
            lines += [f"The {by_stage[stage]} built-in components are listed in the [component catalog](components.md#{_anchor(s['title'])}).", ""]
    return "\n".join(lines).rstrip() + "\n"


def _anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).strip().replace(" ", "-")


def _components_page(components: List[dict]) -> str:
    missing = [(c["stage"], c["name"]) for c in components if (c["stage"], c["name"]) not in NEEDS]
    if missing:
        raise SystemExit(
            "scripts/gen_catalog_doc.py: say what these components need in NEEDS: "
            + ", ".join(f"{s}/{n}" for s, n in missing)
        )
    stale = sorted(set(NEEDS) - {(c["stage"], c["name"]) for c in components})
    if stale:
        raise SystemExit(f"scripts/gen_catalog_doc.py: NEEDS names components that are gone: {stale}")

    lines = [
        "# Components",
        "",
        NOTICE,
        "",
        f"{len(components)} components ship with the package, across eight stages. Name one in a",
        "sweep, or run one with `lasagna benchmark run <benchmark> --reranker <name>`. The same",
        "list prints from `lasagna component list`.",
        "",
        "The last column is what the component needs beyond `pip install lasagna-rag`. "
        '"An LLM" means the run\'s LLM, set up as in [Installation](../get-started/installation.md#an-llm-for-the-generation-track). '
        '"Downloads a model" means the weights come from the Hugging Face Hub on first use '
        "(see [Model licenses](../project/model-licenses.md)). Extras are listed in "
        "[Installation](../get-started/installation.md#extras).",
        "",
    ]
    for stage in STAGE_ORDER:
        s = STAGES[stage]
        rows = sorted((c for c in components if c["stage"] == stage), key=lambda c: c["name"])
        lines += [
            f"## {s['title']}",
            "",
            f"{len(rows)} built in. Swap this stage to change {s['changes']}. "
            f"Contract: [`{s['base']}`](stages.md#{_anchor(s['title'])}).",
            "",
            "| Name | What it does | Source | Needs |",
            "|---|---|---|---|",
        ]
        for c in rows:
            lines.append(
                f"| `{c['name']}` | {_cell(c['description'])} | {_paper(c.get('paper'))} "
                f"| {_cell(NEEDS[(stage, c['name'])])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _benchmarks_page(rows: List[Tuple[str, object, bool]]) -> str:
    lines = [
        "# Benchmarks",
        "",
        NOTICE,
        "",
        f"{len(rows)} public benchmarks are registered. List them with `lasagna benchmark list`,",
        "and print the license and citation of each with `lasagna benchmark licenses`.",
        "",
        "Lasagna redistributes no benchmark data. Each dataset is fetched from its own",
        "source on first use, under its own terms, and those terms are yours to honor,",
        "including for anything you publish. The `smoke` benchmark is separate: 24",
        "documents and 12 queries held inside the package, for checking that a component is",
        "wired in. A score on it is never a result.",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| Scores | What the benchmark measures: retrieval, generation, or both |",
        "| Queries, Documents | Size as the adapter records it; blank when the adapter does not say |",
        "| Live | `yes` if the benchmark ships a retrievable corpus, so a run can ingest it and search it. Without one, only generation against gold context can be scored |",
        "| License | As the registry records it. `Varies by dataset` means check the upstream source |",
        "",
        "| Benchmark | Task | Scores | Queries | Documents | License | Live | Source |",
        "|---|---|---|--:|--:|---|:--:|---|",
    ]
    for name, info, live in rows:
        lines.append(
            f"| `{name}` | {CATEGORY[info.category.value]} | {MODE[info.eval_mode.value]} "
            f"| {_count(info.num_queries)} | {_count(info.num_documents)} "
            f"| {_cell(info.license or 'unknown')} | {'yes' if live else '-'} | {_paper(info.paper)} |"
        )
    lines += [
        "",
        "## Descriptions",
        "",
        "| Benchmark | Description |",
        "|---|---|",
    ]
    for name, info, _ in rows:
        lines.append(f"| `{name}` | {_cell(info.description)} |")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load() -> Tuple[List[dict], List[Tuple[str, object, bool]]]:
    import lasagna.components  # noqa: F401  (registers the built-ins)
    from lasagna.core.registry import registry
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
    from lasagna.knowledge_evaluation_sdk.experiment import _adapter_has_corpus

    components = [c for c in registry.list_components() if c["builtin"]]
    rows = []
    for name, info in sorted(benchmark_registry.list_benchmarks().items()):
        if name == "smoke" or name.startswith("private_eval/") or not benchmark_registry.is_builtin(name):
            continue
        rows.append((name, info, bool(_adapter_has_corpus(benchmark_registry.get(name)))))
    return components, rows


def render() -> Dict[str, str]:
    components, benchmarks = _load()
    return {
        "stages.md": _stages_page(components),
        "components.md": _components_page(components),
        "benchmarks.md": _benchmarks_page(benchmarks),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a page is out of date")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    pages = render()
    if args.check:
        stale = [
            name
            for name, text in pages.items()
            if not (OUT_DIR / name).exists() or (OUT_DIR / name).read_text(encoding="utf-8") != text
        ]
        if stale:
            print(
                "docs/catalog/" + ", docs/catalog/".join(stale)
                + " out of date: run python scripts/gen_catalog_doc.py",
                file=sys.stderr,
            )
            return 1
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")
        print(f"wrote docs/catalog/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
