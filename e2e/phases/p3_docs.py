"""Phase 3 — Documentation as executable spec.

Every fenced code block in ``README.md``, ``CONTRIBUTING.md`` and
``docs/guide.md`` that claims to work is either **run** or **skipped with a
stated reason**. Nothing is assumed to work because it looks plausible.

``docs/guide.md`` is the user-facing walkthrough, and it is covered here for a
specific reason: of the 342 runnable blocks across this project's
documentation, only the files named below are executed by anything. The rest
are prose that happens to be formatted as code, and they drift -- an audit
found six documented names that no longer existed, including two exception
classes and a `from lasagna import KnowledgeClient` that never worked. A guide
teaching people how to use the package has to be held to a higher standard than
that, so its examples run on every gate invocation.

What this phase does, per block:

* ``python`` — executed in a subprocess with ``PYTHONPATH=<repo>/src``. A
  non-zero exit is a CRITICAL failure naming the file and the line the block
  starts at.
* ``bash`` — never executed. These install packages, clone repos and start
  servers; running them would mutate the machine and prove little. Instead the
  *claims inside them* are verified statically: every ``lasagna <cmd>`` must
  name a real CLI command, every ``pip install lasagna-rag[<extra>]`` must name
  a real extra, and ``pip install lasagna-rag`` must name the real
  distribution.
* anything else (untagged diagrams, output samples) — skipped with a reason.

**What this phase does NOT prove.**

1. It runs against the *source tree*, not an installed wheel. A module missing
   from the wheel would pass here. Phases 1-2 cover packaging.
2. The bash blocks are read, not run. ``pytest tests/…`` naming a real path is
   not the same as that suite passing.
3. **The README quickstart is not executed.** It is a valid script (it wraps
   its body in ``async def main()`` and calls ``asyncio.run``), so the reason
   is purely environmental: it needs an embedding-model download and a
   populated store, neither of which this hermetic phase provides. This phase
   reports it as SKIPPED, loudly, and separately verifies that the API
   surface it names — ``KnowledgeClient()``, ``await client.upload(...)``,
   ``await client.search_documents(...)``, ``await client.close()``,
   ``results.results[i].score`` and ``.chunk.content`` — actually exists with
   those names and shapes. That catches a renamed method or a changed result
   shape. It does **not** catch a quickstart that runs and returns nothing
   useful. Executing it end to end belongs in a live phase with services up.

   If the quickstart cannot be *found*, that is a CRITICAL failure, not a
   skip: this phase hardcodes what it looks like, and a gate that silently
   stops checking its most important claim is worse than one that fails.

**Hermetic.** Blocks run with a throwaway ``HOME`` (so executing the docs never
touches the developer's real ``~/.rag_knowledge`` store) and with
``HF_HUB_OFFLINE=1`` (so a block that wanted to download half a gigabyte of
weights fails fast instead of silently taking ten minutes). A block that fails
*because* of that offline setting is downgraded from a failure to a SKIP with
that reason — we could not run it, so we must not claim it passed.

**Reading order matters.** A README is read top to bottom, so a block may
legitimately rely on names bound by an earlier block in the same file (the
``registry.list_rerankers()`` example uses the ``registry`` imported three
blocks above). Blocks are therefore executed with a prelude of the earlier
blocks *from the same file that themselves executed successfully*. Skipped
blocks contribute nothing to the prelude, so a skipped block can never make a
later one appear to work. The prelude is recorded in each check's detail and
reflected in its repro command.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..contract import PhaseResult, Severity, Status, timed

#: Derived from this file's location, never hardcoded: the gate has to run on a
#: contributor's machine and in CI, and an absolute developer path both breaks
#: there and writes that developer's home directory into a committed report.
LASAGNA_ROOT = Path(__file__).resolve().parents[2]
LASAGNA_SRC = LASAGNA_ROOT / "src"
PYPROJECT = LASAGNA_ROOT / "pyproject.toml"

DOC_FILES = ("README.md", "CONTRIBUTING.md", "docs/guide.md")

#: Generous: a cold import of the package pulls in torch and
#: sentence-transformers (~5 s idle, much worse on a cold FS cache). A timeout
#: is reported as a FAILURE, not a skip — a documented snippet that hangs for
#: four minutes is broken from a reader's point of view.
BLOCK_TIMEOUT_S = 240
PROBE_TIMEOUT_S = 300

PY_LANGS = {"python", "py", "python3"}
SH_LANGS = {"bash", "sh", "shell", "console", "zsh"}

#: Substrings that mean a block talks to something this gate will not stand up:
#: a service, a network host, or a paid API. Conservative on purpose — a false
#: positive costs one skipped block, a false negative costs a bogus failure.
NEEDS_SERVICE_TOKENS = (
    "KnowledgeClient",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "http://",
    "https://",
    "requests.",
    "httpx.",
    "openai.",
)

#: Error fragments that mean "this block wanted the network and we forbade it".
#: Such a block was not validated, so it is downgraded to a SKIP rather than
#: reported as a failure of the documentation.
OFFLINE_MARKERS = (
    "HF_HUB_OFFLINE",
    "offline mode",
    "OfflineModeIsEnabled",
    "Couldn't reach",
    "couldn't connect",
    "Connection error",
    "Name or service not known",
    "Temporary failure in name resolution",
    "nodename nor servname",
)

#: A fence, optionally behind blockquote markers or indentation.
#:
#: Anchoring this at column 0 (the obvious first cut) silently drops every
#: fence inside a blockquote or a list item. That is not an under-report, it is
#: a hole: the README's ``> ```bash`` / ``> lasagna configure --profile speed``
#: tip lives in a blockquote, so a typo there was invisible to this phase while
#: it reported PASSED. A block this phase cannot see is a block nobody checked.
FENCE_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?:>[ \t]?)*)(?P<ticks>`{3,})[ \t]*(?P<lang>[A-Za-z0-9_+.-]*)[ \t]*$"
)

_QUOTE_RE = re.compile(r"^[ \t]*>[ \t]?")


def _strip_quote(line: str, depth: int, indent: str) -> str:
    """Remove ``depth`` blockquote markers, then the opening fence's indent."""
    for _ in range(depth):
        m = _QUOTE_RE.match(line)
        if not m:
            break
        line = line[m.end() :]
    if indent and line.startswith(indent):
        line = line[len(indent) :]
    return line


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """One fenced code block, located precisely enough to reproduce."""

    filename: str
    #: 1-based line number of the *opening* fence, which is what a reader sees
    #: when a check says "README.md:79".
    start_line: int
    lang: str
    code: str
    #: 1-based line range of the code itself, for the ``sed`` repro command.
    first_code_line: int = 0
    last_code_line: int = 0
    #: Number of ``>`` markers the fence sits behind, and the indent before it.
    #: Both are stripped from ``code``; they are kept so the repro command can
    #: strip them too, otherwise the repro feeds ``> lasagna …`` to a shell.
    quote_depth: int = 0
    indent: str = ""
    #: False when the document ended before the closing fence — malformed
    #: markdown, reported rather than silently swallowed.
    closed: bool = True

    @property
    def label(self) -> str:
        return f"{self.filename}:{self.start_line} [{self.lang or 'untagged'}]"

    @property
    def sed_range(self) -> str:
        cmd = f"sed -n '{self.first_code_line},{self.last_code_line}p' {self.filename}"
        if self.quote_depth:
            cmd += " | sed -E 's/^[[:space:]]*>[[:space:]]?//'"
        elif self.indent:
            cmd += f" | sed -E 's/^{re.escape(self.indent)}//'"
        return cmd


def _extract_blocks(path: Path) -> List[Block]:
    """Fenced blocks with their line numbers, including quoted/indented ones.

    Blockquote markers and the opening fence's indentation are stripped from
    the recorded code, so a quoted block is checked as the reader would run it.
    An unterminated fence is recorded with ``closed=False`` rather than
    swallowing the remainder of the document into one bogus block.
    """
    blocks: List[Block] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        ticks, lang = m.group("ticks"), m.group("lang").lower()
        prefix = m.group("prefix")
        depth = prefix.count(">")
        indent = prefix if depth == 0 else ""
        start = i + 1  # 1-based line of the opening fence
        body: List[str] = []
        j = i + 1
        closed = False
        while j < len(lines):
            close = FENCE_RE.match(lines[j])
            if (
                close
                and not close.group("lang")
                and len(close.group("ticks")) >= len(ticks)
                and close.group("prefix").count(">") == depth
            ):
                closed = True
                break
            body.append(_strip_quote(lines[j], depth, indent))
            j += 1
        blocks.append(
            Block(
                filename=path.name,
                start_line=start,
                lang=lang,
                code="\n".join(body) + ("\n" if body else ""),
                first_code_line=start + 1,
                last_code_line=start + len(body),
                quote_depth=depth,
                indent=indent,
                closed=closed,
            )
        )
        i = j + 1 if closed else j
    return blocks


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _is_all_comments(code: str) -> bool:
    """True when *every* meaningful line is a comment — an output sample.

    The tempting version of this test is "the first meaningful line starts with
    ``#``". It is wrong, and wrong in the dangerous direction: a perfectly
    runnable block that opens with an explanatory comment (``# Register your
    component``) is then skipped whole, so anything broken underneath it is
    never executed and the phase still reports PASSED. Requiring *all* lines to
    be comments keeps the output-sample case without handing every block a
    one-line opt-out from being checked.
    """
    meaningful = [ln.strip() for ln in code.splitlines() if ln.strip()]
    return bool(meaningful) and all(ln.startswith("#") for ln in meaningful)


def _code_without_comments(code: str, tree: Optional[ast.AST]) -> str:
    """The block's code with comments removed, for token scanning.

    Scanning the raw text means a comment can disarm a check: ``# see
    https://example.com`` above working code makes the block look like it needs
    the network, so it is skipped and never run. Round-tripping through the AST
    drops comments while keeping identifiers and string literals (a URL in a
    real string still counts — that one plausibly does dial out).
    """
    if tree is None:
        return code
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def _parse(code: str) -> Tuple[Optional[ast.AST], bool, str]:
    """Parse a block. Returns (tree, needs_top_level_await, syntax_error).

    Top-level ``await`` is not valid in a module, so a block using it cannot be
    run as a script no matter what services are up. It is still valid *doc* —
    the reader is assumed to be in a notebook — so it is classified separately
    from a genuine syntax error.

    ``ast.parse`` alone is not enough to detect it: "'await' outside function"
    is raised at the *compile* stage, not by the grammar, so ``ast.parse``
    accepts it. Both stages are therefore run. The same is true of module-level
    ``return``, ``yield`` and ``nonlocal``, which is why the syntax-error repro
    emitted for this check uses ``compile`` rather than ``ast.parse``.
    """
    try:
        tree: Optional[ast.AST] = ast.parse(code)
    except SyntaxError as exc:
        try:
            return (
                compile(
                    code, "<block>", "exec", ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
                ),
                True,
                "",
            )
        except SyntaxError:
            return None, False, f"{exc.msg} (line {exc.lineno})"

    try:
        compile(code, "<block>", "exec")
    except SyntaxError as exc:
        try:
            compile(code, "<block>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            return tree, True, ""
        except SyntaxError:
            return tree, False, f"{exc.msg} (line {exc.lineno})"
    return tree, False, ""


def _has_top_level_ellipsis(tree: ast.AST) -> bool:
    """``...`` as a *module-level* statement — an elided fragment.

    Note this is narrower than "the block contains ``...``". An ellipsis used as
    a function or method body (``def rerank(...): ...``) is perfectly valid
    Python that executes fine, and the README's component-registration example
    is exactly that shape. Skipping it would mean reporting "not validated" for
    a block that does in fact run — and that block is one of the more load-
    bearing claims in the README (that ``@registry.register(..., paper=...,
    claims=...)`` accepts a ``BaseReranker`` subclass), so it is run.

    What running it does NOT prove: the elided ``rerank`` body. Registration
    succeeding says the decorator and the base class accept the component, not
    that the component reranks anything.
    """
    body = getattr(tree, "body", [])
    return any(
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is Ellipsis
        for n in body
    )


def _skip_reasons(block: Block, tree: Optional[ast.AST], needs_await: bool) -> List[str]:
    """Why this python block cannot be executed here. Empty list ⇒ run it."""
    reasons: List[str] = []
    code = block.code

    if _is_all_comments(code):
        reasons.append("output sample (block is comment lines showing results, not code to run)")
    if re.search(r"^\s*#\s*WARNING\b", code, re.MULTILINE):
        reasons.append("contains an inlined output sample (`# WARNING …`), not runnable input")
    if needs_await:
        reasons.append("uses top-level `await`: written for a notebook/async REPL, not a script")

    # Scan the comment-free form: see _code_without_comments.
    hits = sorted({t for t in NEEDS_SERVICE_TOKENS if t in _code_without_comments(code, tree)})
    if hits:
        reasons.append(
            "needs services / network / a model download (mentions: " + ", ".join(hits) + ")"
        )
    if tree is not None and _has_top_level_ellipsis(tree):
        reasons.append("illustrative fragment: `...` appears as a module-level statement")
    return reasons


# ---------------------------------------------------------------------------
# Static facts about the package, read from source (not from the docs)
# ---------------------------------------------------------------------------


def _load_pyproject() -> Tuple[Optional[dict], str]:
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:  # pragma: no cover - py3.10
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return None, "no TOML parser (python < 3.11 and tomli not installed)"
    try:
        return tomllib.loads(PYPROJECT.read_text(encoding="utf-8")), ""
    except Exception as exc:
        return None, f"could not parse {PYPROJECT}: {type(exc).__name__}: {exc}"


_CLI_PROBE = r"""
import json
from lasagna.cli import app

names = set()
for c in app.registered_commands:
    names.add(c.name or c.callback.__name__.replace("_", "-"))
for g in app.registered_groups:
    n = g.name or getattr(getattr(g, "typer_instance", None), "info", None)
    names.add(n if isinstance(n, str) else getattr(n, "name", ""))
print("===CLI===" + json.dumps(sorted(x for x in names if x)))
"""

#: Verifies the *names and shapes* the README quickstart uses. See the module
#: docstring for what this deliberately does not cover.
_QUICKSTART_SURFACE_PROBE = r"""
import dataclasses, inspect, json, typing

missing = []
from lasagna.sdk import KnowledgeClient
from lasagna.sdk.types import SearchResults

sig = inspect.signature(KnowledgeClient.__init__)
required = [
    n for n, p in list(sig.parameters.items())[1:]
    if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
]
if required:
    missing.append("KnowledgeClient() takes required args %s, README calls it bare" % required)

for meth in ("upload", "search_documents", "close"):
    fn = getattr(KnowledgeClient, meth, None)
    if fn is None:
        missing.append("KnowledgeClient.%s is missing" % meth)
    elif not inspect.iscoroutinefunction(fn):
        missing.append("KnowledgeClient.%s is not awaitable, README awaits it" % meth)

names = {f.name for f in dataclasses.fields(SearchResults)}
if "results" not in names:
    missing.append("SearchResults.results is missing (README does results.results)")
else:
    elem = typing.get_args(typing.get_type_hints(SearchResults)["results"])[0]
    efields = {f.name for f in dataclasses.fields(elem)}
    if "score" not in efields:
        missing.append("%s.score is missing (README does r.score)" % elem.__name__)
    chunk = typing.get_type_hints(elem).get("chunk")
    if chunk is None or not dataclasses.is_dataclass(chunk):
        missing.append("%s.chunk is missing (README does r.chunk.content)" % elem.__name__)
    elif "content" not in {f.name for f in dataclasses.fields(chunk)}:
        missing.append("%s.content is missing (README does r.chunk.content)" % chunk.__name__)

print("===SURFACE===" + json.dumps(missing))
"""


def _bash_lasagna_commands(code: str) -> List[Tuple[int, str]]:
    """``lasagna <cmd>`` invocations, as (line offset within block, subcommand).

    Only matches when ``lasagna`` is the first token of a command position, so
    ``cd lasagna`` and ``pip install lasagna-rag`` are not mistaken for CLI
    invocations.
    """
    found: List[Tuple[int, str]] = []
    for offset, raw in enumerate(code.splitlines()):
        line = raw.split("#", 1)[0].strip()
        for segment in re.split(r"&&|\|\||;|\|", line):
            toks = segment.split()
            if len(toks) >= 2 and toks[0] == "lasagna":
                found.append((offset, toks[1]))
    return found


def _bash_extras(code: str) -> List[str]:
    """Extras named by any ``pip install`` in the block."""
    extras: List[str] = []
    for pattern in (r"lasagna-rag\[([^\]]+)\]", r"[\"']?\.\[([^\]]+)\][\"']?"):
        for m in re.finditer(pattern, code):
            extras.extend(e.strip() for e in m.group(1).split(",") if e.strip())
    return sorted(set(extras))


def _installs_lasagna(code: str) -> bool:
    return bool(re.search(r"pip\s+install[^\n]*lasagna-rag", code))


def _bash_pytest_paths(code: str) -> List[str]:
    """Test paths named by any ``pytest`` invocation, including ``--ignore=``.

    Checking these exist is not the same as checking they pass — that is what
    a test phase is for. It catches the cheaper and commoner rot: a README that
    still tells a contributor to run a file that was renamed or deleted.
    """
    paths: List[str] = []
    for raw in code.splitlines():
        line = raw.split("#", 1)[0].strip()
        for segment in re.split(r"&&|\|\||;|\|", line):
            toks = segment.split()
            if not toks or toks[0] != "pytest":
                continue
            for tok in toks[1:]:
                if tok.startswith("--ignore="):
                    tok = tok.split("=", 1)[1]
                elif tok.startswith("-"):
                    continue
                paths.append(tok.strip("\"'").rstrip("/"))
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


#: Environment that would point a doc snippet at a *real* backend or a paid
#: API. Overriding ``HOME`` protects ``~/.rag_knowledge``, but nothing about
#: ``HOME`` stops ``RAG_VECTOR_BACKEND=pgvector`` + ``PGVECTOR_DSN=<prod>`` in
#: the developer's shell from being inherited straight into the subprocess —
#: at which point the gate is writing to a live store and its "hermetic" claim
#: is false. Names are matched by prefix or suffix.
_UNSAFE_ENV_PREFIXES = (
    "RAG_",
    "LASAGNA_",
    "MILVUS_",
    "PGVECTOR_",
    "QDRANT_",
    "WEAVIATE_",
    "PINECONE_",
    "OPENSEARCH_",
    "OLLAMA_",
    "VLLM_",
    "WATSONX_",
    "OPENROUTER_",
    "OPENAI_",
    "ANTHROPIC_",
    "GOOGLE_",
    "HF_MODEL_",
    "LLAMACPP_",
)
_UNSAFE_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_DSN", "_URI", "_URL")


def _child_env(home: Path) -> Dict[str, str]:
    env = dict(os.environ)
    for key in [
        k for k in env if k.startswith(_UNSAFE_ENV_PREFIXES) or k.endswith(_UNSAFE_ENV_SUFFIXES)
    ]:
        env.pop(key, None)

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{LASAGNA_SRC}{os.pathsep}{existing}" if existing else str(LASAGNA_SRC)
    # A throwaway HOME: importing lasagna auto-configures a store under
    # ~/.rag_knowledge. Running the docs must not write into the developer's
    # real one. (Verified: lasagna resolves the store via Path.home(), which
    # reads HOME, so this is sufficient for the store itself.)
    env["HOME"] = str(home)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env.pop("PYTHONSTARTUP", None)
    return env


#: Prefix for every emitted repro command, so that running a repro reproduces
#: the *conditions of the verdict* and does not quietly write the developer's
#: real ``~/.rag_knowledge`` while doing it.
REPRO_ENV = 'HOME="$(mktemp -d)" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src'


def _run_script(python: str, code: str, env: Dict[str, str], timeout: int) -> Tuple[int, str]:
    """Run ``code`` on stdin. Returns (returncode, combined tail of output)."""
    try:
        proc = subprocess.run(
            [python, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(LASAGNA_ROOT),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _tail(text: str, n: int = 12) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _pick_python(env: Dict[str, str]) -> Tuple[Optional[str], str]:
    """First interpreter that can ``import lasagna`` with the source on path.

    ``E2E_PYTHON`` first so CI can point this at whatever it built, then the
    interpreter running the gate. There is deliberately no hardcoded fallback:
    one pointing at a particular developer's virtualenv works on exactly one
    machine and leaks that path into every report the gate commits.
    """
    detail = "no candidate interpreter found"
    for cand in (os.environ.get("E2E_PYTHON"), sys.executable):
        if not cand or not Path(cand).exists():
            continue
        rc, out = _run_script(cand, "import lasagna\n", env, PROBE_TIMEOUT_S)
        if rc == 0:
            return cand, ""
        detail = f"{cand}: import lasagna failed\n{_tail(out, 6)}"
    return None, detail


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


@dataclass
class _Counters:
    """Block accounting. ``executed + static + skipped + invalid`` must equal
    ``found`` — a block in none of those buckets is a block nobody looked at,
    which is the failure mode this phase exists to make impossible."""

    found: int = 0
    #: python blocks run in a subprocess (whether or not they exited 0)
    executed: int = 0
    #: bash blocks whose claims were verified without running them
    static: int = 0
    #: blocks not run and not asserted on, each with a stated reason
    skipped: int = 0
    #: blocks that were not valid Python (never ran, still broken docs)
    invalid: int = 0
    #: how many python blocks existed at all, to detect "validated nothing"
    py_found: int = 0
    failed: int = 0
    exec_s: float = 0.0


def run() -> PhaseResult:  # noqa: C901 - one linear pass over the blocks
    p = PhaseResult(key="p3", title="Documentation as executable spec")
    c = _Counters()
    home: Optional[Path] = None

    try:
        # Inside the guard: mkdtemp is the one line here that can raise before
        # the phase has a chance to report anything.
        home = Path(tempfile.mkdtemp(prefix="lasagna-e2e-p3-home-"))
        _run_phase(p, c, home)
    except Exception as exc:  # never raise out of run()
        p.check(
            "p3 completed without crashing",
            False,
            detail=f"{type(exc).__name__}: {exc}",
            repro="python -m e2e.run --phase p3",
        )
    finally:
        # Rule 1: metrics land even when the phase failed halfway.
        p.measure(
            "doc blocks found",
            c.found,
            "blocks",
            lower_is_better=False,
            note=f"fenced blocks in {', '.join(DOC_FILES)}",
        )
        p.measure(
            "doc blocks executed",
            c.executed,
            "blocks",
            lower_is_better=False,
            note="python blocks actually run in a subprocess",
        )
        p.measure(
            "doc blocks statically verified",
            c.static,
            "blocks",
            lower_is_better=False,
            note="bash blocks: claims inside checked against the package, never run",
        )
        p.measure(
            "doc blocks skipped",
            c.skipped,
            "blocks",
            lower_is_better=True,
            note="not run and not asserted on: output samples, illustrative fragments",
        )
        p.measure(
            "doc blocks failed",
            c.failed,
            "blocks",
            lower_is_better=True,
            note="executed python blocks that exited non-zero, plus invalid-syntax blocks",
        )
        p.measure(
            "doc block execution time",
            round(c.exec_s, 1),
            "s",
            note="wall clock, subprocess startup included (torch import dominates)",
        )
        if home is not None:
            shutil.rmtree(home, ignore_errors=True)

    return p


def _run_phase(p: PhaseResult, c: _Counters, home: Path) -> None:
    missing = [f for f in DOC_FILES if not (LASAGNA_ROOT / f).exists()]
    if missing:
        # Nothing to check at all — the phase did not run, which is not a pass.
        p.skipped_reason = f"documentation not found: {', '.join(missing)} under {LASAGNA_ROOT}"
        return

    env = _child_env(home)
    python, why_no_python = _pick_python(env)
    # Why the phase may end up unvalidated. Applied at the very end rather than
    # here, because ``skipped_reason`` short-circuits ``PhaseResult.status``:
    # setting it now would render a run that went on to *find a real break* as
    # SKIPPED. "We could not check" and "we checked and it is broken" are
    # different verdicts, and the second one must win.
    unvalidated: str = ""
    if python is None:
        # The static bash checks below still run and are still reported, but
        # the phase's actual job — running the documentation — did not happen.
        # Reporting PASSED off the back of the checks that *did* run would be
        # the exact "nobody checked" → "checked and fine" slippage this gate
        # exists to prevent, so the phase reports SKIPPED (unvalidated).
        unvalidated = (
            "no python blocks were executed: " + why_no_python + " — any checks below "
            "come from static inspection only"
        )

    pyproject, pyproject_err = _load_pyproject()
    extras: Optional[Set[str]] = None
    dist_name: Optional[str] = None
    if pyproject is not None:
        extras = set(pyproject.get("project", {}).get("optional-dependencies", {}))
        dist_name = pyproject.get("project", {}).get("name")

    # Real CLI command names, read from the package rather than from the docs.
    cli_commands: Optional[Set[str]] = None
    cli_err = why_no_python
    if python:
        rc, out = _run_script(python, _CLI_PROBE, env, PROBE_TIMEOUT_S)
        marker = "===CLI==="
        if rc == 0 and marker in out:
            cli_commands = set(json.loads(out.split(marker, 1)[1].splitlines()[0]))
        else:
            cli_err = f"could not enumerate CLI commands (rc={rc})\n{_tail(out, 8)}"

    quickstart_block: Optional[Block] = None

    for filename in DOC_FILES:
        path = LASAGNA_ROOT / filename
        blocks = _extract_blocks(path)
        # A README is read top to bottom; see module docstring.
        prelude: List[Block] = []

        for block in blocks:
            c.found += 1
            name = block.label

            if not block.closed:
                c.invalid += 1
                c.failed += 1
                p.check(
                    name,
                    False,
                    detail="unterminated code fence: the document ends before the closing "
                    "```. Everything after it is being read as code, so the blocks below "
                    "it are not being checked.",
                    repro=f"cd {LASAGNA_ROOT} && grep -n '^\\s*>*\\s*```' {filename}",
                )
                continue

            if block.lang in PY_LANGS:
                c.py_found += 1
                if "KnowledgeClient" in block.code and filename == "README.md":
                    if quickstart_block is None:
                        quickstart_block = block
                ran = _handle_python(p, c, block, prelude, python, why_no_python, env)
                if ran:
                    prelude.append(block)
            elif block.lang in SH_LANGS:
                _handle_bash(p, c, block, cli_commands, cli_err, extras, pyproject_err, dist_name)
            else:
                c.skipped += 1
                p.skip(
                    name,
                    "no language tag: prose/diagram block, nothing to execute or verify",
                    severity=Severity.WARN,
                )

    _handle_quickstart(p, quickstart_block, python, why_no_python, env)

    # The phase's headline promise, asserted rather than trusted. If the
    # extractor ever drifts (a fence shape it does not recognise, a bucket that
    # forgets to count) this fails instead of quietly checking less.
    accounted = c.executed + c.static + c.skipped + c.invalid
    p.check(
        "every documented block was run, statically verified, or skipped with a reason",
        accounted == c.found,
        detail=(
            f"found {c.found}; executed {c.executed}, statically verified {c.static}, "
            f"skipped {c.skipped}, invalid {c.invalid} (total {accounted})"
        ),
        repro=f"cd {LASAGNA_ROOT} && grep -cn '^\\s*>*\\s*```' {' '.join(DOC_FILES)}",
    )

    # Honesty rule, generalised. The no-interpreter case above was one route to
    # "the phase asserted nothing about the documentation and still said
    # PASSED"; every python block being skipped for any other reason is
    # another, and the surviving bash checks are happy to carry the phase to
    # green on their own. If nothing ran, the phase is unvalidated.
    if c.py_found and c.executed == 0 and not unvalidated:
        unvalidated = (
            f"none of the {c.py_found} documented python blocks were executed, so this "
            "phase validated no documentation. Any passing checks below come from static "
            "inspection of the bash blocks only. See the per-block skip reasons."
        )

    if unvalidated and not any(ch.status is Status.FAILED for ch in p.checks):
        p.skipped_reason = unvalidated


def _handle_python(
    p: PhaseResult,
    c: _Counters,
    block: Block,
    prelude: Sequence[Block],
    python: Optional[str],
    why_no_python: str,
    env: Dict[str, str],
) -> bool:
    """Run one python block. Returns True iff it executed and exited 0."""
    name = block.label
    tree, needs_await, syntax_err = _parse(block.code)

    if syntax_err:
        # Not a skip: a documented python block that is not valid Python is
        # broken documentation, full stop.
        c.invalid += 1
        c.failed += 1
        p.check(
            name,
            False,
            detail=f"block is not valid Python: {syntax_err}",
            # compile(), not ast.parse(). ast.parse accepts module-level
            # `return`/`yield`/`nonlocal`/`await` — the grammar allows them and
            # the compile stage rejects them — so an ast.parse repro exits 0
            # for a whole class of the failures reported here, which makes it
            # a repro that does not reproduce.
            repro=f"cd {LASAGNA_ROOT} && {block.sed_range} | python -c "
            f'\'import sys; compile(sys.stdin.read(), "<block>", "exec")\'',
        )
        return False

    reasons = _skip_reasons(block, tree, needs_await)
    if reasons:
        c.skipped += 1
        p.skip(name, "not executed — " + "; ".join(reasons), severity=Severity.WARN)
        return False

    if python is None:
        c.skipped += 1
        p.skip(name, f"no interpreter can import lasagna: {why_no_python}")
        return False

    parts = [b.code for b in prelude] + [block.code]
    script = "\n".join(parts)
    seds = "; ".join(b.sed_range for b in list(prelude) + [block])
    # The repro carries the same throwaway HOME and offline settings the
    # verdict was produced under. Without them the reader reproduces under
    # different conditions *and* runs the documentation against their real
    # ~/.rag_knowledge store, which this phase went to some trouble to avoid.
    repro = f"cd {LASAGNA_ROOT} && {{ {seds}; }} | {REPRO_ENV} {python} -"

    with timed() as t:
        rc, out = _run_script(python, script, env, BLOCK_TIMEOUT_S)
    c.exec_s += t.elapsed

    prelude_note = (
        f" (with prelude: {', '.join(b.label for b in prelude)})" if prelude else " (standalone)"
    )

    if rc != 0 and any(m.lower() in out.lower() for m in OFFLINE_MARKERS):
        # We forbade the network, so this block was not validated. Reporting it
        # as a documentation failure would be wrong; reporting it as a pass
        # would be worse.
        c.skipped += 1
        p.skip(
            name,
            "not executed — needs the network/model weights, which this hermetic "
            f"gate blocks (HF_HUB_OFFLINE=1):\n{_tail(out, 6)}",
            severity=Severity.WARN,
        )
        return False

    ok = rc == 0
    c.executed += 1  # it ran either way; whether it worked is the check
    if not ok:
        c.failed += 1

    p.check(
        name,
        ok,
        detail=(
            f"executed{prelude_note}, exit {rc}"
            + ("" if ok else f"\n{_tail(out)}")
            + ("\n(timed out)" if rc == 124 else "")
        ),
        repro=repro,
        duration_s=round(t.elapsed, 2),
    )
    return ok


def _handle_bash(
    p: PhaseResult,
    c: _Counters,
    block: Block,
    cli_commands: Optional[Set[str]],
    cli_err: str,
    extras: Optional[Set[str]],
    pyproject_err: str,
    dist_name: Optional[str],
) -> None:
    """Bash blocks are read, never run. See module docstring."""
    name = block.label
    invocations = _bash_lasagna_commands(block.code)
    named_extras = _bash_extras(block.code)
    installs = _installs_lasagna(block.code)

    verified_something = False

    if invocations:
        verified_something = True
        if cli_commands is None:
            p.skip(f"{name} lasagna commands exist", f"could not read the CLI: {cli_err}")
        else:
            bad = sorted({cmd for _, cmd in invocations if cmd not in cli_commands})
            used = sorted({cmd for _, cmd in invocations})
            p.check(
                f"{name} lasagna commands exist",
                not bad,
                detail=(
                    f"documented: {used}; registered: {sorted(cli_commands)}"
                    + (f"; NOT A COMMAND: {bad}" if bad else "")
                ),
                repro=f"cd {LASAGNA_ROOT} && PYTHONPATH=src python -m lasagna --help",
            )

    if named_extras or installs:
        verified_something = True
        if extras is None or dist_name is None:
            p.skip(f"{name} pip install target is real", pyproject_err or "pyproject unreadable")
        else:
            problems = []
            if installs and dist_name != "lasagna-rag":
                problems.append(f"docs install 'lasagna-rag' but pyproject name is '{dist_name}'")
            bad = sorted(e for e in named_extras if e not in extras)
            if bad:
                problems.append(f"undefined extras: {bad}")
            p.check(
                f"{name} pip install target is real",
                not problems,
                detail=(
                    f"extras used: {named_extras or '[]'}; defined: {sorted(extras)}"
                    + ("; " + "; ".join(problems) if problems else "")
                ),
                repro=f'cd {LASAGNA_ROOT} && python -c "import tomllib,pathlib;'
                f"d=tomllib.loads(pathlib.Path('pyproject.toml').read_text());"
                f"print(d['project']['name'], sorted(d['project']['optional-dependencies']))\"",
            )

    test_paths = _bash_pytest_paths(block.code)
    if test_paths:
        verified_something = True
        absent = sorted(pth for pth in test_paths if not (LASAGNA_ROOT / pth).exists())
        p.check(
            f"{name} pytest paths exist",
            not absent,
            # CRITICAL. This was WARN, on the reasoning that a stale path is
            # "doc rot, not a broken package" — but WARN in this contract means
            # "a performance threshold a busy laptop can miss", and the release
            # proceeds. A README that tells a contributor to run a file that no
            # longer exists is a documented command that fails on the first
            # try: the same class of defect as `lasagna teleport`, which is
            # CRITICAL two checks up. What it still does not claim is that
            # those tests *pass* — that is a test phase's job.
            detail=(
                f"paths named: {test_paths}"
                + (f"; MISSING: {absent}" if absent else "; all present (not run)")
            ),
            repro=f"cd {LASAGNA_ROOT} && ls -d {' '.join(test_paths)}",
        )

    if verified_something:
        c.static += 1
    else:
        c.skipped += 1
        p.skip(
            name,
            "not executed (bash blocks install packages / clone repos / start servers) "
            "and contains no `lasagna <cmd>`, `pip install lasagna-rag` or `pytest <path>` "
            "claim to verify statically",
            severity=Severity.WARN,
        )


def _handle_quickstart(
    p: PhaseResult,
    block: Optional[Block],
    python: Optional[str],
    why_no_python: str,
    env: Dict[str, str],
) -> None:
    """The README quickstart: say loudly that it was not run, then check what can be.

    The quickstart working verbatim is the single most valuable thing this
    phase could prove, and it is precisely the thing it cannot prove here: the
    snippet uses top-level ``await`` and needs an embedding model plus an
    ingested document. So it is reported as NOT VALIDATED, and the API surface
    it names is checked instead.
    """
    if block is None:
        # A failure, not a skip. This phase hardcodes what the quickstart looks
        # like; if that goes stale, both the "not validated" notice and the API
        # surface check below silently stop being emitted, and the phase keeps
        # reporting PASSED while no longer checking the single most important
        # thing in the README. A gate whose coverage can shrink to nothing
        # without saying so is the defect, so losing the quickstart blocks the
        # release until someone re-points this phase at it.
        p.check(
            "README quickstart located",
            False,
            detail="no README python block referencing KnowledgeClient was found. Either "
            "the quickstart moved/was renamed, or this phase's detection is stale — "
            "either way the quickstart API-surface check below did not run, so the "
            "quickstart is entirely unchecked. Re-point _handle_quickstart at it.",
            repro=f"cd {LASAGNA_ROOT} && grep -n 'KnowledgeClient' README.md",
        )
        p.skip(
            "README quickstart API surface exists",
            "not attempted: the quickstart block could not be located (see above)",
        )
        return

    p.skip(
        "README quickstart runs verbatim (end to end)",
        f"NOT VALIDATED. The quickstart at {block.filename}:{block.start_line} was NOT executed: "
        "it requires an embedding-model "
        "download plus an ingested document. This phase proves only that the names it uses "
        "exist (see the next check). Nobody has run the quickstart end to end in this gate — "
        "that needs a live phase with services up and network access.",
    )

    if python is None:
        p.skip(
            "README quickstart API surface exists",
            f"no interpreter can import lasagna: {why_no_python}",
        )
        return

    with timed() as t:
        rc, out = _run_script(python, _QUICKSTART_SURFACE_PROBE, env, PROBE_TIMEOUT_S)
    marker = "===SURFACE==="
    repro = (
        f"cd {LASAGNA_ROOT} && {REPRO_ENV} {python} -c "
        '"import inspect;from lasagna.sdk import KnowledgeClient;'
        'print(inspect.signature(KnowledgeClient.search_documents))"'
    )

    if rc != 0 or marker not in out:
        p.check(
            "README quickstart API surface exists",
            False,
            detail=f"surface probe failed (rc={rc})\n{_tail(out)}",
            repro=repro,
            duration_s=round(t.elapsed, 2),
        )
        return

    problems = json.loads(out.split(marker, 1)[1].splitlines()[0])
    p.check(
        "README quickstart API surface exists",
        not problems,
        detail=(
            "KnowledgeClient(), await upload(), await search_documents(), "
            "results.results[i].score and .chunk.content all resolve"
            if not problems
            else "; ".join(problems)
        ),
        repro=repro,
        duration_s=round(t.elapsed, 2),
    )
