"""Phase 4 — Core journey.

The workflow this package exists for: a researcher registers a custom
component, finds it in the registry next to comparable baselines, swaps it into
a pipeline, and runs it. If that does not work, nothing else about the package
matters — so every correctness check here is CRITICAL.

**What "swaps it into a pipeline" means here.** ``replace()`` writing a name
into ``_component_names`` is not the claim; the claim is that the pipeline then
*hands out the component you asked for*. Those are different assertions, and
only the second one is what a user experiences. ``replace()`` also has to
invalidate the pipeline's instance cache — a pipeline that already resolved its
reranker will otherwise keep serving the old one forever, with the name field
reporting the new one. Step 6 therefore resolves through
``pipeline.get_component()`` after populating that cache, and step 8 runs the
reranker through the pipeline-resolved instance rather than through the
registry, so a swap that does not reach execution fails the gate.

**Fidelity limitation.** This phase runs against the *source tree*
(``PYTHONPATH=<repo>/src``), not an installed wheel. It therefore proves the
code works; it does *not* prove the packaging works. A module missing from the
wheel, a bad ``packages =`` glob, or a missing runtime dependency in
``pyproject.toml`` would all pass here and fail for a real user. Phases 1-3
(install / import / wheel contents) are what cover that. The trade-off is
deliberate: this phase is about the API surface, and running it in-tree keeps
it fast and independent of whether the build succeeded.

**Hermetic, and it checks that it is.** No model downloads, no network, no
services, no database. The registry and pipeline APIs are exercised directly
with hand-built ``ScoredDocument`` objects. The subprocess runs with
``HF_HUB_OFFLINE=1``, a throwaway ``HOME``, LLM credentials stripped from the
environment, and a throwaway *working directory*. Steps 1-2 assert that this
actually held, because "hermetic" written in a docstring is not a property, it
is a hope: with live keys in the environment, a component that quietly called
out to a provider would pass here and fail in CI. Step 1 covers what this phase
controls (HOME, inherited vars) and is CRITICAL; step 2 covers credentials a
*dependency* injects at import time — on this machine ``pymilvus`` calls
``load_dotenv()``, which picks up whatever ``.env`` sits above site-packages —
and is WARN, since that is not a lasagna release defect.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..contract import PhaseResult, Severity, timed

#: Derived from this file's location, not hard-coded. A hard-coded absolute
#: path means a checkout anywhere else silently misses the source tree, and a
#: phase that skips is a phase that stops blocking releases — the gate would
#: report GO with the core journey merely listed as "unvalidated".
LASAGNA_ROOT = Path(__file__).resolve().parents[2]
LASAGNA_SRC = LASAGNA_ROOT / "src"

#: Generous. The driver imports the config layer and the retrieval package
#: (~5 s cold on an idle laptop); a busy machine or a cold FS cache is slower.
#: A timeout here is reported as a failure, not a skip — the journey not
#: completing in five minutes is a real problem.
DRIVER_TIMEOUT_S = 300

#: Soft budget for the whole journey. Breaching it is a WARN, not a blocker: a
#: loaded laptop can miss it without the package being broken. It exists so an
#: import-time regression (a new top-level model load, say) shows up as a
#: trend rather than being invisible until it hits the hard timeout.
JOURNEY_BUDGET_S = 60.0

_BEGIN = "===P4_JSON_BEGIN==="
_END = "===P4_JSON_END==="

#: The names the journey registers under. Deliberately unlikely to collide with
#: anything the package ships, so a name clash cannot make registration look like it
#: passed by finding someone else's component.
COMPONENT_NAME = "e2e_wordcount_reranker"
#: A second, deliberately different reranker. Step 6 seats this one first so the
#: pipeline's instance cache is populated, then replaces it — which is the only
#: way to tell a real swap from a name-field update.
CONTROL_NAME = "e2e_control_reranker"
PAPER = "arxiv:0000.00000"
CLAIMS = {"ndcg@10": 0.4242, "note": "not a real result"}

#: Env vars that would let a component reach a real LLM. Removed from the
#: driver's environment so the hermeticity claim in the docstring is true
#: rather than aspirational — the gate inherits the developer's shell, which on
#: may hold live keys.
_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "WATSONX_API_KEY",
    "WATSONX_APIKEY",
    "WATSONX_PROJECT_ID",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


# ---------------------------------------------------------------------------
# The journey, as a standalone script.
#
# It runs in a subprocess so that: (a) PYTHONPATH can point at the source tree
# without polluting the gate's own interpreter, (b) the registry singleton is
# fresh on every run — re-running this phase in one process would hit the
# "Overwriting existing" path and test something different from what a user
# does, and (c) a hard crash or a hang in the package cannot take the gate down
# with it.
#
# Every step records {"ok", "detail"}, or {"ok": None, "skipped": reason} when
# the package's state makes the step meaningless to run. Steps that were never
# reached stay absent from the payload and are reported by run() as SKIPPED,
# never as passed. The payload is emitted from a finally block so a crash
# mid-journey still yields the metrics measured before it.
#
# Constants are injected as a prelude rather than through str.format(), so the
# body below is ordinary Python: no doubled braces, and no chance of a stray
# "{" turning into a build-time KeyError.
# ---------------------------------------------------------------------------
DRIVER_BODY = r'''
import copy, io, json, re, sys, traceback, warnings

steps = {}
metrics = {}

OWN = {NAME, CONTROL}


class _Skip(Exception):
    """Raised by a step whose premise does not hold, so it cannot mean anything.

    Reported as SKIPPED, never as passed: "there was nothing to check" and
    "we checked and it was fine" are different statements.
    """


def step(key, fn):
    """Run one journey step. Records ok/skipped/detail; never raises."""
    if key in steps:
        return steps[key].get("ok") is True
    try:
        detail = fn()
    except _Skip as exc:
        steps[key] = {"ok": None, "skipped": str(exc)}
        return False
    except Exception:
        steps[key] = {"ok": False, "detail": traceback.format_exc(limit=6).strip()}
        return False
    steps[key] = {"ok": True, "detail": detail or ""}
    return True


def emit():
    print(BEGIN)
    print(json.dumps({"steps": steps, "metrics": metrics}, default=str))
    print(END)


def mentions(text, name):
    """True when `name` appears in `text` as a whole identifier.

    Substring matching would let 'mmr' look like a named alternative merely
    because 'mmr_v2' was in the sentence, which is how a warning that names
    nothing useful passes a check that meant to require one.
    """
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def credentials_in_env():
    import os
    return sorted(
        k for k in os.environ
        if k.endswith(("_API_KEY", "_APIKEY", "_TOKEN", "_SECRET"))
    )


try:
    creds_before_import = credentials_in_env()

    from loguru import logger
    from lasagna.core.registry import registry
    from lasagna.core.base_components import BaseReranker, ScoredDocument
    from lasagna.core.pipeline import KnowledgePipeline

    # -- 1. the sandbox the gate itself controls (CRITICAL) --------------
    def do_sandbox():
        # Entirely within this phase's control, so a failure here is a defect
        # in the gate: HOME must be the throwaway one, or the journey is
        # reading and writing the developer's real ~/.rag_knowledge, and no
        # credential may be handed down from the parent environment.
        import os as _os
        home = _os.path.expanduser("~")
        if home != EXPECTED_HOME:
            raise AssertionError(
                "HOME resolves to %r, not the throwaway %r — the journey may be "
                "reading or writing the developer's real ~/.rag_knowledge"
                % (home, EXPECTED_HOME)
            )
        if creds_before_import:
            raise AssertionError(
                "credential env var(s) %s were inherited from the parent "
                "environment; add them to _CREDENTIAL_VARS in this phase"
                % (creds_before_import,)
            )
        return "HOME=%s, cwd=%s, no inherited credentials" % (home, _os.getcwd())

    step("sandbox", do_sandbox)

    # -- 2. nothing in the import graph injected credentials (WARN) ------
    def do_no_credentials():
        # WARN, not CRITICAL. A dependency that calls load_dotenv() at import
        # scoops up whatever .env sits above site-packages — which on a
        # developer machine can be an unrelated project's live keys. That is
        # not a lasagna release defect and must not produce a false NO-GO, but
        # it does void this phase's hermeticity claim: with keys in the
        # environment, a component that quietly called a provider would pass
        # here and fail in CI. So it is on the record, every run.
        leaked = sorted(set(credentials_in_env()) - set(creds_before_import))
        if leaked:
            raise AssertionError(
                "importing lasagna injected credential env var(s) %s into the "
                "environment (a dependency calling load_dotenv() at import time "
                "picks up any .env above site-packages). The journey is not "
                "hermetic: a component reaching a real provider would succeed "
                "here instead of failing loudly." % (leaked,)
            )
        return "no credential env vars after importing lasagna"

    step("no_credentials", do_no_credentials)

    # -- 3. register a custom reranker -----------------------------------
    holder = {}

    def do_register():
        @registry.register("reranker", NAME, paper=PAPER, claims=CLAIMS)
        class WordCountReranker(BaseReranker):
            """Ranks by word count. Deliberately unlike retrieval-score order."""

            def rerank(self, query, documents, top_k=5, **kwargs):
                # Deep-copy first: a stage must not mutate what it was handed.
                out = [copy.deepcopy(d) for d in documents]
                for d in out:
                    d.score = float(len(d.content.split()))
                out.sort(key=lambda d: d.score, reverse=True)
                return out[:top_k]

        @registry.register("reranker", CONTROL)
        class ControlReranker(BaseReranker):
            """Returns documents untouched, in input order.

            Exists so step 6 can tell "the pipeline swapped the component" from
            "the pipeline updated a name field": this is what comes back when
            the swap does not take effect.
            """

            def rerank(self, query, documents, top_k=5, **kwargs):
                return [copy.deepcopy(d) for d in documents][:top_k]

        holder["cls"] = WordCountReranker
        holder["control"] = ControlReranker
        missing = [n for n in (NAME, CONTROL) if not registry.has("reranker", n)]
        if missing:
            raise AssertionError(
                "registry.has() False immediately after register for %s" % (missing,)
            )
        return "registered %s as %r (and %s as %r)" % (
            WordCountReranker.__name__, NAME, ControlReranker.__name__, CONTROL
        )

    registered = step("register", do_register)

    # Counts are measured here, before any assertion that could fail, so the
    # report still carries them if a later step blows up. Both blocks use
    # .get("implements_contract") deliberately: these are best-effort metrics,
    # and the strict as-documented form is what step 5 asserts.
    try:
        metrics["counts"] = {k: len(v) for k, v in registry.list_all().items()}
    except Exception as exc:
        metrics["counts_error"] = str(exc)

    try:
        _entries = registry.list_rerankers()
        _compliant = [e["name"] for e in _entries if e.get("implements_contract") is True]
        metrics["contract_compliant_rerankers"] = len(_compliant)
        metrics["baselines_excluding_own"] = len([n for n in _compliant if n not in OWN])
    except Exception as exc:
        metrics["baseline_count_error"] = str(exc)

    # -- 4. it is discoverable, with provenance intact -------------------
    def do_discover():
        entries = registry.list_rerankers()
        mine = [e for e in entries if e["name"] == NAME]
        if len(mine) != 1:
            raise AssertionError(
                "expected exactly 1 listing for %r, got %d (names: %s)"
                % (NAME, len(mine), [e["name"] for e in entries])
            )
        e = mine[0]
        if e.get("implements_contract") is not True:
            raise AssertionError(
                "implements_contract is %r, expected True" % (e.get("implements_contract"),)
            )
        if e.get("paper") != PAPER:
            raise AssertionError("paper round-trip: %r != %r" % (e.get("paper"), PAPER))
        if e.get("claims") != CLAIMS:
            raise AssertionError("claims round-trip: %r != %r" % (e.get("claims"), CLAIMS))
        if e.get("class") != holder["cls"].__name__:
            raise AssertionError("class name: %r" % (e.get("class"),))
        return "listed with implements_contract=True, paper=%s, claims=%s" % (
            e["paper"], e["claims"]
        )

    if registered:
        step("discover", do_discover)

    # -- 5. comparable baselines exist -----------------------------------
    def do_baselines():
        # The exact expression the docs tell a researcher to use. Run as-written
        # so a KeyError here is reported rather than papered over with .get().
        compliant = [
            r["name"] for r in registry.list_rerankers() if r["implements_contract"]
        ]
        others = sorted(n for n in compliant if n not in OWN)
        if not others:
            raise AssertionError(
                "no contract-compliant reranker other than the ones this phase "
                "registered — nothing to compare against (all rerankers: %s)"
                % ([r["name"] for r in registry.list_rerankers()],)
            )
        return "%d contract-compliant baselines besides our own: %s" % (
            len(others), others
        )

    if registered:
        step("baselines", do_baselines)

    # -- 6. the pipeline actually swaps ----------------------------------
    def do_replace():
        # Seat the control component and resolve it, so the pipeline's instance
        # cache is populated. Replacing into a fresh pipeline proves nothing:
        # there is no stale instance for a broken replace() to keep serving.
        p = KnowledgePipeline(reranker=CONTROL)
        seated = p.get_component("reranker")
        if not isinstance(seated, holder["control"]):
            raise AssertionError(
                "pipeline did not resolve its constructor reranker: "
                "get_component() gave %s, expected %s"
                % (type(seated).__name__, holder["control"].__name__)
            )

        p.replace("reranker", NAME)

        got_name = p._component_names.get("reranker")
        if got_name != NAME:
            raise AssertionError("pipeline reranker name is %r, expected %r" % (got_name, NAME))

        got = p.get_component("reranker")
        if not isinstance(got, holder["cls"]):
            raise AssertionError(
                "replace() set the name to %r but the pipeline still hands out "
                "%s — the swap never reaches execution (stale instance cache?). "
                "A user calling replace() would silently keep running the old "
                "component." % (got_name, type(got).__name__)
            )
        return "replace() -> name %r and get_component() -> %s (was %s)" % (
            got_name, type(got).__name__, type(seated).__name__
        )

    if registered:
        step("replace", do_replace)

    # -- 7. selecting a non-conforming built-in warns, and names an
    #       alternative that is conforming ---------------------------------
    def do_warning():
        entries = {e["name"]: e for e in registry.list_rerankers()}
        non_conforming = sorted(
            n for n, e in entries.items()
            if n not in OWN and e.get("implements_contract") is False
        )
        if not non_conforming:
            # Nothing for the warning to fire on. That is a legitimate state
            # (every built-in conforms), and it is not evidence the warning
            # works — so it is a skip, not a pass.
            raise _Skip(
                "no non-conforming built-in reranker is registered, so the "
                "interchangeability warning has nothing to fire on "
                "(rerankers: %s)" % (sorted(entries),)
            )
        # Prefer the one the README documents, so the repro matches the docs.
        chosen = "cross_encoder" if "cross_encoder" in non_conforming else non_conforming[0]
        metrics["warning_component"] = chosen

        # A plain script sees Python warnings; the package's own log lines are
        # off unless the host enables them. Either channel counts.
        buf = io.StringIO()
        sink = logger.add(buf, level="WARNING", format="{message}")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                KnowledgePipeline().replace("reranker", chosen)
        finally:
            logger.remove(sink)
        text = buf.getvalue() + "".join(str(w.message) for w in caught)

        if not text.strip():
            raise AssertionError(
                "selecting non-conforming built-in %r emitted no WARNING at all" % (chosen,)
            )
        if not mentions(text, chosen):
            raise AssertionError(
                "warning does not name the chosen component %r: %r" % (chosen, text)
            )
        # The alternative offered must be a real one: package-provided (not a
        # component this phase just registered), contract-compliant, and not
        # the component the user already chose. Whole-identifier matching, so
        # naming 'cross_encoder' cannot be read as naming 'cross_encoder_x'.
        alternatives = sorted(
            n for n, e in entries.items()
            if n not in OWN and n != chosen and e.get("implements_contract") is True
        )
        named = [n for n in alternatives if mentions(text, n)]
        if not named:
            raise AssertionError(
                "warning for %r names no conforming alternative. "
                "candidates=%s warning=%r" % (chosen, alternatives, text)
            )
        return "%r warned, naming conforming alternative(s) %s" % (chosen, sorted(named))

    if registered:
        step("warning", do_warning)

    # -- 8. it actually runs, through the pipeline -----------------------
    docs_after = {}

    def do_run():
        # Word count runs *opposite* to retrieval score here, so a framework
        # that quietly returned the input (or re-sorted by score) produces the
        # reverse of the expected order and fails this check.
        docs = [
            ScoredDocument(content="alpha", score=0.9, document_id="d1",
                           chunk_id="c1", metadata={"tag": "one"}),
            ScoredDocument(content="beta beta", score=0.5, document_id="d2",
                           chunk_id="c2", metadata={"tag": "two"}),
            ScoredDocument(content="gamma gamma gamma", score=0.1, document_id="d3",
                           chunk_id="c3", metadata={"tag": "three"}),
        ]
        before = copy.deepcopy(docs)

        # Resolve through the *pipeline*, not the registry: that is the object
        # a user's query would actually be reranked by. Resolving through
        # registry.get() here would pass even if the pipeline ignored the swap.
        p = KnowledgePipeline().replace("reranker", NAME)
        inst = p.get_component("reranker")
        if not isinstance(inst, BaseReranker):
            raise AssertionError("pipeline.get_component() returned %r" % (type(inst),))
        direct = registry.get("reranker", NAME)
        if type(direct) is not type(inst):
            raise AssertionError(
                "registry.get() -> %s but pipeline.get_component() -> %s; the two "
                "resolution paths disagree" % (type(direct).__name__, type(inst).__name__)
            )

        out = inst.rerank("gamma", docs, top_k=2)

        docs_after["before"] = before
        docs_after["docs"] = docs
        docs_after["out"] = out

        if [d.chunk_id for d in out] != ["c3", "c2"]:
            raise AssertionError(
                "order %s, expected ['c3','c2'] (custom word-count logic)"
                % ([d.chunk_id for d in out],)
            )
        if [d.score for d in out] != [3.0, 2.0]:
            raise AssertionError("scores %s, expected [3.0, 2.0]" % ([d.score for d in out],))
        score_desc = [d.chunk_id for d in sorted(docs, key=lambda x: -x.score)][:2]
        if [d.chunk_id for d in out] == score_desc:
            raise AssertionError(
                "output matches plain score-descending order — cannot tell the "
                "custom logic ran"
            )
        return (
            "via pipeline.get_component() -> %s: top_k=2 -> %s with scores %s "
            "(input score order was %s)"
            % (
                type(inst).__name__,
                [d.chunk_id for d in out],
                [d.score for d in out],
                ["c1", "c2", "c3"],
            )
        )

    ran = step("run", do_run) if registered else False

    # -- 9. stage isolation: the input was not mutated -------------------
    def do_isolation():
        before = docs_after["before"]
        after = docs_after["docs"]
        out = docs_after["out"]
        if len(before) != len(after):
            raise AssertionError("input list length changed: %d -> %d" % (len(before), len(after)))
        diffs = []
        for b, a in zip(before, after):
            for field in ("content", "score", "document_id", "chunk_id", "metadata"):
                if getattr(b, field) != getattr(a, field):
                    diffs.append("%s.%s: %r -> %r" % (b.chunk_id, field,
                                                      getattr(b, field), getattr(a, field)))
        if diffs:
            raise AssertionError("reranker mutated its input: " + "; ".join(diffs))
        # Field equality is not enough: returning the caller's own objects
        # means a later stage mutating the result silently rewrites this
        # stage's input, which is the leak the isolation guarantee exists to
        # prevent. The returned documents must be distinct objects.
        shared = [o.chunk_id for o in out for d in after if o is d]
        if shared:
            raise AssertionError(
                "reranker returned the caller's own document objects (%s); a "
                "downstream mutation would reach back into the input" % (shared,)
            )
        return (
            "3 input documents byte-identical after rerank "
            "(content/score/ids/metadata) and outputs are distinct objects"
        )

    if ran:
        step("isolation", do_isolation)

except Exception:
    steps["_harness"] = {"ok": False, "detail": traceback.format_exc(limit=10).strip()}
finally:
    emit()
'''


def _build_driver(expected_home: str) -> str:
    """Prepend the injected constants to the driver body."""
    prelude = "\n".join(
        [
            f"NAME = {COMPONENT_NAME!r}",
            f"CONTROL = {CONTROL_NAME!r}",
            f"PAPER = {PAPER!r}",
            f"CLAIMS = {CLAIMS!r}",
            f"BEGIN = {_BEGIN!r}",
            f"END = {_END!r}",
            f"EXPECTED_HOME = {expected_home!r}",
            "",
        ]
    )
    return prelude + DRIVER_BODY


def _repro(body: str) -> str:
    """A copy-pasteable shell command that reproduces one step on its own.

    Carries the same environment the gate used — throwaway HOME *and* cwd,
    offline, CPU. A repro run under different conditions is not a repro: run
    this from the repo root instead and the config layer picks up the repo's
    .env, which is a materially different run.
    """
    return (
        f'cd "$(mktemp -d)" && HOME=$(mktemp -d) PYTHONPATH={LASAGNA_SRC} '
        f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAG_USE_GPU=false "
        f"{sys.executable} - <<'PY'\n{body.strip()}\nPY"
    )


_R_HEAD = (
    "import copy\n"
    "from lasagna.core.registry import registry\n"
    "from lasagna.core.base_components import BaseReranker, ScoredDocument\n"
    "from lasagna.core.pipeline import KnowledgePipeline\n"
    "@registry.register('reranker', 'demo', paper='arxiv:0000.00000',\n"
    "                   claims={'ndcg@10': 0.4242})\n"
    "class R(BaseReranker):\n"
    "    def rerank(self, query, documents, top_k=5, **kw):\n"
    "        out = [copy.deepcopy(d) for d in documents]\n"
    "        for d in out: d.score = float(len(d.content.split()))\n"
    "        out.sort(key=lambda d: d.score, reverse=True)\n"
    "        return out[:top_k]\n"
    "@registry.register('reranker', 'demo_control')\n"
    "class C(BaseReranker):\n"
    "    def rerank(self, query, documents, top_k=5, **kw):\n"
    "        return [copy.deepcopy(d) for d in documents][:top_k]\n"
)

_R_DOCS = (
    "docs = [ScoredDocument(content='alpha', score=0.9, chunk_id='c1'),\n"
    "        ScoredDocument(content='beta beta', score=0.5, chunk_id='c2'),\n"
    "        ScoredDocument(content='gamma gamma gamma', score=0.1, chunk_id='c3')]\n"
)


def _repros(warning_component: str) -> Dict[str, str]:
    """Per-step repro commands.

    Built at report time because the warning step picks its component from what
    is actually registered; a repro naming a different component than the run
    used would not reproduce.
    """
    return {
        "sandbox": _repro(
            "import os\n"
            "print('HOME         :', os.path.expanduser('~'))\n"
            "print('cwd          :', os.getcwd())\n"
            "print('inherited    :', sorted(k for k in os.environ if"
            " k.endswith(('_API_KEY','_APIKEY','_TOKEN','_SECRET'))), '-- must be []')"
        ),
        "no_credentials": _repro(
            "import os, traceback\n"
            "def creds(): return sorted(k for k in os.environ if"
            " k.endswith(('_API_KEY','_APIKEY','_TOKEN','_SECRET')))\n"
            "_set = os.environ.__class__.__setitem__\n"
            "def spy(self, k, v):\n"
            "    if k.endswith(('_API_KEY','_APIKEY','_TOKEN','_SECRET')):\n"
            "        print('INJECTED', k, 'by:')\n"
            "        for ln in traceback.format_stack()[-5:-1]:\n"
            "            print('   ', ln.strip().splitlines()[0])\n"
            "    return _set(self, k, v)\n"
            "os.environ.__class__.__setitem__ = spy\n"
            "print('before:', creds())\n"
            "import lasagna.core.registry\n"
            "print('after :', creds(), '-- must be []')"
        ),
        "register": _repro(_R_HEAD + "print(registry.has('reranker', 'demo'))"),
        "discover": _repro(
            _R_HEAD + "print([e for e in registry.list_rerankers() if e['name'] == 'demo'])"
        ),
        "baselines": _repro(
            _R_HEAD + "print([r['name'] for r in registry.list_rerankers()"
            " if r['implements_contract']])"
        ),
        "replace": _repro(
            _R_HEAD + "p = KnowledgePipeline(reranker='demo_control')\n"
            "print('seated  :', type(p.get_component('reranker')).__name__)\n"
            "p.replace('reranker', 'demo')\n"
            "print('name    :', p._component_names['reranker'])\n"
            "print('hands out:', type(p.get_component('reranker')).__name__,\n"
            "      '-- expected R; C means replace() did not reach execution')"
        ),
        "warning": _repro(
            "import warnings\n"
            "from lasagna.core.pipeline import KnowledgePipeline\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            f"    KnowledgePipeline().replace('reranker', {warning_component!r})\n"
            "print([str(w.message) for w in caught])"
        ),
        "run": _repro(
            _R_HEAD + _R_DOCS + "inst = KnowledgePipeline().replace('reranker', 'demo')"
            ".get_component('reranker')\n"
            "out = inst.rerank('gamma', docs, top_k=2)\n"
            "print([(d.chunk_id, d.score) for d in out], '-- expected c3=3.0, c2=2.0')"
        ),
        "isolation": _repro(
            _R_HEAD + _R_DOCS + "before = copy.deepcopy(docs)\n"
            "inst = KnowledgePipeline().replace('reranker', 'demo')"
            ".get_component('reranker')\n"
            "out = inst.rerank('gamma', docs, top_k=2)\n"
            "print('unmutated:', [(b.content, b.score) for b in before] =="
            " [(d.content, d.score) for d in docs])\n"
            "print('distinct :', not any(o is d for o in out for d in docs))"
        ),
    }


#: Human-readable check names, in journey order, with the severity a failure
#: carries. Everything the journey itself asserts is CRITICAL — if a researcher
#: cannot register, find, swap and run a component, the package has no purpose.
#: ``no_credentials`` is WARN because its cause lives outside the package (a
#: dependency importing someone else's .env) and a false NO-GO there would
#: teach people to ignore the gate.
STEP_TITLES = [
    (
        "sandbox",
        "journey runs under a throwaway HOME with no inherited credentials",
        Severity.CRITICAL,
    ),
    (
        "no_credentials",
        "importing lasagna injects no credentials into the environment",
        Severity.WARN,
    ),
    ("register", "custom reranker registers via @registry.register", Severity.CRITICAL),
    (
        "discover",
        "appears in list_rerankers() with contract + provenance intact",
        Severity.CRITICAL,
    ),
    ("baselines", "contract-compliant baselines exist to compare against", Severity.CRITICAL),
    ("replace", "replace() swaps the component the pipeline actually hands out", Severity.CRITICAL),
    (
        "warning",
        "non-conforming built-in warns and names a conforming alternative",
        Severity.CRITICAL,
    ),
    ("run", "custom reranker runs via the pipeline and its ordering wins", Severity.CRITICAL),
    ("isolation", "reranker does not mutate the documents it was given", Severity.CRITICAL),
]
_STEP_KEYS = {k for k, _, _ in STEP_TITLES}


def _parse_payload(stdout: str) -> Optional[Dict[str, Any]]:
    if _BEGIN not in stdout or _END not in stdout:
        return None
    blob = stdout.split(_BEGIN, 1)[1].split(_END, 1)[0].strip()
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _tail(text: str, n: int = 900) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


def run() -> PhaseResult:
    p = PhaseResult(key="p4", title="Core journey")

    if not LASAGNA_SRC.is_dir():
        # Not a skip. This phase lives inside the repo it gates, so a missing
        # source tree means a broken checkout, not an unavailable capability —
        # and a skip here would let the gate report GO with the core journey
        # never run.
        p.check(
            "lasagna source tree present",
            False,
            severity=Severity.CRITICAL,
            detail=f"expected the package source at {LASAGNA_SRC}, which does not exist",
            repro=f"ls {LASAGNA_SRC}",
        )
        for _, title, severity in STEP_TITLES:
            p.skip(
                title,
                "not reached: no source tree to run the journey against",
                severity=severity,
            )
        return p

    workdir = tempfile.mkdtemp(prefix="lasagna-p4-")
    # A throwaway HOME keeps the run hermetic: the config layer creates
    # ~/.rag_knowledge on import, and the gate must not read or write the
    # developer's real database.
    fake_home = os.path.join(workdir, "home")
    os.makedirs(fake_home, exist_ok=True)
    driver_path = os.path.join(workdir, "journey.py")

    payload: Optional[Dict[str, Any]] = None
    proc: Optional[subprocess.CompletedProcess] = None
    harness_detail = ""
    elapsed = 0.0

    try:
        Path(driver_path).write_text(_build_driver(fake_home))

        env = dict(os.environ)
        for var in _CREDENTIAL_VARS:
            env.pop(var, None)
        # Anything else that looks like a secret, so a key this list has never
        # heard of does not quietly make the journey non-hermetic.
        for var in [k for k in env if k.endswith(("_API_KEY", "_APIKEY", "_TOKEN", "_SECRET"))]:
            env.pop(var, None)
        env.update(
            {
                "PYTHONPATH": str(LASAGNA_SRC),
                "HOME": fake_home,
                # Fail loudly instead of downloading if anything reaches for a model.
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                # MPS compiler crashes have been seen on this machine; CPU only.
                "RAG_USE_GPU": "false",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )

        with timed() as t:
            try:
                proc = subprocess.run(
                    [sys.executable, driver_path],
                    # Throwaway cwd, not the repo root: the config layer
                    # discovers a .env relative to the working directory, and
                    # running from the checkout loaded the developer's
                    # gitignored keys into the journey's environment.
                    cwd=workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=DRIVER_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                harness_detail = f"journey driver exceeded {DRIVER_TIMEOUT_S}s and was killed"
            except Exception as exc:  # driver could not be launched at all
                harness_detail = f"could not launch journey driver: {exc!r}"
        elapsed = t.elapsed

        if proc is not None:
            payload = _parse_payload(proc.stdout)
            if payload is None:
                harness_detail = (
                    f"driver exited {proc.returncode} without a result payload. "
                    f"stderr: {_tail(proc.stderr)}"
                )
    except Exception as exc:  # run() must never raise
        harness_detail = f"phase harness error: {exc!r}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # -- metrics first, assertions after ------------------------------------
    p.measure(
        "journey_wall_time",
        round(elapsed, 2),
        "s",
        note="subprocess incl. package import; not a latency SLO",
    )

    steps: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    if payload:
        raw_steps = payload.get("steps")
        steps = raw_steps if isinstance(raw_steps, dict) else {}
        raw_metrics = payload.get("metrics")
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        if raw_steps is not None and not isinstance(raw_steps, dict):
            harness_detail = harness_detail or f"driver payload 'steps' was {type(raw_steps)}"

        counts = metrics.get("counts")
        if isinstance(counts, dict):
            for comp_type, count in sorted(counts.items()):
                try:
                    value = float(count)
                except (TypeError, ValueError):
                    continue
                p.measure(
                    f"registered_{comp_type}s",
                    value,
                    "components",
                    lower_is_better=False,
                    note="includes the 2 this phase registered"
                    if comp_type == "reranker"
                    else "built-ins only",
                )

        for src_key, out_name, note in (
            (
                "contract_compliant_rerankers",
                "contract_compliant_rerankers",
                "rerankers implementing BaseReranker, incl. this phase's",
            ),
            (
                "baselines_excluding_own",
                "reranker_baselines_available",
                "contract-compliant rerankers a researcher could compare against",
            ),
        ):
            if src_key in metrics:
                try:
                    p.measure(
                        out_name,
                        float(metrics[src_key]),
                        "components",
                        lower_is_better=False,
                        note=note,
                    )
                except (TypeError, ValueError):
                    pass

        for err_key in ("counts_error", "baseline_count_error"):
            if err_key in metrics:
                harness_detail = harness_detail or f"{err_key}: {metrics[err_key]}"

    warning_component = str(metrics.get("warning_component") or "cross_encoder")
    repros = _repros(warning_component)

    # -- checks --------------------------------------------------------------
    # Always reported, pass or fail: it asserts the subprocess ran the whole
    # journey and emitted a well-formed payload. A driver that never ran must
    # be a visible failure, not seven silent skips with no cause attached.
    harness_ok = bool(steps) and "_harness" not in steps
    p.check(
        "journey driver executed",
        harness_ok,
        severity=Severity.CRITICAL,
        detail=_tail(
            (
                f"{len(steps)} step(s) recorded"
                if harness_ok
                else (
                    harness_detail
                    or (steps.get("_harness", {}).get("detail", "") if steps else "")
                    or "journey driver produced no steps"
                )
            ),
            1200,
        ),
        repro=(
            f"cd {LASAGNA_ROOT} && HOME=$(mktemp -d) PYTHONPATH=src {sys.executable} -c "
            f"'import lasagna.core.registry, lasagna.core.pipeline'"
        ),
    )

    any_failed = any(isinstance(r, dict) and r.get("ok") is False for r in steps.values())

    for key, title, severity in STEP_TITLES:
        rec = steps.get(key)
        if not isinstance(rec, dict):
            # Not run. Not a pass — say why, so a reader is not left hunting for
            # a journey step failure that never happened.
            if not harness_ok:
                p.skip(
                    title,
                    "not reached: journey driver did not complete — see the "
                    "'journey driver executed' check",
                    severity=severity,
                )
            elif any_failed:
                p.skip(
                    title,
                    "not reached: an earlier journey step failed, making this unreachable",
                    severity=severity,
                )
            else:
                # Nothing failed and nothing skipped, yet the step is absent.
                # That means the driver never recorded it — a defect in this
                # phase, and silently dropping it would turn a missing check
                # into a passing run.
                p.check(
                    title,
                    False,
                    severity=severity,
                    detail=(
                        "the driver completed with no failed step but never recorded "
                        f"{key!r}; the phase is not running the check it claims to. "
                        f"Recorded steps: {sorted(steps)}"
                    ),
                    repro=repros.get(key, ""),
                )
            continue
        if rec.get("skipped"):
            p.skip(title, str(rec["skipped"]), severity=severity)
            continue
        p.check(
            title,
            rec.get("ok") is True,
            severity=severity,
            detail=_tail(str(rec.get("detail", "")), 1200),
            repro=repros.get(key, ""),
        )

    # A step recorded under a key this phase does not report would vanish from
    # the report — including a failing one. Surface it rather than drop it.
    unknown = sorted(k for k in steps if k not in _STEP_KEYS and k != "_harness")
    if unknown:
        p.check(
            "every step the driver recorded is reported",
            False,
            severity=Severity.CRITICAL,
            detail=(
                f"driver recorded step(s) {unknown} that STEP_TITLES does not list, "
                "so their result never reaches the report"
            ),
            repro=f"grep -n 'step(\"' {Path(__file__).resolve()}",
        )

    # Performance, deliberately WARN: a slow laptop should not block a release,
    # but an import-time regression should still be on the record.
    if harness_ok:
        p.check(
            f"journey completes within {JOURNEY_BUDGET_S:.0f}s",
            elapsed <= JOURNEY_BUDGET_S,
            severity=Severity.WARN,
            detail=f"{elapsed:.2f}s (budget {JOURNEY_BUDGET_S:.0f}s, hard timeout {DRIVER_TIMEOUT_S}s)",
            repro=f"cd {LASAGNA_ROOT} && time {sys.executable} -m e2e.run --phase p4",
        )
    else:
        p.skip(
            f"journey completes within {JOURNEY_BUDGET_S:.0f}s",
            "not measured: the journey did not complete, so its wall time is not a "
            "performance number",
            severity=Severity.WARN,
        )

    p.duration_s = elapsed
    return p


if __name__ == "__main__":  # pragma: no cover - manual invocation
    r = run()
    print(json.dumps(r.to_dict(), indent=2, default=str))
