"""Phase 2 — Cold start: a user's first ten minutes, from nothing.

What this phase claims to prove
-------------------------------
That someone who has never run Lasagna before can, without reading the source:

1. ``import lasagna`` quickly, without dragging in torch/pymilvus (the package
   advertises lazy imports; a regression here makes every CLI invocation and
   every host application's startup pay a multi-second tax).
2. Discover the CLI — ``--help`` at the top level, on every registered command,
   and on every command the README quickstart tells them to type — without a
   configured backend, an API key, or a downloaded model.
3. Ingest a document and get *that document* back from a search, end to end, on
   a cold machine: the bytes land where they were configured to land, the search
   finds the answer that is in the corpus, and the chunk count the user is shown
   agrees with what search can find. That includes downloading the default
   embedding model and reranker, several GB. It is slow, and that slowness *is*
   the user experience, so it is measured rather than engineered away.

What this phase does NOT prove
------------------------------
It runs against the repository source via ``PYTHONPATH=src``, not against an
installed wheel. This is a deliberate fidelity trade: building a wheel and
resolving torch + pymilvus + sentence-transformers into a fresh venv costs many
minutes and several GB before the first useful measurement, and phase p1 already
covers exactly that. The consequences, stated plainly:

* The ``lasagna`` console script from ``[project.scripts]`` is never executed.
  Commands are driven through ``python -m lasagna``, which shares the Typer app
  but not the entry-point wiring. That check is *skipped*, not passed.
* Dependency metadata is not exercised. A missing or mis-pinned dependency that
  happens to be present in the driving interpreter would pass here and fail for
  a real user. p1 is the phase that catches that.
* Packaged data files (``settings.lasagna.toml``, templates, deploy manifests)
  are read from the source tree, where they always exist, rather than from a
  wheel, where they exist only if ``package-data`` is right.
* ``__pycache__`` in the source tree is warm, so the measured import time is a
  best case. A genuinely cold install pays bytecode compilation on top.

Isolation
---------
Every child process runs with a scrubbed environment: ``HOME``, ``HF_HOME``,
``RAG_DB_DIR`` and ``RAG_UPLOAD_DIR`` all point inside a throwaway sandbox, and
no LLM API keys are inherited. That is both hygiene (the real
``~/.rag_knowledge`` and the real model cache are left alone — asserted at the
end of the phase, not assumed) and fidelity: a first-time user does not have an
``OPENAI_API_KEY`` sitting in their shell, and if the quickstart secretly needs
one, this phase should be the thing that finds out.

``TMPDIR`` is the exception and stays on the system default; see ``_child_env``
for why. Temporary files therefore outlive the sandbox cleanup.

How this phase is built so it can actually fail
-----------------------------------------------
A gate is only worth running if there is a state of the world in which it says
NO-GO. Four properties are load-bearing, and each one exists because its absence
was a way for a broken product to sail through:

* **Nothing is asserted about code this phase did not load.** The import probe
  reports ``lasagna.__file__`` and the phase asserts it resolves under ``src/``.
  An editable install of a *different* checkout in the driving interpreter
  installs a meta-path finder that outranks ``PYTHONPATH``, so without that
  assertion the whole phase can pass while testing code that is not being
  released.
* **The quickstart commands are probed by name, not only by enumeration.** If
  ``@app.command()`` registration breaks, ``typer.main.get_command`` returns an
  empty mapping, and a loop over "every registered command" asserts nothing and
  passes. The four commands the README tells a user to type are therefore
  hardcoded and probed regardless of what enumeration returns.
* **Running out of patience is a failure, not a shrug.** If the journey has not
  finished when the ceiling hits, the check that says so is CRITICAL. Whether
  the process was wedged or still moving changes the *diagnosis* (it is in the
  detail) but not the verdict: a first run that has not produced a search result
  after ``E2E_P2_TIMEOUT_S`` is a release-blocking first-run experience, and
  "it was probably just slow" is exactly the reasoning that lets a hang ship.
  A genuinely slow link is accommodated by raising the knob deliberately, which
  is recorded, rather than by the gate quietly passing itself.
* **Correctness is CRITICAL even when it is embarrassing.** Per ``e2e/README.md``:
  correctness is never WARN. A chunk count that contradicts what search returns,
  or a search that cannot find the only document in the corpus, blocks — the
  gate's job is to say so, not to grade on a curve.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from ..contract import PhaseResult, Severity, timed

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

# Heavy modules the lazy-import design promises not to load at `import lasagna`.
# Each one costs seconds of process startup, so this is a user-visible contract,
# not an aesthetic preference.
FORBIDDEN_AT_IMPORT = ("torch", "transformers", "sentence_transformers", "pymilvus")

# The commands the README quickstart tells a first-time user to type, verbatim
# from its code block. They are hardcoded rather than discovered because
# discovery is the thing most likely to break silently: a Typer upgrade or a
# botched registration yields an empty command list, and a phase that only
# iterates that list asserts nothing while reporting success. Probing these by
# name means a CLI that registers nothing fails loudly.
QUICKSTART_COMMANDS = ("init", "upload", "search", "serve")

# Fixed so the repro commands can name the same session the phase used.
SESSION_ID = "e2e-p2-cold-start"

# Thresholds. All WARN: a loaded laptop or a slow link must not block a release.
IMPORT_BUDGET_S = 1.5
# A cold run measured 1340 s on an M-series laptop on a home connection, almost
# all of it downloading the default models. The budget is set below that on
# purpose: the WARN is the honest signal that a first run takes twenty minutes,
# not a threshold anyone has met.
FIRST_SEARCH_BUDGET_S = 600.0
# Hard ceiling. Beyond this we stop waiting and fail — critically — with the
# elapsed time attached, rather than hanging the gate. Set to roughly twice the
# measured cold run, so reaching it means the first-run experience is at least
# 45 minutes of nothing, which no release should ship.
#
# Raise it with E2E_P2_TIMEOUT_S when the link is known to be slow. That is a
# deliberate, recorded act by whoever runs the gate; the alternative — the phase
# excusing itself — is how a process that never finishes gets reported as fine.
FIRST_SEARCH_TIMEOUT_S = 2700.0
# If nothing observable about the child has moved for this long when the ceiling
# hits — not its log, not the model cache, not its CPU time — the run is wedged
# rather than merely slow. This changes the *diagnosis* printed in the failure
# detail, not the verdict: both are failures, because "it was still moving" is
# unfalsifiable — a retry loop writing to the log looks exactly like progress —
# and a gate that accepts it can be hung forever without ever saying NO-GO.
FIRST_SEARCH_STALL_S = 600.0
CLI_HELP_TIMEOUT_S = 120.0
IMPORT_PROBE_TIMEOUT_S = 120.0
# How long the child waits for an asynchronous chunk count to appear before
# concluding the number is wrong rather than merely late.
CHUNK_COUNT_POLL_S = 30.0


def _timeout_ceiling() -> float:
    """Journey ceiling, overridable for genuinely slow links."""
    raw = os.environ.get("E2E_P2_TIMEOUT_S", "").strip()
    if raw:
        try:
            return max(60.0, float(raw))
        except ValueError:
            pass
    return FIRST_SEARCH_TIMEOUT_S


# ---------------------------------------------------------------------------
# Interpreter selection
# ---------------------------------------------------------------------------


def _python() -> str:
    """Interpreter used for every child process.

    ``E2E_PYTHON`` wins so CI can point this at whatever it built. Otherwise fall
    back to a known-good local venv, then to the interpreter running the gate.
    Note this is *not* a clean environment — see the module docstring.
    """
    env_py = os.environ.get("E2E_PYTHON")
    if env_py and Path(env_py).exists():
        return env_py
    # No hardcoded fallback: one pointing at a particular developer's
    # virtualenv works on exactly one machine, and writes that path into every
    # report the gate commits.
    return sys.executable


# ---------------------------------------------------------------------------
# Child environment
# ---------------------------------------------------------------------------

# Passed through when present. Proxy and CA settings are here because without
# them a corporate network cannot reach Hugging Face at all, and "no network" is
# not the defect this phase is looking for. API keys are deliberately absent.
_PASSTHROUGH = (
    "PATH",
    "LANG",
    "LC_ALL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def _child_env(sandbox: Path) -> Dict[str, str]:
    env: Dict[str, str] = {k: os.environ[k] for k in _PASSTHROUGH if os.environ.get(k)}
    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    env.update(
        {
            "HOME": str(sandbox / "home"),
            # TMPDIR is deliberately NOT redirected into the sandbox. milvus-lite
            # opens its control socket under the temp dir, and a sandbox path is
            # long enough to exceed the ~104-byte sockaddr_un limit on macOS, so
            # pointing TMPDIR here fails the phase for a reason that has nothing
            # to do with Lasagna. Cost of that choice, stated plainly: milvus's
            # socket and any other tempfiles land in the system temp dir and are
            # not removed by this phase's cleanup.
            #
            # Three cache variables, because the model stack has three
            # generations of cache convention and any one of them leaking writes
            # gigabytes into the developer's real cache.
            "HF_HOME": str(sandbox / "cache"),
            "HUGGINGFACE_HUB_CACHE": str(sandbox / "cache" / "hub"),
            "SENTENCE_TRANSFORMERS_HOME": str(sandbox / "cache"),
            "RAG_DB_DIR": str(sandbox / "data" / "dbs"),
            "RAG_UPLOAD_DIR": str(sandbox / "data" / "uploads"),
            "PYTHONPATH": str(SRC),
            "PYTHONUNBUFFERED": "1",
            # Silences a fork warning that would otherwise pollute the log we
            # attach to failures.
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _make_sandbox() -> Path:
    sandbox = Path(tempfile.mkdtemp(prefix="lasagna-e2e-p2-"))
    for sub in ("home", "cache", "data/dbs", "data/uploads"):
        (sandbox / sub).mkdir(parents=True, exist_ok=True)
    return sandbox


def _dir_bytes(path: Path) -> int:
    """Recursive apparent size in bytes. 0 for a missing directory."""
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total


def _dir_mb(path: Path) -> float:
    """Recursive apparent size in MB. 0.0 for a missing directory."""
    return round(_dir_bytes(path) / 1e6, 1)


def _in_cache(cache: Path, model: str) -> bool:
    """Whether ``model`` was fetched into ``cache`` (as HF_HOME or its hub folder).

    A bare sentence-transformers name ("all-MiniLM-L6-v2") is that org's model.
    """
    repo = model if "/" in model else f"sentence-transformers/{model}"
    folder = "models--" + repo.replace("/", "--")
    return any((root / folder / "snapshots").is_dir() for root in (cache, cache / "hub"))


class Child(NamedTuple):
    """Outcome of one child process."""

    #: Exit status, or None if the timeout ceiling was hit.
    rc: Optional[int]
    out: str
    elapsed: float
    #: Seconds since anything observable about the child last changed (its log,
    #: the watched directory, or its process group's CPU time), when it was
    #: watched. Lets a caller tell "still working when the clock ran out" from
    #: "wedged" — a difference in diagnosis, not in verdict.
    stalled_s: Optional[float] = None


def _run(
    args: List[str],
    env: Dict[str, str],
    timeout: float,
    progress_path: Optional[Path] = None,
) -> Child:
    """Run a child under a timeout, optionally watching a directory for progress.

    Children run with cwd inside the sandbox, not in the repo, because Lasagna
    falls back to ``./rag_uploads`` and ``./dbs`` when no config provider
    resolves, and a gate that silently litters the working tree is its own bug.

    Output goes to a file rather than a pipe. A pipe would need draining while we
    poll, and an undrained 64 KB pipe buffer deadlocks a chatty child — which
    this one is, at DEBUG level.
    """
    t0 = time.monotonic()
    workdir = Path(env.get("HOME", tempfile.gettempdir()))
    log = workdir / f"child-{os.urandom(4).hex()}.log"

    # start_new_session puts the child in its own process group so a timeout can
    # kill the whole tree. Signalling only the direct child would leave the
    # milvus-lite server the ingest path forks still running — still
    # downloading, still holding files — inside a sandbox we are about to delete.
    try:
        # The file object is closed as soon as Popen returns; the child holds its
        # own duplicated descriptor and keeps writing.
        with open(log, "w") as fh:
            proc = subprocess.Popen(
                args,
                env=env,
                cwd=str(workdir),
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
    except OSError as exc:
        return Child(-1, f"{type(exc).__name__}: {exc}", time.monotonic() - t0)

    # start_new_session makes the child a session and group leader, so its pgid
    # equals its pid. Capture it now: after the child is reaped os.getpgid()
    # raises, and the group is exactly what needs killing — the ingest path forks
    # a milvus-lite server that outlives its parent.
    pgid = proc.pid

    def _read_log() -> str:
        try:
            return log.read_text(errors="replace")
        except OSError:
            return ""

    slow = [0.0, 0.0, 0.0]  # [next sample time, last dir bytes, last group CPU s]

    def _progress_signal(now: float) -> tuple:
        """Everything observable about whether the child is still doing anything.

        Three signals, because each one is blind on its own — and two of them
        were the whole signal in earlier versions of this phase, which is how it
        managed to mis-diagnose healthy runs:

        * **child log bytes.** Silent for the entire download: Hugging Face's
          progress bars are suppressed when stdout is not a TTY.
        * **model cache bytes.** Measured here, a 2 GiB bge-m3 blob jumps from
          0 to its full apparent size the moment ``hf_xet`` pre-allocates the
          sparse file, then does not change again while the bytes stream in.
          Allocated blocks do not move either — xet buffers chunks before
          finalising. So the cache can look frozen for the whole download.
        * **process-group CPU time**, via ``ps``. This keeps ticking through
          exactly the window the other two miss, and goes flat for a process
          that is genuinely stuck on a lock or a dead socket.

        What it still cannot do is tell useful work from a retry loop. That is
        why the caller fails a timed-out run either way: this tuple diagnoses,
        it does not exonerate.

        The directory walk and the ``ps`` call are throttled to once every 10s —
        walking a multi-gigabyte cache every 2s costs enough I/O to slow the very download
        this phase is timing.
        """
        try:
            log_bytes = float(log.stat().st_size)
        except OSError:
            log_bytes = 0.0
        if now >= slow[0]:
            slow[0] = now + 10.0
            slow[1] = float(_dir_bytes(progress_path)) if progress_path is not None else 0.0
            slow[2] = _group_cpu_seconds(pgid)
        return (log_bytes, slow[1], slow[2])

    watching = progress_path is not None
    last_signal: Optional[tuple] = None
    last_change = t0
    try:
        while True:
            try:
                rc = proc.wait(timeout=2.0)
                return Child(
                    rc,
                    _read_log(),
                    time.monotonic() - t0,
                    (time.monotonic() - last_change) if watching else None,
                )
            except subprocess.TimeoutExpired:
                pass

            if watching:
                signal_now = _progress_signal(time.monotonic())
                if signal_now != last_signal:
                    last_signal, last_change = signal_now, time.monotonic()

            if time.monotonic() - t0 >= timeout:
                stalled = time.monotonic() - last_change
                _kill_group(pgid, proc)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                return Child(
                    None, _read_log(), time.monotonic() - t0, stalled if watching else None
                )
    finally:
        # Also on the *normal* exit path: a child that exits cleanly can still
        # leave the forked milvus-lite server running, holding open files in a
        # sandbox this phase is about to delete. Signalling the group after the
        # child is reaped risks hitting a recycled pgid in theory; in practice
        # the window is microseconds and this process forks nothing in it, which
        # is a better trade than leaking a server per gate run.
        _kill_group(pgid, proc)


def _group_cpu_seconds(pgid: int) -> float:
    """Cumulative CPU time of every process in ``pgid``, via ``ps``.

    ``ps -Ao pgid=,time=`` and filtering here rather than ``ps -g``: the meaning
    of ``-g`` differs between BSD and GNU ps (process group on macOS, session or
    effective group on Linux), and a flag that selects the wrong thing would
    silently report 0 forever — which reads as "wedged".

    Returns 0.0 if ps is unavailable or unparseable. A constant 0.0 makes this
    signal inert rather than wrong: the other two still move.
    """
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pgid=,time="], capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    want = str(pgid)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != want:
            continue
        secs = 0.0
        for chunk in parts[-1].split(":"):  # [[HH:]MM:]SS.ss
            try:
                secs = secs * 60 + float(chunk)
            except ValueError:
                secs = secs * 60
        total += secs
    return round(total, 2)


def _kill_group(pgid: int, proc: "subprocess.Popen") -> None:
    """SIGKILL a process group, falling back to the direct child."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _last_json(output: str, tag: str = "E2E_JSON:") -> Optional[Dict[str, Any]]:
    """Pull the child's result line out of a log full of loguru chatter."""
    for line in reversed(output.splitlines()):
        if line.startswith(tag):
            try:
                return json.loads(line[len(tag) :])
            except json.JSONDecodeError:
                return None
    return None


def _tail(text: str, n: int = 1500) -> str:
    text = text.strip()
    return text if len(text) <= n else "...\n" + text[-n:]


# ---------------------------------------------------------------------------
# Child programs
# ---------------------------------------------------------------------------

# Measures `import lasagna` in a process that has imported nothing else heavy,
# and reports which forbidden modules ended up in sys.modules. It checks
# sys.modules rather than patching the import machinery, so it catches a
# transitive pull-in as well as a direct one.
_IMPORT_PROBE = r"""
import json, sys, time
FORBIDDEN = %(forbidden)r
pre = sorted(m for m in FORBIDDEN if m in sys.modules)
t0 = time.perf_counter()
try:
    import lasagna
    elapsed = time.perf_counter() - t0
    payload = {
        "ok": True,
        "elapsed": elapsed,
        "file": lasagna.__file__,
        "preloaded": pre,
        "leaked": sorted(m for m in FORBIDDEN if m in sys.modules),
        "n_public": len(getattr(lasagna, "__all__", [])),
    }
except BaseException as exc:
    import traceback
    payload = {
        "ok": False,
        "elapsed": time.perf_counter() - t0,
        "error": "%%s: %%s" %% (type(exc).__name__, exc),
        "tb": traceback.format_exc()[-1500:],
    }
print("E2E_JSON:" + json.dumps(payload))
""" % {"forbidden": list(FORBIDDEN_AT_IMPORT)}


# Enumerates the CLI the way Click sees it, so the names used for `--help` are
# the names a user actually types (Typer rewrites some of them).
_COMMAND_LIST = r"""
import json
try:
    import typer.main
    from lasagna.cli import app
    group = typer.main.get_command(app)
    print("E2E_JSON:" + json.dumps({"ok": True, "commands": sorted(group.commands)}))
except BaseException as exc:
    import traceback
    print("E2E_JSON:" + json.dumps({
        "ok": False,
        "error": "%s: %s" % (type(exc).__name__, exc),
        "tb": traceback.format_exc()[-1500:],
    }))
"""


# The headline journey: ingest one small text file and search it, timing the
# whole thing including the model download. Deliberately uses the documented
# public SDK surface (lasagna.sdk.KnowledgeClient) rather than internals.
_FIRST_SEARCH = r"""
import asyncio, json, os, time, traceback

DOC = os.environ["E2E_DOC"]
SESSION = os.environ["E2E_SESSION"]
POLL_S = float(os.environ["E2E_CHUNK_POLL_S"])
QUERY = "What is the melting point of gallium?"
EXPECT = "29.76"
# The default configuration's query router sends a short factual question like
# QUERY to its fast tier, which does not rerank, so the reranker is fetched by
# the first query that is reranked. This one is, so the download measured
# after the journey is the configuration's whole first run, not its embedder.
RERANKED_QUERY = "Which semiconductors use gallium?"

async def _count_for(client, document_id):
    listed = await client.list_documents()
    doc = next(
        (d for d in listed if getattr(d, "document_id", None) == document_id), None
    )
    return (
        len(listed),
        getattr(doc, "chunk_count", None) if doc is not None else None,
        str(getattr(doc, "status", "")) if doc is not None else "",
    )

async def journey(out):
    from lasagna.sdk import KnowledgeClient

    client = KnowledgeClient(session_id=SESSION)
    try:
        t_up = time.monotonic()
        up = await client.upload(DOC)
        out["ingest_s"] = time.monotonic() - t_up
        out["document_id"] = getattr(up, "document_id", None)
        out["chunk_count"] = getattr(up, "chunk_count", None)
        out["status"] = str(getattr(up, "status", ""))

        # The number `lasagna upload` prints comes straight off UploadResult.
        # Poll list_documents() for a bounded window so a count that is merely
        # *late* (indexing still running when upload() returned) is not reported
        # as a count that is *wrong*. Only a count that never arrives, on a
        # document search can already find, is a defect.
        t_poll = time.monotonic()
        while True:
            try:
                docs, count, status = await _count_for(client, out["document_id"])
            except BaseException as exc:
                out["list_error"] = "%s: %s" % (type(exc).__name__, exc)
                break
            out["listed_docs"], out["listed_chunk_count"] = docs, count
            out["listed_status"] = status
            if count:
                break
            if time.monotonic() - t_poll >= POLL_S:
                break
            await asyncio.sleep(1.0)
        out["chunk_poll_s"] = time.monotonic() - t_poll

        t_se = time.monotonic()
        results = await client.search_documents(QUERY, top_k=5)
        out["search_s"] = time.monotonic() - t_se

        out["n_results"] = len(results)
        if results:
            top = results[0]
            content = getattr(getattr(top, "chunk", None), "content", "") or ""
            out["top_score"] = float(getattr(top, "score", 0.0) or 0.0)
            out["top_snippet"] = content[:300]
            out["answer_present"] = any(
                EXPECT in (getattr(getattr(r, "chunk", None), "content", "") or "")
                for r in results
            )
        out["ok"] = bool(results)

        # One more read after the search: if the search itself is what forces
        # indexing to settle, the count should exist by now. If it still does
        # not, two public APIs disagree about the same document.
        try:
            _, count, status = await _count_for(client, out["document_id"])
            out["listed_chunk_count_final"] = count
            out["listed_status_final"] = status
        except BaseException as exc:
            out["list_error_final"] = "%s: %s" % (type(exc).__name__, exc)
        out["first_search_elapsed"] = time.monotonic() - t0

        t_rr = time.monotonic()
        try:
            reranked = await client.search_documents(RERANKED_QUERY, top_k=5)
            out["reranked_n_results"] = len(reranked)
        except BaseException as exc:
            out["reranked_error"] = "%s: %s" % (type(exc).__name__, exc)
        out["reranked_search_s"] = time.monotonic() - t_rr

        from lasagna.core.config import rag_config

        config = rag_config.with_session(SESSION)
        out["models"] = [config.embedding_model] + (
            [config.reranker_model] if config.enable_reranking else []
        )
    finally:
        try:
            await client.close()
        except BaseException:
            pass

out = {"ok": False}
t0 = time.monotonic()
try:
    asyncio.run(journey(out))
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["tb"] = traceback.format_exc()[-2000:]
out["elapsed"] = time.monotonic() - t0
print("E2E_JSON:" + json.dumps(out))
"""


SAMPLE_DOC = """Gallium: properties and uses

Gallium is a soft, silvery post-transition metal with atomic number 31.
Its melting point is 29.76 degrees Celsius, low enough that a solid piece
melts in the palm of a hand. Its boiling point, by contrast, is 2400 degrees
Celsius, giving gallium one of the largest liquid ranges of any element.

Gallium arsenide and gallium nitride are used in high-speed semiconductors,
laser diodes and light-emitting diodes. Gallium was predicted by Mendeleev as
eka-aluminium and discovered in 1875 by Paul-Emile Lecoq de Boisbaudran.
"""


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


def run() -> PhaseResult:
    """Entry point. Never raises: a phase that dies takes the whole gate with it.

    Everything, including the pre-flight snapshots, happens inside the guard.
    ``Path.home()`` raises when HOME is unset, and a gate that crashes while
    working out where the user's cache lives is a gate nobody gets a report from.
    """
    p = PhaseResult(key="p2", title="Cold start")
    try:
        _run_phase(p)
    except Exception as exc:
        p.check(
            "p2 completed without an internal error",
            False,
            severity=Severity.CRITICAL,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
            repro="python -m e2e.run --phase p2",
        )
    return p


def _guard(p: PhaseResult, label: str, fn: Callable[..., Any], *args: Any) -> None:
    """Run one step so its crash cannot erase the steps that follow.

    A single try around the whole phase means an exception in the first step
    throws away the CLI checks, the journey, and every metric already measured —
    the report says "it broke and here is nothing", which is the output this
    system exists to replace. Each step's crash is its own CRITICAL failure.
    """
    try:
        fn(p, *args)
    except Exception as exc:
        p.check(
            f"p2 step '{label}' completed without an internal error",
            False,
            severity=Severity.CRITICAL,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
            repro="python -m e2e.run --phase p2",
        )


def _run_phase(p: PhaseResult) -> None:
    py = _python()
    if not Path(py).exists():
        p.skipped_reason = f"no usable Python interpreter (tried {py})"
        return
    if not (SRC / "lasagna" / "__init__.py").exists():
        p.skipped_reason = f"lasagna source not found at {SRC}"
        return

    base_repro = f"PYTHONPATH={SRC} {py}"

    sandbox: Optional[Path] = None
    # Snapshot the two locations a leak would land in, so the isolation claim is
    # asserted rather than asserted-by-comment.
    real_data = Path.home() / ".rag_knowledge"
    before = (
        real_data.exists(),
        _dir_mb(real_data),
        _dir_mb(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))),
    )

    try:
        sandbox = _make_sandbox()
        env = _child_env(sandbox)

        _guard(p, "import purity", _import_purity, py, env, base_repro)
        _guard(p, "cli help", _cli_help, py, env, base_repro)
        _guard(p, "first search", _first_search, py, env, sandbox, base_repro)
        # If the journey step died before it could report — a crash inside the
        # phase itself, not in the product — its assertions are missing rather
        # than failed, and a missing check reads exactly like a passing one to
        # anyone counting rows in the report. Name them.
        reported = {c.name for c in p.checks}
        for name in _JOURNEY_CHECKS:
            if name not in reported:
                p.skip(name, "the journey step did not run to the point of asserting this")
        _guard(p, "isolation", _isolation, real_data, before)
    finally:
        if sandbox is not None:
            shutil.rmtree(sandbox, ignore_errors=True)
            # rmtree(ignore_errors=True) fails silently, and a sandbox holding
            # gigabytes of weights that silently survives every run is a real cost to
            # whoever runs this. Assert the cleanup rather than trusting it.
            leftover = _dir_mb(sandbox)
            p.check(
                "the throwaway sandbox is removed",
                not sandbox.exists(),
                severity=Severity.WARN,
                detail=(
                    f"{sandbox} still holds {leftover} MB"
                    if sandbox.exists()
                    else f"{sandbox} removed"
                ),
                repro=f"ls -la {sandbox.parent}/lasagna-e2e-p2-*",
            )
            p.measure(
                "sandbox_leftover",
                leftover,
                "MB",
                note="left behind after cleanup; excludes milvus-lite's socket in "
                "TMPDIR, which this phase does not sandbox (see _child_env)",
            )

    # Honest about what running from source cannot show. A skip renders as
    # unvalidated, which is the truth here — not a pass.
    p.skip(
        "console script 'lasagna' is installed and on PATH",
        "p2 runs from source via PYTHONPATH and drives the CLI as "
        "`python -m lasagna`; the [project.scripts] entry point is only created "
        "by a real install, which phase p1 covers.",
    )


def _isolation(p: PhaseResult, real_data: Path, before: tuple) -> None:
    """The run must leave the developer's real data dir and model cache alone.

    A package that ignores HF_HOME/RAG_DB_DIR dumps gigabytes somewhere the user
    did not choose, so this is a correctness property, not hygiene.

    What it detects: creation of ~/.rag_knowledge, and growth of either the real
    data dir or the real model cache. What it does not detect: a rewrite in place
    that leaves total size unchanged. What can make it lie: an unrelated process
    downloading a model on the same machine during the run, which would show up
    as growth this phase did not cause.
    """
    data_existed, data_before_mb, cache_before_mb = before
    real_cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    data_after, data_after_mb, cache_after_mb = (
        real_data.exists(),
        _dir_mb(real_data),
        _dir_mb(real_cache),
    )
    leaked_data = data_after_mb - data_before_mb
    leaked_cache = cache_after_mb - cache_before_mb
    p.check(
        "cold start writes nothing to the real ~/.rag_knowledge or model cache",
        (data_after == data_existed) and leaked_data < 0.1 and leaked_cache < 1.0,
        severity=Severity.CRITICAL,
        detail=(
            f"{real_data}: existed={data_existed}->{data_after}, "
            f"{data_before_mb} MB -> {data_after_mb} MB ({leaked_data:+.1f}); "
            f"{real_cache}: {cache_before_mb} MB -> {cache_after_mb} MB "
            f"({leaked_cache:+.1f})"
        ),
        repro=(
            f"du -sm {real_data} {real_cache}; python -m e2e.run --phase p2; "
            f"du -sm {real_data} {real_cache}"
        ),
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _import_purity(p: PhaseResult, py: str, env: Dict[str, str], base_repro: str) -> None:
    """Import speed and lazy-import purity.

    Note what this does not prove: that every *public* name is reachable. It
    imports the package, not its lazy attributes, so a broken submodule behind
    `lasagna.RAGBackend` would still pass here.
    """
    rc, out, wall, _ = _run([py, "-c", _IMPORT_PROBE], env, IMPORT_PROBE_TIMEOUT_S)
    payload = _last_json(out)

    # Measure first, assert after: the number is worth having even if the
    # assertions below fail.
    if payload and payload.get("elapsed") is not None:
        p.measure(
            "import_time",
            round(float(payload["elapsed"]), 3),
            "s",
            note="`import lasagna` in a fresh process; source-tree __pycache__ is "
            "warm, so a real cold install is slower",
        )
    else:
        p.measure(
            "import_time", round(wall, 3), "s", note="process wall time; probe produced no result"
        )

    repro = f'{base_repro} -c "import lasagna"'

    if payload is None:
        p.check(
            "`import lasagna` succeeds",
            False,
            severity=Severity.CRITICAL,
            detail=f"probe produced no result (rc={rc})\n{_tail(out)}",
            repro=repro,
            duration_s=wall,
        )
        # Every downstream assertion is named here rather than silently omitted:
        # a phase whose check count quietly drops from 24 to 21 has stopped
        # verifying three things and told nobody.
        for name in (
            "`import lasagna` resolves to the source tree under test",
            f"`import lasagna` loads none of {', '.join(FORBIDDEN_AT_IMPORT)}",
        ):
            p.skip(name, f"the import probe produced no result (rc={rc})")
        p.skip(
            f"`import lasagna` under {IMPORT_BUDGET_S}s",
            "the import probe produced no result, so its timing means nothing",
            severity=Severity.WARN,
        )
        return

    ok_import = p.check(
        "`import lasagna` succeeds",
        bool(payload.get("ok")),
        severity=Severity.CRITICAL,
        detail=payload.get("error", "")
        or f"{payload.get('file')} ({payload.get('n_public')} public names)",
        repro=repro,
        duration_s=wall,
    )

    purity_name = f"`import lasagna` loads none of {', '.join(FORBIDDEN_AT_IMPORT)}"
    if not ok_import:
        p.skip(
            "`import lasagna` resolves to the source tree under test",
            "import failed, so there is no module to locate",
        )
        p.skip(
            purity_name,
            "import failed, so nothing could be said about which modules it pulls in",
        )
        p.skip(
            f"`import lasagna` under {IMPORT_BUDGET_S}s",
            "import failed, so how long it took is not a measurement of anything",
            severity=Severity.WARN,
        )
        return

    # Everything else in this phase is a statement about *some* copy of lasagna.
    # This is the check that says which copy. PYTHONPATH outranks site-packages,
    # but an editable install registers a meta-path finder that outranks
    # PYTHONPATH — so without this, the entire phase can pass green against a
    # different checkout than the one being released.
    imported = payload.get("file") or ""
    try:
        resolved = str(Path(imported).resolve())
        src_prefix = str(SRC.resolve()) + os.sep
        under_src = resolved.startswith(src_prefix)
    except (OSError, ValueError):
        resolved, under_src = imported, False
    p.check(
        "`import lasagna` resolves to the source tree under test",
        under_src,
        severity=Severity.CRITICAL,
        detail=(
            f"lasagna.__file__ = {resolved}"
            + ("" if under_src else f" — which is NOT under {SRC}; this phase tested other code")
        ),
        repro=f'{base_repro} -c "import lasagna; print(lasagna.__file__)"',
    )

    preloaded = list(payload.get("preloaded", []))
    if preloaded:
        # A forbidden module already in sys.modules before the import (a
        # sitecustomize, an interpreter with a preloaded torch) makes purity
        # unknowable rather than fine. Subtracting it and reporting "clean" — the
        # previous behaviour — reports an unrun check as passed.
        p.skip(
            purity_name,
            f"{', '.join(preloaded)} was already in sys.modules before `import lasagna`, "
            f"so this interpreter cannot attribute the import to lasagna. "
            f"Re-run with E2E_PYTHON pointed at an interpreter without a sitecustomize.",
        )
    else:
        leaked = list(payload.get("leaked", []))
        p.check(
            purity_name,
            not leaked,
            severity=Severity.CRITICAL,
            detail=(
                f"eagerly imported: {', '.join(leaked)}"
                if leaked
                else "sys.modules clean after import"
            ),
            repro=(
                f'{base_repro} -c "import sys, lasagna; '
                f'print([m for m in {list(FORBIDDEN_AT_IMPORT)} if m in sys.modules])"'
            ),
        )

    elapsed = float(payload.get("elapsed") or wall)
    p.check(
        f"`import lasagna` under {IMPORT_BUDGET_S}s",
        elapsed < IMPORT_BUDGET_S,
        severity=Severity.WARN,
        detail=f"{elapsed:.3f}s (budget {IMPORT_BUDGET_S}s)",
        repro=f'{base_repro} -X importtime -c "import lasagna"',
        duration_s=elapsed,
    )


def _cli_help(p: PhaseResult, py: str, env: Dict[str, str], base_repro: str) -> None:
    """`--help` at the top level, on every registered command, and on the
    quickstart's commands whether or not they are registered.

    Help text is the cheapest possible smoke test of the import graph behind each
    command, and the one thing a lost user reaches for first. It proves the
    command *loads*; it proves nothing about whether it works.

    The probe list is the union of what enumeration finds and what the README
    quickstart documents. That union is the point: enumeration failing used to
    skip the whole loop, so a CLI that registered nothing at all — or a Typer
    upgrade that broke `get_command` — produced zero assertions and a green
    phase. Probing `upload`, `search`, `init` and `serve` by name means a missing
    command fails with `rc != 0` no matter what enumeration says.
    """
    rc, out, wall, _ = _run([py, "-m", "lasagna", "--help"], env, CLI_HELP_TIMEOUT_S)
    p.check(
        "`lasagna --help` works",
        rc == 0 and "Usage" in out,
        severity=Severity.CRITICAL,
        detail=f"rc={rc}\n{_tail(out, 800)}" if rc != 0 else f"rc=0 in {wall:.2f}s",
        repro=f"{base_repro} -m lasagna --help",
        duration_s=wall,
    )
    p.measure(
        "cli_help_time",
        round(wall, 2),
        "s",
        note="top-level `lasagna --help`; the latency a user feels when lost",
    )

    # Enumeration is a *superset* probe, not the source of truth. A broken
    # top-level --help no longer aborts this step: each command's help is an
    # independent assertion, and running them anyway says which ones are broken
    # rather than nothing at all.
    enum_rc, enum_out, _wall, _ = _run([py, "-c", _COMMAND_LIST], env, CLI_HELP_TIMEOUT_S)
    listing = _last_json(enum_out)
    enumerated: List[str] = (
        sorted(listing.get("commands", [])) if listing and listing.get("ok") else []
    )

    registered_name = "the CLI registers every command the README quickstart documents"
    enum_repro = (
        f'{base_repro} -c "import typer.main; from lasagna.cli import app; '
        f'print(sorted(typer.main.get_command(app).commands))"'
    )
    if enumerated:
        missing = [c for c in QUICKSTART_COMMANDS if c not in enumerated]
        p.check(
            registered_name,
            not missing,
            severity=Severity.CRITICAL,
            detail=(
                f"registered: {', '.join(enumerated)}"
                + (f" — MISSING from the CLI: {', '.join(missing)}" if missing else "")
            ),
            repro=enum_repro,
        )
    else:
        p.skip(
            registered_name,
            f"could not enumerate commands from lasagna.cli (rc={enum_rc}): "
            f"{_tail(enum_out, 300) or 'no output'}. Each quickstart command is "
            f"still probed directly below, so this skip loses coverage of the "
            f"non-quickstart commands only.",
        )

    commands = sorted(set(enumerated) | set(QUICKSTART_COMMANDS))
    failures: List[str] = []
    for name in commands:
        c_rc, c_out, c_wall, _ = _run(
            [py, "-m", "lasagna", name, "--help"], env, CLI_HELP_TIMEOUT_S
        )
        # Click prints a usage line on its "No such command" error too, so
        # "Usage" alone would pass for a command that does not exist. The exit
        # code is what distinguishes them.
        ok = c_rc == 0 and "Usage" in c_out and "No such command" not in c_out
        if not ok:
            failures.append(f"{name}: rc={c_rc}")
        p.check(
            f"`lasagna {name} --help` works",
            ok,
            severity=Severity.CRITICAL,
            detail=f"rc={c_rc}\n{_tail(c_out, 600)}" if not ok else f"rc=0 in {c_wall:.2f}s",
            repro=f"{base_repro} -m lasagna {name} --help",
            duration_s=c_wall,
        )

    p.measure(
        "cli_commands_probed",
        float(len(commands)),
        "count",
        lower_is_better=False,
        note=f"{', '.join(commands)}" + (f" | {len(failures)} failed" if failures else ""),
    )


def _first_search(
    p: PhaseResult,
    py: str,
    env: Dict[str, str],
    sandbox: Path,
    base_repro: str,
) -> None:
    """Time-to-first-search on a cold machine, model download included.

    This is the number that decides whether a first-time user stays. It is
    dominated by the embedding model the default configuration fetches, which
    is the point: an install that is fast only when someone else already
    downloaded the models is not the install being shipped. A second query, one
    the router reranks, then fetches the reranker, so ``model_download`` is the
    configuration's whole first-run download.

    Five things are asserted about the product here, in the order a reader wants
    them, and all five are CRITICAL because all five are correctness:

    1. the journey finishes at all, inside the ceiling;
    2. the document is ingested;
    3. the bytes land where ``RAG_DB_DIR``/``RAG_UPLOAD_DIR`` said to put them;
    4. the search returns something, and returns *the document it was given* —
       the corpus is one document and the answer is in it;
    5. the chunk count the user is shown agrees with what search can find.

    Only the wall-clock budget is WARN, because only it is performance.
    """
    doc = sandbox / "gallium.txt"
    doc.write_text(SAMPLE_DOC, encoding="utf-8")

    cache_dir = Path(env["HF_HOME"])
    data_dir = sandbox / "data"
    ceiling = _timeout_ceiling()

    child_env = dict(env)
    child_env["E2E_DOC"] = str(doc)
    child_env["E2E_SESSION"] = SESSION_ID
    child_env["E2E_CHUNK_POLL_S"] = str(CHUNK_COUNT_POLL_S)

    script = sandbox / "first_search.py"
    script.write_text(_FIRST_SEARCH, encoding="utf-8")

    repro = _journey_repro(py)

    with timed() as t:
        rc, out, wall, stalled_s = _run(
            [py, str(script)], child_env, ceiling, progress_path=cache_dir
        )
    payload = _last_json(out)

    # -- measure first, unconditionally ------------------------------------
    # Up to the first search's result, as before the reranked query was added.
    first = payload and (payload.get("first_search_elapsed") or payload.get("elapsed"))
    elapsed = float(first) if first else wall
    timed_out = rc is None
    p.measure(
        "time_to_first_search",
        round(elapsed, 1),
        "s",
        note=(
            "cold cache: ingest one text file + one search, model download included"
            + ("; TIMED OUT, value is the timeout" if timed_out else "")
        ),
    )
    p.measure(
        "model_download",
        _dir_mb(cache_dir),
        "MB",
        note="fetched into an empty HF_HOME by the default configuration: one upload, "
        "the first search and the first reranked search",
    )
    p.measure(
        "data_dir_size",
        _dir_mb(data_dir),
        "MB",
        note="RAG_DB_DIR + RAG_UPLOAD_DIR after ingesting one ~700-byte document",
    )
    if payload and payload.get("ingest_s"):
        p.measure(
            "ingest_time",
            round(float(payload["ingest_s"]), 1),
            "s",
            note="first upload() call: on a cold cache this is dominated by "
            "fetching and loading the embedding model, not by the document",
        )
    if payload and payload.get("search_s"):
        p.measure(
            "first_query_time",
            round(float(payload["search_s"]), 2),
            "s",
            note="the search itself, with models already resident",
        )
    if payload and payload.get("reranked_search_s"):
        p.measure(
            "first_reranked_query_time",
            round(float(payload["reranked_search_s"]), 1),
            "s",
            note="the first query the router reranks; on a cold cache this includes "
            "fetching the reranker",
        )

    # -- assert after -------------------------------------------------------
    finished = (not timed_out) and payload is not None

    # This is the check that makes the phase capable of failing. Whatever went
    # wrong — wedged, crashed, or still grinding at 45 minutes — the journey did
    # not produce a search result, and that blocks.
    #
    # The wedged/progressing distinction survives, in the detail, because it is
    # the first thing whoever reads this needs. It does not change the verdict:
    # "it was still moving" is unfalsifiable (a retry loop appending to a log
    # looks exactly like progress), and a gate that accepts it can be hung
    # forever without ever saying NO-GO. A genuinely slow link is handled by
    # raising E2E_P2_TIMEOUT_S on purpose, which is recorded in the repro.
    mb = _dir_mb(cache_dir)
    if timed_out:
        moving = stalled_s is not None and stalled_s < FIRST_SEARCH_STALL_S
        diagnosis = (
            (
                f"still working when the ceiling hit (log, model cache or CPU "
                f"time last moved {stalled_s:.0f}s ago) — either genuinely slow, "
                f"or looping. If this machine was contended, re-run idle or "
                f"raise E2E_P2_TIMEOUT_S; a first run with no result after "
                f"{ceiling:.0f}s is a shipping blocker either way."
            )
            if moving
            else (
                f"wedged: no log output, no model-cache change and no CPU time "
                f"for "
                f"{'an unknown period' if stalled_s is None else f'{stalled_s:.0f}s'} "
                f"before the ceiling"
            )
        )
        detail = (
            f"no result after {wall:.0f}s (ceiling {ceiling:.0f}s); {mb} MB fetched. "
            f"{diagnosis}\n{_tail(out)}"
        )
    elif payload is None:
        detail = f"the child exited (rc={rc}) without a result line; {mb} MB fetched\n{_tail(out)}"
    else:
        detail = f"completed in {elapsed:.0f}s (child rc={rc}); {mb} MB fetched"

    p.check(
        f"the cold-start journey finishes within {ceiling:.0f}s",
        finished,
        severity=Severity.CRITICAL,
        detail=detail,
        repro=repro,
        duration_s=t.elapsed,
    )

    if not finished:
        # Named, not omitted: these were never evaluated, and a check that
        # vanishes from the report is indistinguishable from one that passed.
        _unevaluated(p, "the cold-start journey did not complete")
        p.check(
            f"time to first search under {FIRST_SEARCH_BUDGET_S:.0f}s",
            False,
            severity=Severity.WARN,
            detail=f"no result after {elapsed:.0f}s "
            f"(budget {FIRST_SEARCH_BUDGET_S:.0f}s); {mb} MB fetched",
            repro=repro,
            duration_s=elapsed,
        )
        return

    err = payload.get("error", "")
    ingested = p.check(
        "document is ingested",
        bool(payload.get("document_id")),
        severity=Severity.CRITICAL,
        detail=(
            f"document_id={payload.get('document_id')} status={payload.get('status')}"
            if payload.get("document_id")
            else f"upload produced no document: {err or _tail(out, 800)}"
        ),
        repro=repro,
        duration_s=float(payload.get("ingest_s") or 0.0),
    )

    # The isolation check at the end of the phase proves nothing landed in the
    # user's real directories. It cannot prove anything landed in the ones they
    # *chose*: Lasagna falls back to ./dbs and ./rag_uploads when no config
    # provider resolves, and the child's cwd is inside the sandbox too — so a
    # build that ignored RAG_DB_DIR entirely would leave every other check in
    # this phase green while silently writing a user's knowledge base somewhere
    # they did not ask for.
    data_bytes = _dir_bytes(data_dir)
    if ingested:
        p.check(
            "the data lands in the configured RAG_DB_DIR / RAG_UPLOAD_DIR",
            data_bytes > 4096,
            severity=Severity.CRITICAL,
            detail=(
                f"{data_dir} holds {data_bytes / 1e6:.2f} MB after ingest"
                + (
                    ""
                    if data_bytes > 4096
                    else " — the document was ingested but nothing was written where "
                    "RAG_DB_DIR/RAG_UPLOAD_DIR pointed, so the env vars are being ignored"
                )
            ),
            repro=repro,
        )
    else:
        p.skip(
            "the data lands in the configured RAG_DB_DIR / RAG_UPLOAD_DIR",
            "no document was created, so where its data landed says nothing",
        )

    ok_search = p.check(
        "first search returns a result",
        bool(payload.get("ok")) and int(payload.get("n_results") or 0) > 0,
        severity=Severity.CRITICAL,
        detail=(
            f"{payload.get('n_results')} result(s), top score {payload.get('top_score')}"
            if payload.get("ok")
            else f"no results. {err or ''}\n{payload.get('tb', '') or _tail(out, 1200)}"
        ),
        repro=repro,
        duration_s=float(payload.get("search_s") or 0.0),
    )

    # CRITICAL, not WARN. Ranking quality is p4's subject and would be a fair
    # WARN; this is not ranking quality. The corpus is one document, the answer
    # is in it, and the search asks for five results — so failing means the
    # pipeline cannot find the only thing it was given. "Search returned three
    # rows" is not the claim a user cares about; "search found my document" is,
    # and a green phase that only proves the former is decoration.
    if ok_search:
        p.check(
            "top results contain the answer to the query",
            bool(payload.get("answer_present")),
            severity=Severity.CRITICAL,
            detail=(
                f"looked for '29.76' in {payload.get('n_results')} result(s) over a "
                f"one-document corpus; top snippet: "
                f"{(payload.get('top_snippet') or '')[:160]!r}"
            ),
            repro=repro,
        )
    else:
        p.skip(
            "top results contain the answer to the query",
            "search returned nothing, so relevance could not be evaluated",
        )

    # The count is what `lasagna upload` prints back. A user told "Chunks: 0"
    # about a document that is in fact indexed concludes the upload failed.
    #
    # This was WARN, on the argument that it is a trust bug rather than a broken
    # install, and on an observation — chunk_count=0 *and* list_documents()=0
    # while search returned rows — that turns out to have been a measurement
    # artefact: the earlier probe listed once, immediately, and read the count
    # mid-indexing. Polling shows it settling to the right number about a second
    # later. So the defect is smaller than it looked and the assertion is
    # sharper: the count must become discoverable, and must agree with what
    # search can find.
    #
    # CRITICAL at that sharpened form, per e2e/README.md's rule with no
    # exception — "Correctness is never WARN" — because what is left is not a
    # performance threshold: it fires only when two public APIs contradict each
    # other about the same document, search finding content that every count
    # says does not exist. It can no longer fire on a count that was merely late.
    #
    # What is deliberately *not* blocked: `upload()` returning chunk_count=0 with
    # status=PROCESSING, which is what the CLI prints. That is an asynchronous
    # contract, not a wrong answer, and the honest way to hold it to account is
    # the settle-time metric below — a 1s settle becoming a 60s settle is a
    # regression a reader can see, without the gate blocking a release over a
    # design decision.
    reported = payload.get("chunk_count")
    listed = payload.get("listed_chunk_count")
    listed_final = payload.get("listed_chunk_count_final")
    settled = float(payload.get("chunk_poll_s") or 0.0)
    p.measure(
        "chunk_count_settle",
        round(settled, 1),
        "s",
        note=(
            f"after upload() returned chunk_count={reported} (status="
            f"{payload.get('status')}), how long until list_documents() reported "
            f"a non-zero count — this is the window in which `lasagna upload` "
            f"prints 'Chunks: 0' for a document that is being indexed"
        ),
    )
    if ok_search:
        counts_agree = bool(reported or listed or listed_final)
        p.check(
            "upload reports the number of chunks it created",
            counts_agree,
            severity=Severity.CRITICAL,
            detail=(
                f"UploadResult.chunk_count={reported} (status={payload.get('status')}); "
                f"list_documents() said {listed} after {settled:.0f}s of polling "
                f"(status={payload.get('listed_status')}) and {listed_final} after the "
                f"search (status={payload.get('listed_status_final')}); search returned "
                f"{payload.get('n_results')} result(s) from that document"
                + (
                    " — the count settled and agrees with what search can find"
                    if counts_agree
                    else " — the content is indexed and every count still says zero, so "
                    "the two public APIs contradict each other"
                )
                + (f"; list error: {payload['list_error']}" if payload.get("list_error") else "")
            ),
            repro=repro,
        )
    else:
        p.skip(
            "upload reports the number of chunks it created",
            "search returned nothing, so a zero chunk count is consistent rather "
            "than contradictory — the search failure above is the defect to fix first",
        )

    p.check(
        f"time to first search under {FIRST_SEARCH_BUDGET_S:.0f}s",
        elapsed < FIRST_SEARCH_BUDGET_S,
        severity=Severity.WARN,
        detail=f"{elapsed:.1f}s on a cold cache (budget {FIRST_SEARCH_BUDGET_S:.0f}s); "
        f"downloaded {mb} MB",
        repro=repro,
        duration_s=elapsed,
    )

    # model_download is what the docs' first-run size is checked against
    # (tests/test_pypi_readme.py), so it has to include every model the
    # configuration fetches. It measured the embedder alone, twice over, while
    # the docs described the embedder and the reranker.
    models = list(payload.get("models") or [])
    missing = [m for m in models if not _in_cache(cache_dir, m)]
    if models and not missing:
        detail = f"cached after the journey: {', '.join(models)}"
    else:
        detail = f"not fetched: {', '.join(missing) or 'the child reported no models'}"
        if payload.get("reranked_error"):
            detail += f"; the reranked query failed: {payload['reranked_error']}"
        detail += "; check that RERANKED_QUERY is still routed to a tier that reranks"
    p.check(
        "the download measured includes every model the configuration uses",
        bool(models) and not missing,
        severity=Severity.WARN,
        detail=detail,
        repro=repro,
    )


#: Every assertion the journey makes about the product, in report order. Kept in
#: one place so the "we never got to run these" path cannot drift out of sync
#: with the path that runs them — a check that silently stops being emitted is
#: the same defect as a check that lies.
_JOURNEY_CHECKS = (
    "document is ingested",
    "the data lands in the configured RAG_DB_DIR / RAG_UPLOAD_DIR",
    "first search returns a result",
    "top results contain the answer to the query",
    "upload reports the number of chunks it created",
    "the download measured includes every model the configuration uses",
)


def _unevaluated(p: PhaseResult, reason: str) -> None:
    for name in _JOURNEY_CHECKS:
        p.skip(name, reason)


def _journey_repro(py: str) -> str:
    """A single shell line that actually reproduces the cold-start journey.

    The previous repro named ``gallium.txt`` in a sandbox this phase deletes on
    the way out, under environment variables it only described in a comment, in
    a different session id than the one it used — three ways to not reproduce
    anything. This builds the sandbox, the document and the environment inline,
    so it can be pasted into a shell as-is.

    It drives the CLI rather than the SDK the child uses. That is deliberate:
    ``lasagna upload`` calls the same ``KnowledgeClient.upload``, and the CLI is
    the surface a reader can check against their own terminal. Kept to one line
    because the report renders ``repro`` inside a markdown code span.
    """
    doc = " ".join(SAMPLE_DOC.split())  # single line; contains no quotes
    envs = (
        'HOME="$d" HF_HOME="$d/cache" HUGGINGFACE_HUB_CACHE="$d/cache/hub" '
        'SENTENCE_TRANSFORMERS_HOME="$d/cache" RAG_DB_DIR="$d/data/dbs" '
        f'RAG_UPLOAD_DIR="$d/data/uploads" PYTHONPATH={SRC}'
    )
    return (
        f"d=$(mktemp -d) && cd \"$d\" && printf %s '{doc}' > gallium.txt && "
        f'env {envs} {py} -m lasagna upload "$d/gallium.txt" -s {SESSION_ID} && '
        f"env {envs} {py} -m lasagna search "
        f"'What is the melting point of gallium?' -k 5 -s {SESSION_ID} && "
        f"env {envs} {py} -m lasagna search "
        f"'Which semiconductors use gallium?' -k 5 -s {SESSION_ID}"
        f"  # downloads the default models; raise E2E_P2_TIMEOUT_S if the gate ran out of patience"
    )
