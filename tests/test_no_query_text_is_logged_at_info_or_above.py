"""No log line at INFO or above carries the text of a query.

A query is what a user asked, and server logs are kept, shipped and read by
people who were not asked. The agentic loop logged each hop's query, each
rewritten query and each sub-query at INFO, and several warnings quoted the
query they were about. Query text goes to DEBUG; a line at INFO or above may
name a query by its index, id or hash.

The check reads every ``logger.<level>(...)`` call under ``src/lasagna``: an
f-string field, a positional argument or a loguru keyword argument whose value
is a variable or attribute named like a query (``query``, ``new_query``,
``sample.question``, ``prompt``) fails it. ``len(query)`` and names ending in
``_count``, ``_id``, ``_idx``, ``_hash`` and the like say nothing of the text.

A tool call's ``arguments`` (a search tool's query among them) and the text a
tool returned (``result_text``) count as query text too: the MCP ``/docs``
tester logged both at INFO.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "lasagna"

LEVELS = {"info", "success", "warning", "error", "critical", "exception"}
QUERY_WORD = re.compile(
    r"(^|_)(query|queries|question|questions|prompt|arguments|result_text)($|_)"
)
NOT_TEXT = ("_count", "_id", "_ids", "_idx", "_index", "_len", "_hash", "_type", "_tokens", "_ms")
#: Calls whose result says nothing of the text passed in.
OPAQUE_CALLS = {"len", "hash_text", "bool", "type", "id"}


def _text_names(node: ast.AST, opaque: bool = False) -> list:
    """Query-like names whose text would reach the message."""
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
        inner = opaque or name in OPAQUE_CALLS or "hash" in name
        found = [n for a in node.args for n in _text_names(a, inner)]
        if isinstance(fn, ast.Attribute):
            found += _text_names(fn.value, inner)
        return found
    for kind, attr in ((ast.Name, "id"), (ast.Attribute, "attr")):
        if isinstance(node, kind):
            label = getattr(node, attr)
            hit = not opaque and QUERY_WORD.search(label) and not label.endswith(NOT_TEXT)
            rest = _text_names(node.value, opaque) if isinstance(node, ast.Attribute) else []
            return ([label] if hit else []) + rest
    return [n for child in ast.iter_child_nodes(node) for n in _text_names(child, opaque)]


def _logger_call(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr not in LEVELS:
        return False
    base = node.func.value
    while isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute):
        base = base.func.value  # logger.bind(...).info, logger.opt(...).warning
    return isinstance(base, ast.Name) and base.id in ("logger", "log", "_logger")


def _offenders():
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not _logger_call(node):
                continue
            values = list(node.args) + [k.value for k in node.keywords if k.arg != "exc_info"]
            names = []
            for value in values:
                if isinstance(value, ast.JoinedStr):
                    for part in value.values:
                        if isinstance(part, ast.FormattedValue):
                            names += _text_names(part.value)
                elif not isinstance(value, ast.Constant):
                    names += _text_names(value)
            if names:
                where = path.relative_to(PACKAGE.parent)
                yield f"{where}:{node.lineno} {node.func.attr} {sorted(set(names))}"


def test_no_log_call_at_info_or_above_carries_query_text():
    found = list(_offenders())
    assert not found, "query text logged at INFO or above:\n" + "\n".join(found)


def test_the_check_sees_each_way_a_query_reaches_a_message():
    source = """
logger.info(f"hop query={query[:50]}")
logger.info("rewritten '{q}'", q=new_query[:80])
logger.warning("failed for %s", sample.question)
logger.bind(x=1).error(f"{prompt}")
logger.info(f"{len(query)} characters, {query_hash}")
logger.debug(f"{query}")
logger.info(f"API call: args={_redact_credentials(arguments)}")
logger.info(f"Extracted text: {result_text[:100]}...")
"""
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if _logger_call(n)]
    flagged = []
    for call in calls:
        values = list(call.args) + [k.value for k in call.keywords]
        names = []
        for value in values:
            if isinstance(value, ast.JoinedStr):
                for part in value.values:
                    if isinstance(part, ast.FormattedValue):
                        names += _text_names(part.value)
            elif not isinstance(value, ast.Constant):
                names += _text_names(value)
        flagged.append(bool(names))
    assert flagged == [True, True, True, True, False, True, True]
