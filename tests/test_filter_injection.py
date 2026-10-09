"""Filter values must not be able to alter the filter.

Backends translate `FilterExpression` into query strings — Milvus expressions,
Lucene queries, SQL. Any value that escapes its literal changes which rows the
query matches, and because filters are how tenant and session scoping are
enforced, that is a cross-tenant read.

These payloads are checked by round-trip: the emitted literal must parse back
to exactly the input. Counting quotes is not enough — an escaped `\\"` looks
like a quote to a naive check and passes something dangerous.
"""

import json

import pytest

from lasagna.vector_backends.filters import FilterExpression, FilterOperator

# Each payload tries to break out of the quoted literal a different way.
BREAKOUT_PAYLOADS = [
    pytest.param('x" or tenant != "', id="close-quote-and-or"),
    pytest.param('x\\" or 1==1 or "', id="escaped-quote-breakout"),
    pytest.param('x\nor tenant != "y', id="newline-injection"),
    pytest.param("x or 1==1", id="unicode-line-separator"),
    pytest.param("x' or '1'=='1", id="single-quote-variant"),
    pytest.param("x\r\ndrop", id="crlf"),
    pytest.param('"; DROP TABLE t; --', id="sql-style"),
    pytest.param("x\\", id="trailing-backslash"),
    pytest.param('""', id="bare-quotes"),
]


def _literal(value: str) -> str:
    expr = FilterExpression(field="tenant", operator=FilterOperator.EQ, value=value)
    return expr.to_milvus().split("==", 1)[1].strip()


class TestMilvusExpressionEscaping:
    @pytest.mark.parametrize("payload", BREAKOUT_PAYLOADS)
    def test_payload_stays_inside_its_literal(self, payload):
        """The emitted literal must decode back to exactly what went in.

        If it decodes to something else, part of the payload became syntax.
        """
        literal = _literal(payload)
        try:
            decoded = json.loads(literal)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"payload produced a malformed literal, so it escaped its quotes: "
                f"{literal!r} ({exc})"
            )
        assert decoded == payload, (
            f"value was altered in translation: sent {payload!r}, filter matches {decoded!r}"
        )

    @pytest.mark.parametrize("ctrl", ["\x00", "\x01", "\x1f"])
    def test_control_characters_are_rejected_not_stripped(self, ctrl):
        """Stripping would silently change which rows match.

        These were previously removed, so "a\\x01b" and "ab" produced the same
        filter — two distinct identifiers collapsing into one, which for a
        tenant filter means one tenant's query matching another's rows.
        """
        with pytest.raises(ValueError, match="control character"):
            _literal(f"a{ctrl}b")

    @pytest.mark.parametrize("ws", ["\n", "\r", "\t"])
    def test_ordinary_whitespace_still_works(self, ws):
        """Whitespace is legitimate in content and must survive, escaped."""
        assert json.loads(_literal(f"a{ws}b")) == f"a{ws}b"

    def test_oversized_value_is_rejected(self):
        with pytest.raises(ValueError, match="too long"):
            _literal("x" * 70000)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_numbers_are_rejected(self, bad):
        """NaN/inf stringify into tokens the expression parser may misread."""
        with pytest.raises(ValueError):
            FilterExpression(field="score", operator=FilterOperator.GT, value=bad).to_milvus()

    def test_in_operator_escapes_every_element(self):
        """A list filter must escape each member, not just the first."""
        expr = FilterExpression(
            field="chunk_id",
            operator=FilterOperator.IN,
            value=["safe", 'evil" or "1"=="1'],
        )
        rendered = expr.to_milvus()
        inner = rendered.split("[", 1)[1].rsplit("]", 1)[0]
        assert json.loads(f"[{inner}]") == ["safe", 'evil" or "1"=="1']
