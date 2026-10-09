"""Backend-agnostic filter expressions for vector queries.

This module provides a portable filter expression system that gets
translated to backend-specific syntax (Milvus, Pinecone, Qdrant, etc.).

Design Philosophy:
1. Composable: Filters can be combined with AND/OR/NOT
2. Type-safe: Python types translate to correct backend types
3. Secure: Values are escaped to prevent injection
4. Readable: Pythonic API for building expressions

Usage Examples:
    # Simple equality
    expr = FilterExpression.eq("session_id", "user123")

    # Compound expression
    expr = (
        FilterExpression.eq("session_id", "user123")
        & FilterExpression.in_list("document_id", ["doc1", "doc2"])
    )

    # Nested JSON field
    expr = FilterExpression.json_eq("metadata", "session_id", "user123")

    # Convert to backend syntax
    milvus_expr = expr.to_milvus()  # 'session_id == "user123"'
    qdrant_expr = expr.to_qdrant()  # {'must': [{'key': 'session_id', 'match': {'value': 'user123'}}]}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class UnsupportedFilterError(Exception):
    """Raised when a filter operator is not supported by a specific backend.

    This ensures silent data leaks never occur — if a backend cannot honor
    a filter, it fails loudly rather than returning unfiltered results.

    Attributes:
        operator: The FilterOperator that is unsupported
        backend: The backend name that doesn't support it
    """

    def __init__(self, operator: "FilterOperator", backend: str, detail: str = ""):
        self.operator = operator
        self.backend = backend
        msg = f"Filter operator '{operator.value}' is not supported by {backend} backend"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class FilterOperator(str, Enum):
    """Supported filter operators."""

    # Comparison
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"

    # Collection
    IN = "in"
    NOT_IN = "not_in"

    # String
    LIKE = "like"
    PREFIX = "prefix"

    # Logical
    AND = "and"
    OR = "or"
    NOT = "not"

    # JSON
    JSON_EQ = "json_eq"
    JSON_CONTAINS = "json_contains"


@dataclass
class FilterExpression:
    """A portable filter expression that translates to backend syntax.

    FilterExpression is immutable and composable. Use class methods
    to create expressions, and & (AND), | (OR), ~ (NOT) to combine them.
    """

    operator: FilterOperator
    field: Optional[str] = None
    value: Any = None
    children: Optional[List["FilterExpression"]] = None
    json_key: Optional[str] = None  # For JSON field access

    # =========================================================================
    # Factory Methods for Creating Expressions
    # =========================================================================

    @classmethod
    def eq(cls, field: str, value: Any) -> "FilterExpression":
        """Create equality expression: field == value."""
        return cls(operator=FilterOperator.EQ, field=field, value=value)

    @classmethod
    def ne(cls, field: str, value: Any) -> "FilterExpression":
        """Create inequality expression: field != value."""
        return cls(operator=FilterOperator.NE, field=field, value=value)

    @classmethod
    def gt(cls, field: str, value: Union[int, float]) -> "FilterExpression":
        """Create greater than expression: field > value."""
        return cls(operator=FilterOperator.GT, field=field, value=value)

    @classmethod
    def gte(cls, field: str, value: Union[int, float]) -> "FilterExpression":
        """Create greater than or equal expression: field >= value."""
        return cls(operator=FilterOperator.GTE, field=field, value=value)

    @classmethod
    def lt(cls, field: str, value: Union[int, float]) -> "FilterExpression":
        """Create less than expression: field < value."""
        return cls(operator=FilterOperator.LT, field=field, value=value)

    @classmethod
    def lte(cls, field: str, value: Union[int, float]) -> "FilterExpression":
        """Create less than or equal expression: field <= value."""
        return cls(operator=FilterOperator.LTE, field=field, value=value)

    @classmethod
    def in_list(cls, field: str, values: List[Any]) -> "FilterExpression":
        """Create in-list expression: field IN [values].

        Args:
            field: Field name to match
            values: List of values to match against

        Raises:
            TypeError: If values is not a list
            ValueError: If values list is empty or too large
        """
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"in_list requires a list, got {type(values).__name__}")
        if len(values) == 0:
            raise ValueError("in_list requires at least one value")
        if len(values) > 1000:
            raise ValueError(f"in_list has too many values: {len(values)} (max 1000)")
        return cls(operator=FilterOperator.IN, field=field, value=list(values))

    @classmethod
    def not_in_list(cls, field: str, values: List[Any]) -> "FilterExpression":
        """Create not-in-list expression: field NOT IN [values].

        Args:
            field: Field name to match
            values: List of values to exclude

        Raises:
            TypeError: If values is not a list
            ValueError: If values list is empty or too large
        """
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"not_in_list requires a list, got {type(values).__name__}")
        if len(values) == 0:
            raise ValueError("not_in_list requires at least one value")
        if len(values) > 1000:
            raise ValueError(f"not_in_list has too many values: {len(values)} (max 1000)")
        return cls(operator=FilterOperator.NOT_IN, field=field, value=list(values))

    @classmethod
    def like(cls, field: str, pattern: str) -> "FilterExpression":
        """Create LIKE expression for pattern matching."""
        return cls(operator=FilterOperator.LIKE, field=field, value=pattern)

    @classmethod
    def prefix(cls, field: str, prefix: str) -> "FilterExpression":
        """Create prefix match expression."""
        return cls(operator=FilterOperator.PREFIX, field=field, value=prefix)

    @classmethod
    def json_eq(cls, field: str, json_key: str, value: Any) -> "FilterExpression":
        """Create JSON field equality: field["json_key"] == value.

        Args:
            field: The JSON field name (e.g., "metadata")
            json_key: The key within the JSON (e.g., "session_id")
            value: The value to match

        Raises:
            ValueError: If json_key contains invalid characters

        Example:
            FilterExpression.json_eq("metadata", "session_id", "user123")
            # Milvus: metadata["session_id"] == "user123"
        """
        # Validate json_key to prevent injection via key name
        if not json_key or not isinstance(json_key, str):
            raise ValueError("json_key must be a non-empty string")
        if len(json_key) > 256:
            raise ValueError(f"json_key too long: {len(json_key)} chars (max 256)")
        # Only allow alphanumeric, underscore, hyphen in keys
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_\-]*$", json_key):
            raise ValueError(
                f"Invalid json_key: {json_key!r}. "
                "Must start with letter/underscore and contain only alphanumeric, underscore, hyphen."
            )
        return cls(
            operator=FilterOperator.JSON_EQ,
            field=field,
            json_key=json_key,
            value=value,
        )

    @classmethod
    def json_contains(cls, field: str, json_key: str, value: Any) -> "FilterExpression":
        """Create JSON array contains: field["json_key"] CONTAINS value.

        Args:
            field: The JSON field name (e.g., "metadata")
            json_key: The key within the JSON (e.g., "tags")
            value: The value to check for containment

        Raises:
            ValueError: If json_key contains invalid characters
        """
        if not json_key or not isinstance(json_key, str):
            raise ValueError("json_key must be a non-empty string")
        if len(json_key) > 256:
            raise ValueError(f"json_key too long: {len(json_key)} chars (max 256)")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_\-]*$", json_key):
            raise ValueError(
                f"Invalid json_key: {json_key!r}. "
                "Must start with letter/underscore and contain only alphanumeric, underscore, hyphen."
            )
        return cls(
            operator=FilterOperator.JSON_CONTAINS,
            field=field,
            json_key=json_key,
            value=value,
        )

    @classmethod
    def and_(cls, *expressions: "FilterExpression") -> "FilterExpression":
        """Combine expressions with AND.

        Args:
            *expressions: Variable number of filter expressions

        Example:
            expr = FilterExpression.and_(
                FilterExpression.eq("a", 1),
                FilterExpression.eq("b", 2),
            )
        """
        return cls(
            operator=FilterOperator.AND,
            children=list(expressions),
        )

    @classmethod
    def or_(cls, *expressions: "FilterExpression") -> "FilterExpression":
        """Combine expressions with OR.

        Args:
            *expressions: Variable number of filter expressions

        Example:
            expr = FilterExpression.or_(
                FilterExpression.eq("status", "active"),
                FilterExpression.eq("status", "pending"),
            )
        """
        return cls(
            operator=FilterOperator.OR,
            children=list(expressions),
        )

    @classmethod
    def and_all(cls, expressions: List["FilterExpression"]) -> Optional["FilterExpression"]:
        """Combine a list of expressions with AND.

        This is a convenience method that accepts a list instead of *args,
        making it easier to use with dynamically built filter lists.

        Args:
            expressions: List of filter expressions to combine

        Returns:
            Combined AND expression, or None if list is empty,
            or the single expression if list has one item

        Example:
            filters = []
            if session_id:
                filters.append(FilterExpression.eq("session_id", session_id))
            if doc_ids:
                filters.append(FilterExpression.in_list("document_id", doc_ids))
            expr = FilterExpression.and_all(filters)
        """
        if not expressions:
            return None
        if len(expressions) == 1:
            return expressions[0]
        return cls(
            operator=FilterOperator.AND,
            children=list(expressions),
        )

    @classmethod
    def or_all(cls, expressions: List["FilterExpression"]) -> Optional["FilterExpression"]:
        """Combine a list of expressions with OR.

        This is a convenience method that accepts a list instead of *args,
        making it easier to use with dynamically built filter lists.

        Args:
            expressions: List of filter expressions to combine

        Returns:
            Combined OR expression, or None if list is empty,
            or the single expression if list has one item

        Example:
            status_filters = [
                FilterExpression.eq("status", s)
                for s in ["active", "pending", "review"]
            ]
            expr = FilterExpression.or_all(status_filters)
        """
        if not expressions:
            return None
        if len(expressions) == 1:
            return expressions[0]
        return cls(
            operator=FilterOperator.OR,
            children=list(expressions),
        )

    @classmethod
    def not_(cls, expression: "FilterExpression") -> "FilterExpression":
        """Negate an expression."""
        return cls(
            operator=FilterOperator.NOT,
            children=[expression],
        )

    # =========================================================================
    # Operator Overloads for Composability
    # =========================================================================

    def __and__(self, other: "FilterExpression") -> "FilterExpression":
        """expr1 & expr2 creates AND expression."""
        return FilterExpression.and_(self, other)

    def __or__(self, other: "FilterExpression") -> "FilterExpression":
        """expr1 | expr2 creates OR expression."""
        return FilterExpression.or_(self, other)

    def __invert__(self) -> "FilterExpression":
        """~expr creates NOT expression."""
        return FilterExpression.not_(self)

    # =========================================================================
    # Backend Translation Methods
    # =========================================================================

    def to_milvus(self) -> str:
        """Convert to Milvus filter expression string.

        Returns:
            Milvus-compatible filter string
        """
        return _MilvusTranslator.translate(self)

    def to_qdrant(self) -> Dict[str, Any]:
        """Convert to Qdrant filter dict.

        Returns:
            Qdrant-compatible filter dictionary
        """
        return _QdrantTranslator.translate(self)

    def to_pinecone(self) -> Dict[str, Any]:
        """Convert to Pinecone filter dict.

        Returns:
            Pinecone-compatible filter dictionary
        """
        return _PineconeTranslator.translate(self)

    def to_weaviate(self):
        """Convert to Weaviate v4 Filter object.

        Returns:
            A weaviate ``_Filters`` instance for use with v4 client query/delete methods,
            or None if this expression is empty.
        """
        return _WeaviateTranslator.translate(self)

    def to_opensearch(self) -> Dict[str, Any]:
        """Convert to OpenSearch Query DSL filter dict.

        Returns:
            OpenSearch-compatible bool query dictionary for use in query/filter contexts.
        """
        return _OpenSearchTranslator.translate(self)

    def is_empty(self) -> bool:
        """Check if this is an empty/null filter.

        A logical operator with empty children (e.g. ``OR()``) is NOT empty —
        it carries semantic meaning (OR-of-nothing = match_none).  Only a
        truly uninitialised expression with no operator is considered empty.
        """
        if self.operator in (FilterOperator.AND, FilterOperator.OR, FilterOperator.NOT):
            return False
        return (
            self.field is None
            and self.value is None
            and (self.children is None or len(self.children) == 0)
        )


# =============================================================================
# Backend-Specific Translators
# =============================================================================


class _MilvusTranslator:
    """Translate FilterExpression to Milvus filter syntax."""

    # Operator mapping
    OPS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.GT: ">",
        FilterOperator.GTE: ">=",
        FilterOperator.LT: "<",
        FilterOperator.LTE: "<=",
    }

    @classmethod
    def translate(cls, expr: FilterExpression) -> str:
        """Convert expression to Milvus filter string."""
        if expr.is_empty():
            return ""

        return cls._translate_node(expr)

    @classmethod
    def _translate_node(cls, expr: FilterExpression) -> str:
        """Recursively translate expression tree."""
        op = expr.operator

        # Logical operators
        if op == FilterOperator.AND:
            parts = [cls._translate_node(c) for c in (expr.children or [])]
            parts = [p for p in parts if p]  # Remove empty
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            return "(" + " && ".join(parts) + ")"

        if op == FilterOperator.OR:
            parts = [cls._translate_node(c) for c in (expr.children or [])]
            parts = [p for p in parts if p]
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            return "(" + " || ".join(parts) + ")"

        if op == FilterOperator.NOT:
            if not expr.children:
                return ""
            child = cls._translate_node(expr.children[0])
            if not child:
                return ""
            return f"!({child})"

        # Comparison operators
        if op in cls.OPS:
            field = cls._escape_field(expr.field)
            value = cls._escape_value(expr.value)
            return f"{field} {cls.OPS[op]} {value}"

        # IN operator
        if op == FilterOperator.IN:
            field = cls._escape_field(expr.field)
            values = ", ".join(cls._escape_value(v) for v in expr.value)
            return f"{field} in [{values}]"

        # NOT IN operator
        if op == FilterOperator.NOT_IN:
            field = cls._escape_field(expr.field)
            values = ", ".join(cls._escape_value(v) for v in expr.value)
            return f"{field} not in [{values}]"

        # LIKE operator
        if op == FilterOperator.LIKE:
            field = cls._escape_field(expr.field)
            value = cls._escape_value(expr.value)
            return f"{field} like {value}"

        # JSON operators
        if op == FilterOperator.JSON_EQ:
            field = cls._escape_field(expr.field)
            key = expr.json_key  # Already validated at construction time
            value = cls._escape_value(expr.value)
            return f'{field}["{key}"] == {value}'

        if op == FilterOperator.JSON_CONTAINS:
            field = cls._escape_field(expr.field)
            key = expr.json_key  # Already validated at construction time
            value = cls._escape_value(expr.value)
            return f'array_contains({field}["{key}"], {value})'

        # PREFIX is supported via LIKE in Milvus
        if op == FilterOperator.PREFIX:
            field = cls._escape_field(expr.field)
            value = str(expr.value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "\\%")
            return f'{field} like "{value}%"'

        raise UnsupportedFilterError(op, "milvus")

    @staticmethod
    def _escape_field(field: str) -> str:
        """Escape field name for Milvus."""
        # Milvus field names should be simple identifiers
        if not field or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", field):
            raise ValueError(f"Invalid field name: {field}")
        return field

    @staticmethod
    def _escape_value(value: Any) -> str:
        """Escape value for Milvus filter.

        Handles all edge cases including:
        - None -> null
        - bool -> true/false
        - numbers -> string representation
        - strings -> double-quoted with proper escaping

        Security: Escapes backslash, quotes, newlines, tabs, control characters,
        and unicode line/paragraph separators to prevent injection.
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            # Validate numeric values (prevent inf/nan injection)
            import math

            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ValueError(f"Invalid numeric value: {value}")
            return str(value)
        if isinstance(value, str):
            # Validate string length (prevent DoS via extremely long values)
            if len(value) > 65535:
                raise ValueError(f"Filter value too long: {len(value)} chars (max 65535)")

            # Comprehensive escape sequence handling for injection prevention
            escaped = value
            escaped = escaped.replace("\\", "\\\\")  # Backslash first
            escaped = escaped.replace('"', '\\"')  # Double quotes
            escaped = escaped.replace("\n", "\\n")  # Newlines
            escaped = escaped.replace("\r", "\\r")  # Carriage returns
            escaped = escaped.replace("\t", "\\t")  # Tabs

            # Reject remaining control characters rather than stripping them.
            #
            # These used to be silently removed, which is unsafe in a different
            # way than injection: it changes what the caller asked for. A filter
            # for "a\x01b" would silently become a filter for "ab", so two
            # distinct identifiers collapse to one and a tenant-scoped query can
            # match another tenant's rows. A control character in a filter value
            # is either a bug or an attack; neither deserves a quiet rewrite.
            for ch in value:
                if ord(ch) < 0x20 and ch not in "\n\r\t":
                    raise ValueError(
                        f"Filter value contains control character U+{ord(ch):04X}. "
                        f"Refusing to strip it, because that would silently change "
                        f"which rows the filter matches."
                    )

            # Handle unicode line/paragraph separators
            escaped = escaped.replace("\u2028", "\\u2028")  # Line separator
            escaped = escaped.replace("\u2029", "\\u2029")  # Paragraph separator

            return f'"{escaped}"'
        # For other types, convert to string and escape
        str_val = str(value)
        if len(str_val) > 65535:
            raise ValueError(f"Filter value too long: {len(str_val)} chars (max 65535)")
        str_val = str_val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{str_val}"'


class _QdrantTranslator:
    """Translate FilterExpression to Qdrant filter format."""

    @classmethod
    def translate(cls, expr: FilterExpression) -> Dict[str, Any]:
        """Convert expression to Qdrant filter dict."""
        if expr.is_empty():
            return {}

        return cls._translate_node(expr)

    @classmethod
    def _translate_node(cls, expr: FilterExpression) -> Dict[str, Any]:
        """Recursively translate expression tree."""
        op = expr.operator

        # Logical operators
        if op == FilterOperator.AND:
            must = [cls._translate_node(c) for c in (expr.children or [])]
            must = [m for m in must if m]
            return {"must": must} if must else {}

        if op == FilterOperator.OR:
            should = [cls._translate_node(c) for c in (expr.children or [])]
            should = [s for s in should if s]
            return {"should": should} if should else {}

        if op == FilterOperator.NOT:
            if not expr.children:
                return {}
            child = cls._translate_node(expr.children[0])
            return {"must_not": [child]} if child else {}

        # For Qdrant, all comparisons are in "key" + "match/range" format
        field = expr.field

        # Equality
        if op == FilterOperator.EQ:
            return {"key": field, "match": {"value": expr.value}}

        if op == FilterOperator.NE:
            return {"must_not": [{"key": field, "match": {"value": expr.value}}]}

        # Range operations
        if op == FilterOperator.GT:
            return {"key": field, "range": {"gt": expr.value}}
        if op == FilterOperator.GTE:
            return {"key": field, "range": {"gte": expr.value}}
        if op == FilterOperator.LT:
            return {"key": field, "range": {"lt": expr.value}}
        if op == FilterOperator.LTE:
            return {"key": field, "range": {"lte": expr.value}}

        # IN operator
        if op == FilterOperator.IN:
            return {"key": field, "match": {"any": expr.value}}

        # NOT IN operator
        if op == FilterOperator.NOT_IN:
            return {"must_not": [{"key": field, "match": {"any": expr.value}}]}

        # JSON operations - Qdrant uses dot notation
        if op == FilterOperator.JSON_EQ:
            key = f"{expr.field}.{expr.json_key}"
            return {"key": key, "match": {"value": expr.value}}

        # LIKE operator - Qdrant supports full-text match via "text" condition
        if op == FilterOperator.LIKE:
            return {"key": field, "match": {"text": expr.value}}

        # PREFIX operator - Qdrant doesn't have native prefix, use match text
        if op == FilterOperator.PREFIX:
            return {"key": field, "match": {"text": expr.value}}

        # JSON_CONTAINS - Qdrant uses nested payload dot notation
        if op == FilterOperator.JSON_CONTAINS:
            key = f"{expr.field}.{expr.json_key}"
            return {
                "key": key,
                "match": {"any": [expr.value] if not isinstance(expr.value, list) else expr.value},
            }

        raise UnsupportedFilterError(op, "qdrant")


class _PineconeTranslator:
    """Translate FilterExpression to Pinecone filter format."""

    @classmethod
    def translate(cls, expr: FilterExpression) -> Dict[str, Any]:
        """Convert expression to Pinecone filter dict."""
        if expr.is_empty():
            return {}

        return cls._translate_node(expr)

    @classmethod
    def _translate_node(cls, expr: FilterExpression) -> Dict[str, Any]:
        """Recursively translate expression tree."""
        op = expr.operator

        # Logical operators
        if op == FilterOperator.AND:
            children = [cls._translate_node(c) for c in (expr.children or [])]
            children = [c for c in children if c]
            if not children:
                return {}
            if len(children) == 1:
                return children[0]
            return {"$and": children}

        if op == FilterOperator.OR:
            children = [cls._translate_node(c) for c in (expr.children or [])]
            children = [c for c in children if c]
            if not children:
                return {}
            if len(children) == 1:
                return children[0]
            return {"$or": children}

        if op == FilterOperator.NOT:
            # Pinecone doesn't have direct NOT — rewrite supported cases
            if not expr.children:
                return {}
            child = expr.children[0]
            child_op = child.operator
            # NOT(EQ) → NE
            if child_op == FilterOperator.EQ:
                return {child.field: {"$ne": child.value}}
            # NOT(IN) → NIN
            if child_op == FilterOperator.IN:
                return {child.field: {"$nin": child.value}}
            # NOT(compound) → push NOT into children via De Morgan's law
            if child_op == FilterOperator.AND and child.children:
                # NOT(A AND B) = NOT(A) OR NOT(B)
                negated = [cls._translate_node(FilterExpression.not_(c)) for c in child.children]
                negated = [n for n in negated if n]
                return {"$or": negated} if len(negated) > 1 else (negated[0] if negated else {})
            if child_op == FilterOperator.OR and child.children:
                # NOT(A OR B) = NOT(A) AND NOT(B)
                negated = [cls._translate_node(FilterExpression.not_(c)) for c in child.children]
                negated = [n for n in negated if n]
                return {"$and": negated} if len(negated) > 1 else (negated[0] if negated else {})
            raise UnsupportedFilterError(
                op,
                "pinecone",
                f"NOT({child_op.value}) cannot be translated — rewrite using EQ/NE/IN operators",
            )

        field = expr.field

        # Comparison operators
        if op == FilterOperator.EQ:
            return {field: {"$eq": expr.value}}
        if op == FilterOperator.NE:
            return {field: {"$ne": expr.value}}
        if op == FilterOperator.GT:
            return {field: {"$gt": expr.value}}
        if op == FilterOperator.GTE:
            return {field: {"$gte": expr.value}}
        if op == FilterOperator.LT:
            return {field: {"$lt": expr.value}}
        if op == FilterOperator.LTE:
            return {field: {"$lte": expr.value}}

        # IN operator
        if op == FilterOperator.IN:
            return {field: {"$in": expr.value}}
        if op == FilterOperator.NOT_IN:
            return {field: {"$nin": expr.value}}

        raise UnsupportedFilterError(op, "pinecone")


class _WeaviateTranslator:
    """Translate FilterExpression to Weaviate v4 Filter objects.

    Uses the weaviate-client v4 ``Filter.by_property()`` API to produce
    proper ``_Filters`` instances (``_FilterValue``, ``_FilterAnd``, etc.)
    that the v4 query/delete methods expect.
    """

    # Lazy-loaded reference to weaviate Filter class
    _Filter = None

    @classmethod
    def _get_filter_class(cls):
        """Lazy-load weaviate Filter to avoid import errors when not installed."""
        if cls._Filter is None:
            from weaviate.collections.classes.filters import Filter

            cls._Filter = Filter
        return cls._Filter

    @classmethod
    def translate(cls, expr: FilterExpression):
        """Convert expression to Weaviate v4 Filter object.

        Returns:
            A weaviate ``_Filters`` instance, or None if the expression is empty.
        """
        if expr.is_empty():
            return None

        return cls._translate_node(expr)

    @classmethod
    def _translate_node(cls, expr: FilterExpression):
        """Recursively translate expression tree to v4 Filter objects."""
        F = cls._get_filter_class()
        op = expr.operator

        # Logical operators
        if op == FilterOperator.AND:
            operands = [cls._translate_node(c) for c in (expr.children or [])]
            operands = [o for o in operands if o is not None]
            if not operands:
                return None
            if len(operands) == 1:
                return operands[0]
            return F.all_of(operands)

        if op == FilterOperator.OR:
            operands = [cls._translate_node(c) for c in (expr.children or [])]
            operands = [o for o in operands if o is not None]
            if not operands:
                return None
            if len(operands) == 1:
                return operands[0]
            return F.any_of(operands)

        if op == FilterOperator.NOT:
            if not expr.children:
                return None
            child = cls._translate_node(expr.children[0])
            return ~child if child is not None else None

        field = expr.field
        value = expr.value
        prop = F.by_property(field)

        # Comparison operators
        if op == FilterOperator.EQ:
            return prop.equal(value)
        if op == FilterOperator.NE:
            return prop.not_equal(value)
        if op == FilterOperator.GT:
            return prop.greater_than(value)
        if op == FilterOperator.GTE:
            return prop.greater_or_equal(value)
        if op == FilterOperator.LT:
            return prop.less_than(value)
        if op == FilterOperator.LTE:
            return prop.less_or_equal(value)
        if op == FilterOperator.LIKE:
            return prop.like(value)

        # IN operator — Weaviate uses contains_any for lists
        if op == FilterOperator.IN:
            values = list(expr.value) if expr.value else []
            if not values:
                return None
            return prop.contains_any(values)

        # NOT IN operator
        if op == FilterOperator.NOT_IN:
            values = list(expr.value) if expr.value else []
            if not values:
                return None
            return ~prop.contains_any(values)

        # PREFIX operator — Weaviate supports Like with wildcard
        if op == FilterOperator.PREFIX:
            return prop.like(f"{expr.value}*")

        # JSON operators — Weaviate doesn't support nested JSON queries natively
        if op in (FilterOperator.JSON_EQ, FilterOperator.JSON_CONTAINS):
            raise UnsupportedFilterError(
                op,
                "weaviate",
                "Weaviate does not support nested JSON field queries. "
                "Store fields as top-level properties instead.",
            )

        raise UnsupportedFilterError(op, "weaviate")


class _OpenSearchTranslator:
    """Translate FilterExpression to OpenSearch Query DSL.

    Produces ``bool`` query structures that work in both ``query`` and
    ``filter`` contexts. Uses ``term``/``terms`` for exact matching on
    keyword fields and ``range`` for numeric comparisons.

    Security: Field names are validated with the same regex used by other
    translators. Values are passed as parameters (not string-interpolated),
    so injection is not possible in the Query DSL dict format.
    """

    # Dots intentionally excluded: they traverse nested objects in OpenSearch.
    # JSON path access (e.g. metadata.session_id) is only allowed via the
    # explicit JSON_EQ/JSON_CONTAINS operators which construct the path safely.
    _FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\-]*$")

    @classmethod
    def translate(cls, expr: FilterExpression) -> Dict[str, Any]:
        """Convert expression tree to OpenSearch Query DSL dict."""
        if expr.is_empty():
            return {"match_all": {}}
        return cls._translate_node(expr)

    @classmethod
    def _translate_node(cls, expr: FilterExpression) -> Dict[str, Any]:
        """Recursively translate expression tree."""
        op = expr.operator

        # --- Logical operators ---
        if op == FilterOperator.AND:
            parts = [cls._translate_node(c) for c in (expr.children or [])]
            parts = [p for p in parts if p]
            if not parts:
                return {"match_all": {}}
            if len(parts) == 1:
                return parts[0]
            return {"bool": {"must": parts}}

        if op == FilterOperator.OR:
            parts = [cls._translate_node(c) for c in (expr.children or [])]
            parts = [p for p in parts if p]
            if not parts:
                # OR of nothing = False (identity element) = match nothing.
                return {"match_none": {}}
            if len(parts) == 1:
                return parts[0]
            return {"bool": {"should": parts, "minimum_should_match": 1}}

        if op == FilterOperator.NOT:
            if not expr.children:
                return {"match_all": {}}
            child = cls._translate_node(expr.children[0])
            # NOT(match_all) = NOT(everything) = match nothing.
            if child == {"match_all": {}}:
                return {"match_none": {}}
            # NOT(match_none) = NOT(nothing) = match everything.
            if child == {"match_none": {}}:
                return {"match_all": {}}
            return {"bool": {"must_not": [child]}}

        # --- Validate field ---
        field = expr.field or ""
        cls._validate_field(field)
        value = expr.value

        # --- Comparison operators ---
        if op == FilterOperator.EQ:
            return {"term": {field: {"value": value}}}

        if op == FilterOperator.NE:
            return {"bool": {"must_not": [{"term": {field: {"value": value}}}]}}

        if op == FilterOperator.GT:
            return {"range": {field: {"gt": value}}}

        if op == FilterOperator.GTE:
            return {"range": {field: {"gte": value}}}

        if op == FilterOperator.LT:
            return {"range": {field: {"lt": value}}}

        if op == FilterOperator.LTE:
            return {"range": {field: {"lte": value}}}

        # --- Collection operators ---
        if op == FilterOperator.IN:
            values = list(value) if value else []
            if not values:
                return {"match_none": {}}
            return {"terms": {field: values}}

        if op == FilterOperator.NOT_IN:
            values = list(value) if value else []
            if not values:
                return {"match_all": {}}
            return {"bool": {"must_not": [{"terms": {field: values}}]}}

        # --- String operators ---
        if op == FilterOperator.LIKE:
            # Translate SQL-style LIKE wildcards to OpenSearch wildcard syntax.
            # Must handle SQL LIKE escape sequences: \% = literal %,
            # \_ = literal _, \\ = literal backslash.
            pattern = str(value)
            result: List[str] = []
            i = 0
            while i < len(pattern):
                ch = pattern[i]
                if ch == "\\" and i + 1 < len(pattern):
                    next_ch = pattern[i + 1]
                    if next_ch == "%":
                        result.append("%")  # SQL \% -> literal %
                        i += 2
                        continue
                    elif next_ch == "_":
                        result.append("_")  # SQL \_ -> literal _
                        i += 2
                        continue
                    elif next_ch == "\\":
                        result.append("\\\\")  # SQL \\ -> OpenSearch literal \
                        i += 2
                        continue
                    else:
                        result.append("\\\\")  # Stray backslash -> escape for OS
                        i += 1
                        continue
                elif ch == "*":
                    result.append("\\*")  # Literal * in OpenSearch
                elif ch == "?":
                    result.append("\\?")  # Literal ? in OpenSearch
                elif ch == "%":
                    result.append("*")  # SQL % -> OpenSearch *
                elif ch == "_":
                    result.append("?")  # SQL _ -> OpenSearch ?
                else:
                    result.append(ch)
                i += 1
            os_pattern = "".join(result)
            return {"wildcard": {field: {"value": os_pattern}}}

        if op == FilterOperator.PREFIX:
            return {"prefix": {field: {"value": value}}}

        # --- JSON operators (dot notation access) ---
        if op == FilterOperator.JSON_EQ:
            json_key = expr.json_key or ""
            cls._validate_field(json_key)
            return {"term": {f"{field}.{json_key}": {"value": value}}}

        if op == FilterOperator.JSON_CONTAINS:
            json_key = expr.json_key or ""
            cls._validate_field(json_key)
            # Use term query for array containment (checks if array field
            # contains the exact value). OpenSearch handles multi-valued
            # fields automatically with term queries.
            return {"term": {f"{field}.{json_key}": {"value": value}}}

        raise UnsupportedFilterError(op, "opensearch")

    @classmethod
    def _validate_field(cls, field: str) -> None:
        """Validate field name to prevent injection."""
        if not field or not cls._FIELD_RE.match(field):
            raise ValueError(
                f"Invalid field name for OpenSearch: '{field}'. "
                "Must match [a-zA-Z_][a-zA-Z0-9_\\-]* (dots not allowed — "
                "use JSON_EQ/JSON_CONTAINS for nested field access)"
            )


# =============================================================================
# Utility Functions
# =============================================================================


def parse_legacy_milvus_filter(filter_str: str) -> Optional[FilterExpression]:
    """Parse a legacy Milvus filter string to FilterExpression.

    This enables gradual migration from raw Milvus strings to
    the portable FilterExpression format.

    Supports:
    - Simple equality: field == "value"
    - JSON equality: field["key"] == "value"
    - IN lists: field in ["val1", "val2"]
    - AND expressions: expr1 && expr2
    - OR expressions: expr1 || expr2

    Args:
        filter_str: Milvus filter string (e.g., 'session_id == "foo"')

    Returns:
        FilterExpression or None if parsing fails
    """
    if not filter_str or not filter_str.strip():
        return None

    filter_str = filter_str.strip()

    # Handle AND expressions (recursive)
    if " && " in filter_str:
        # Split on && but respect parentheses
        parts = _split_logical_expr(filter_str, "&&")
        if len(parts) > 1:
            sub_exprs = [parse_legacy_milvus_filter(p) for p in parts]
            sub_exprs = [e for e in sub_exprs if e is not None]
            if sub_exprs:
                return FilterExpression.and_(*sub_exprs) if len(sub_exprs) > 1 else sub_exprs[0]

    # Handle OR expressions (recursive)
    if " || " in filter_str:
        parts = _split_logical_expr(filter_str, "||")
        if len(parts) > 1:
            sub_exprs = [parse_legacy_milvus_filter(p) for p in parts]
            sub_exprs = [e for e in sub_exprs if e is not None]
            if sub_exprs:
                return FilterExpression.or_(*sub_exprs) if len(sub_exprs) > 1 else sub_exprs[0]

    # Strip outer parentheses
    if filter_str.startswith("(") and filter_str.endswith(")"):
        inner = filter_str[1:-1].strip()
        # Verify balanced
        if inner.count("(") == inner.count(")"):
            return parse_legacy_milvus_filter(inner)

    # IN list: field in ["val1", "val2", ...]
    in_match = re.match(r"^(\w+)\s+in\s+\[([^\]]*)\]$", filter_str, re.IGNORECASE)
    if in_match:
        field, values_str = in_match.groups()
        # Parse the values list
        values = _parse_value_list(values_str)
        if values:
            return FilterExpression.in_list(field, values)

    # Simple equality: field == "value"
    eq_match = re.match(r'^(\w+)\s*==\s*"([^"]*)"$', filter_str)
    if eq_match:
        field, value = eq_match.groups()
        return FilterExpression.eq(field, value)

    # Numeric equality: field == 123
    num_eq_match = re.match(r"^(\w+)\s*==\s*(-?\d+(?:\.\d+)?)$", filter_str)
    if num_eq_match:
        field, value = num_eq_match.groups()
        return FilterExpression.eq(field, float(value) if "." in value else int(value))

    # JSON equality: field["key"] == "value"
    json_eq_match = re.match(r'^(\w+)\["([^"]+)"\]\s*==\s*"([^"]*)"$', filter_str)
    if json_eq_match:
        field, key, value = json_eq_match.groups()
        return FilterExpression.json_eq(field, key, value)

    # Comparison operators: >, >=, <, <=
    cmp_match = re.match(r"^(\w+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)$", filter_str)
    if cmp_match:
        field, op, value = cmp_match.groups()
        num_val = float(value) if "." in value else int(value)
        op_map = {
            ">": FilterExpression.gt,
            ">=": FilterExpression.gte,
            "<": FilterExpression.lt,
            "<=": FilterExpression.lte,
        }
        return op_map[op](field, num_val)

    # Fallback: return None for unsupported expressions
    return None


def _split_logical_expr(expr: str, operator: str) -> List[str]:
    """Split expression by logical operator respecting parentheses."""
    parts = []
    current = ""
    depth = 0

    i = 0
    while i < len(expr):
        c = expr[i]
        if c == "(":
            depth += 1
            current += c
        elif c == ")":
            depth -= 1
            current += c
        elif depth == 0 and expr[i : i + len(operator) + 2] == f" {operator} ":
            if current.strip():
                parts.append(current.strip())
            current = ""
            i += len(operator) + 1  # Skip the operator and spaces
        else:
            current += c
        i += 1

    if current.strip():
        parts.append(current.strip())

    return parts


def _parse_value_list(values_str: str) -> List[Any]:
    """Parse a comma-separated value list from Milvus IN clause."""
    values = []
    # Match quoted strings and numbers
    pattern = r'"([^"]*)"|(-?\d+(?:\.\d+)?)'
    for match in re.finditer(pattern, values_str):
        if match.group(1) is not None:
            values.append(match.group(1))
        elif match.group(2) is not None:
            val = match.group(2)
            values.append(float(val) if "." in val else int(val))
    return values
