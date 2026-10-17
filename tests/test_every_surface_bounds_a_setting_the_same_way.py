"""A numeric setting has one valid range and one session maximum, on every surface.

Four tables bounded the same settings: the settings check that environment
variables and the settings file meet (``_VALID_RANGES``), the configurator
schema that every session write and every arm is validated against, the
largest value a session may set (``SESSION_MAXIMUMS``), and the REST patch
model's own ``Field`` bounds. They disagreed: REST took ``chunk_size`` from 50
where arms and the settings file take 1, ``default_top_k`` up to 100 where a
session may set 1000, and a negative ``similarity_threshold`` that the
validator then refused. The REST model now takes its bounds from the others.
"""

from __future__ import annotations

import pytest

from lasagna.api.rest.models import ConfigPatchRequest
from lasagna.configurator.schema import PARAMETERS
from lasagna.configurator.validation import validate_config
from lasagna.core.config import _VALID_RANGES, _check_bounds
from lasagna.core.session_config._session_keys import SESSION_MAXIMUMS

#: REST's flat names for nested settings (routes/config.py, _DA_FLAT_TO_NESTED).
_NESTED = {"first_tokens_count": "document_awareness.first_tokens_count"}
_SCHEMA = {p.key: p for p in PARAMETERS}


def _rest_bounds():
    for name, field in ConfigPatchRequest.model_fields.items():
        ge = le = None
        for meta in field.metadata:
            ge = getattr(meta, "ge", ge)
            le = getattr(meta, "le", le)
        if ge is not None or le is not None:
            yield _NESTED.get(name, name), ge, le


REST = list(_rest_bounds())


def _validity(key):
    """(lowest, highest) valid value: the settings check's, else the schema's."""
    if key in _VALID_RANGES:
        return _VALID_RANGES[key]
    p = _SCHEMA[key]
    return p.min_value, p.max_value


@pytest.mark.parametrize("key,ge,le", REST, ids=[k for k, _, _ in REST])
def test_rest_takes_the_valid_minimum_and_the_session_maximum(key, ge, le):
    low, high = _validity(key)
    assert ge == low, f"{key}: REST from {ge}, valid from {low}"
    assert le == SESSION_MAXIMUMS[key], f"{key}: REST to {le}, a session to {SESSION_MAXIMUMS[key]}"
    if high is not None:
        assert le <= high, f"{key}: a session may set {le}, past the valid {high}"


@pytest.mark.parametrize("key,ge,le", REST, ids=[k for k, _, _ in REST])
def test_what_rest_accepts_the_validator_and_the_settings_check_accept(key, ge, le):
    parent, _, child = key.rpartition(".")
    for value in (ge, le):
        config = {parent: {child: value}} if parent else {key: value}
        # Range errors only: a cross-field rule (min_chunk_size below
        # max_chunk_size) depends on the other value, not on the bound.
        errors = [
            e
            for e in validate_config(config).errors
            if e["key"] == key and ("minimum" in e["message"] or "maximum" in e["message"])
        ]
        assert not errors, errors
        if not parent:
            _check_bounds(key, value)  # raises ConfigError when outside


def test_the_settings_check_and_the_schema_agree_where_both_bound_a_key():
    for key, (low, high) in _VALID_RANGES.items():
        p = _SCHEMA.get(key)
        if p is None or p.min_value is None:
            continue
        assert (low, high) == (p.min_value, p.max_value), key


def test_a_session_maximum_is_never_past_the_valid_range():
    for key, cap in SESSION_MAXIMUMS.items():
        if key in _VALID_RANGES:
            low, high = _VALID_RANGES[key]
            assert low <= cap <= high, (key, cap, (low, high))
