"""SDK enum members print as their value on every supported Python.

A ``(str, Enum)`` member formatted as ``ready`` on Python 3.10 and as
``DocumentStatus.READY`` on 3.11 and later, so ``lasagna upload`` printed
``Status: DocumentStatus.READY`` or ``Status: ready`` depending on the
interpreter, and ``str(member)`` differed the same way.
"""

from __future__ import annotations

import enum

import pytest

from lasagna.sdk import config_validation, types

_ENUMS = [
    obj
    for module in (types, config_validation)
    for obj in vars(module).values()
    if isinstance(obj, type) and issubclass(obj, enum.Enum) and issubclass(obj, str) and len(obj)
]


def test_every_sdk_string_enum_is_covered():
    assert {e.__name__ for e in _ENUMS} >= {"DocumentStatus", "ChunkingStrategy", "Severity"}


@pytest.mark.parametrize("enum_cls", _ENUMS, ids=lambda e: e.__name__)
def test_members_print_as_their_value(enum_cls):
    for member in enum_cls:
        assert str(member) == member.value
        assert f"{member}" == member.value
        assert f"{member:>12}" == f"{member.value:>12}"
        assert member == member.value  # still a str
