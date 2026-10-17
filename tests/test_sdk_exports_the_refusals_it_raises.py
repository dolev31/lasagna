"""The refusals an SDK call raises import from ``lasagna.sdk`` and are ``KnowledgeError``s.

``set_config`` and the profile switch raised ``DocumentsLeftBehind``,
``ProfileCannotLoad`` and the rest, which lived only in ``lasagna.api`` modules
and subclassed ``ValueError``: an application catching ``KnowledgeError``, as the
SDK documents, missed every one. They keep their old bases, so code that caught
``ValueError`` or ``RuntimeError`` still does.
"""

from __future__ import annotations

import pytest

import lasagna.sdk as sdk

REFUSALS = {
    "EmbeddingSizeConflict": ValueError,
    "DocumentsLeftBehind": ValueError,
    "ProfileSwitchInProgress": ValueError,
    "ProfileCannotLoad": ValueError,
    "UnknownProfileError": ValueError,
    "ReindexFailed": RuntimeError,
}


@pytest.mark.parametrize("name, old_base", sorted(REFUSALS.items()))
def test_each_refusal_is_exported_and_a_knowledge_error(name, old_base):
    assert name in sdk.__all__
    refusal = getattr(sdk, name)
    assert issubclass(refusal, sdk.KnowledgeError)
    assert issubclass(refusal, old_base)


def test_the_exported_classes_are_the_ones_raised():
    from lasagna.api import _profile_guard, _profile_switch

    assert sdk.DocumentsLeftBehind is _profile_guard.DocumentsLeftBehind
    assert sdk.ProfileCannotLoad is _profile_switch.ProfileCannotLoad
    assert str(sdk.ReindexFailed("why", report=None)) == "why"
