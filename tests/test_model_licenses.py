"""Tests for the model-weight license registry.

Lasagna is research tooling, so non-commercially-licensed weights are
legitimate to offer -- comparing them is the point.  What must never happen is
a user picking one *without realising*, then carrying that configuration into
production.  These tests protect the alerting path and, more importantly, stop
the license table from silently drifting out of sync with the models the UI
actually offers.
"""

import re


from lasagna.core.model_licenses import (
    MODEL_LICENSES,
    describe,
    is_commercial_use_restricted,
    restricted_models,
    warn_if_restricted,
)
from tests._paths import PACKAGE_DIR


class TestRegistryContents:
    def test_known_restricted_models_are_flagged(self):
        """The models whose terms do not allow commercial use must be marked as such."""
        expected = {
            "naver/splade-cocondenser-ensembledistil",
            "jinaai/jina-reranker-v2-base-multilingual",
            "BAAI/bge-reranker-v2-gemma",
            "castorini/monot5-base-msmarco-10k",
        }
        assert expected <= set(restricted_models())

    def test_permissive_models_are_not_flagged(self):
        for model in (
            "BAAI/bge-m3",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-large-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "mixedbread-ai/mxbai-rerank-large-v1",
        ):
            assert not is_commercial_use_restricted(model), f"{model} wrongly restricted"

    def test_every_entry_links_to_its_model_card(self):
        """A license claim is only useful if the user can verify it."""
        for name, info in MODEL_LICENSES.items():
            assert info.url.startswith("https://"), f"{name} has no verifiable source"
            assert info.spdx, f"{name} has no license identifier"

    def test_restricted_entries_explain_themselves(self):
        """A bare SPDX id is not enough to act on; require a note."""
        for name, info in restricted_models().items():
            assert info.note, f"{name} is restricted but offers no explanation"

    def test_unknown_model_is_not_silently_treated_as_permissive(self):
        assert describe("some-vendor/never-heard-of-it") is None
        # Unknown must not be reported as "restricted" either -- it is unknown.
        assert not is_commercial_use_restricted("some-vendor/never-heard-of-it")

    def test_lookup_is_case_insensitive(self):
        assert describe("baai/BGE-M3") is not None


class TestRegistryCoversWhatWeOffer:
    """The table must cover every model a user can actually select."""

    def _models_in(self, path, pattern):
        text = (PACKAGE_DIR / path).read_text(encoding="utf-8", errors="ignore")
        return set(re.findall(pattern, text))

    def test_all_configurator_models_are_catalogued(self):
        """Every model in the configuration UI needs license metadata.

        Without this, adding a dropdown entry silently ships a model whose
        terms nobody has checked.
        """
        offered = self._models_in(
            "configurator/schema.py",
            r'EnumOption\("([A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+)"',
        )
        assert offered, "expected the configurator to offer model choices"
        missing = sorted(m for m in offered if describe(m) is None)
        assert not missing, (
            "Models offered in the configurator with no license entry in "
            f"lasagna/core/model_licenses.py: {missing}"
        )

    def test_default_models_in_settings_are_catalogued(self):
        """Defaults shipped in settings.lasagna.toml need metadata too."""
        offered = self._models_in(
            "settings.lasagna.toml",
            r'"([A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+)"',
        )
        # Only assert on identifiers that look like HF model repos.
        candidates = {
            m
            for m in offered
            if any(
                m.startswith(v)
                for v in (
                    "BAAI/",
                    "intfloat/",
                    "cross-encoder/",
                    "naver/",
                    "jinaai/",
                    "mixedbread-ai/",
                    "colbert-ir/",
                    "sentence-transformers/",
                )
            )
        }
        missing = sorted(m for m in candidates if describe(m) is None)
        assert not missing, f"Uncatalogued models in settings.lasagna.toml: {missing}"


class TestEveryDefaultModelIsCatalogued:
    """The model every built-in component loads by default, and every profile's.

    monoT5's default (castorini/monot5-base-msmarco-10k) was "not in Lasagna's
    license table" at DEBUG, as were FlashRank's and the speed profile's
    embedder named without its organization, while CONTRIBUTING said this test
    fails for a selectable model with no entry.
    """

    def _defaults(self):
        from lasagna.core.config import get_rag_config
        from lasagna.core.session_config._profiles import ProfileMixin
        from lasagna.core.setup_text import COMPONENT_MODEL_SETTINGS
        from lasagna.knowledge_evaluation_sdk.answer_metrics import DEFAULT_SIMILARITY_MODEL

        config = get_rag_config()
        settings = set(COMPONENT_MODEL_SETTINGS.values()) | {"embedding_model_fallback"}
        models = {}
        for setting in sorted(settings):
            models[config._config[setting]] = f"default {setting}"
            for profile, spec in ProfileMixin._PROFILE_SPECS.items():
                if spec.get(setting):
                    models[spec[setting]] = f"{profile} {setting}"
        models["BAAI/bge-m3"] = "the bge_m3 embedder"
        models[DEFAULT_SIMILARITY_MODEL] = "answer_similarity"
        return models

    def test_every_default_model_has_a_license_entry(self):
        missing = {m: where for m, where in self._defaults().items() if describe(m) is None}
        assert not missing, f"Default models with no entry in core/model_licenses.py: {missing}"

    def test_every_model_setting_names_its_component(self):
        """A new ``*_model`` setting is a new default to catalogue."""
        from lasagna.core.config import get_rag_config
        from lasagna.core.setup_text import COMPONENT_MODEL_SETTINGS

        model_settings = {
            key
            for key, value in get_rag_config()._config.items()
            if key.endswith("_model") and isinstance(value, str) and value
        }
        covered = set(COMPONENT_MODEL_SETTINGS.values())
        # No weights Lasagna downloads for a component: the fallback embedder
        # (catalogued above), LLMs, and Cohere's hosted reranker.
        known_elsewhere = {
            "embedding_model_fallback",
            "graph_extraction_model",
            "hyde_model",
            "cohere_rerank_model",
        }
        assert model_settings - covered - known_elsewhere == set()

    def test_a_bare_sentence_transformers_name_is_found(self):
        assert describe("all-MiniLM-L6-v2") is describe("sentence-transformers/all-MiniLM-L6-v2")


class TestAlerting:
    def test_restricted_model_emits_warning(self, caplog):
        warn_if_restricted.__globals__["_ANNOUNCED"].clear()
        from loguru import logger

        seen = []
        sink = logger.add(lambda m: seen.append(m), level="WARNING")
        try:
            warn_if_restricted("naver/splade-cocondenser-ensembledistil", component="SPLADE")
        finally:
            logger.remove(sink)

        assert seen, "restricted model must warn"
        message = "".join(seen)
        assert "NOT for commercial use" in message
        assert "CC-BY-NC-SA-4.0" in message
        assert "SPLADE" in message

    def test_warning_is_emitted_once_per_model(self):
        warn_if_restricted.__globals__["_ANNOUNCED"].clear()
        from loguru import logger

        seen = []
        sink = logger.add(lambda m: seen.append(m), level="WARNING")
        try:
            for _ in range(5):
                warn_if_restricted("BAAI/bge-reranker-v2-gemma", component="reranker")
        finally:
            logger.remove(sink)

        assert len(seen) == 1, f"expected exactly one warning, got {len(seen)}"

    def test_permissive_model_does_not_warn(self):
        warn_if_restricted.__globals__["_ANNOUNCED"].clear()
        from loguru import logger

        seen = []
        sink = logger.add(lambda m: seen.append(m), level="WARNING")
        try:
            warn_if_restricted("BAAI/bge-m3", component="embedding model")
        finally:
            logger.remove(sink)

        assert not seen, "permissive model must not warn"

    def test_alerting_never_raises(self):
        """Licence advice must never break retrieval."""
        for bad in (None, "", 123, object()):
            warn_if_restricted(bad, component="x")  # must not raise
