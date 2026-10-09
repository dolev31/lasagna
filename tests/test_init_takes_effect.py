"""What the setup wizard promises has to actually happen.

`lasagna init` asked for a vector backend, an embedding mode and a quality
profile, wrote `settings.lasagna.toml` into the working directory, and printed
"Setup complete!" naming the choices. None of it applied:

* the config loader reads the *installed package* directory or
  `$RAG_SETTINGS_FILE`, never the working directory, so the file was inert; and
* the profile was never written to the file at all — it was collected,
  displayed in the summary, and discarded.

So a user picked "Speed — pure vector search, fastest", was told that is what
they had, and downloaded 5.4 GB on the next command. The wizard is the first
thing the README tells a new user to run, which makes this the worst place in
the package for a silent no-op.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "src" / "lasagna" / "cli.py"


class TestTheLoaderHonoursTheFileInitWrites:
    """The behavioural half: a settings file at RAG_SETTINGS_FILE must win."""

    def test_a_pointed_settings_file_changes_the_resolved_model(self, tmp_path):
        settings = tmp_path / "settings.lasagna.toml"
        settings.write_text(
            '[rag]\nvector_backend = "milvus"\n'
            'embedding_model = "all-MiniLM-L6-v2"\nembedding_dim = 384\n'
        )
        probe = (
            "from lasagna.core.config import rag_config;"
            "print(rag_config.embedding_model, rag_config.embedding_dim)"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
                "PYTHONPATH": str(REPO / "src"),
                "RAG_SETTINGS_FILE": str(settings),
            },
        )
        assert result.returncode == 0, result.stderr[-500:]
        assert "all-MiniLM-L6-v2 384" in result.stdout, (
            f"RAG_SETTINGS_FILE was ignored, so anything init writes is inert: {result.stdout!r}"
        )


class TestInitWiresItUp:
    """The wiring half: init must point at the file and apply the profile.

    Source-level rather than behavioural because `init` is an interactive
    questionary wizard. Weaker than running it, and still enough to catch the
    wiring being removed — which is exactly how it was broken.
    """

    @pytest.fixture(scope="class")
    def init_source(self) -> str:
        text = CLI.read_text(encoding="utf-8")
        start = text.index("def init(")
        end = text.index("\ndef ", start + 1)
        return text[start:end]

    def test_it_exports_rag_settings_file(self, init_source):
        assert "RAG_SETTINGS_FILE" in init_source, (
            "init writes a settings file the loader never reads unless it also "
            "points RAG_SETTINGS_FILE at it"
        )

    def test_it_writes_an_absolute_path(self, init_source):
        """A relative path breaks the moment the user changes directory."""
        assert ".resolve()" in init_source, "the settings path should be absolute"

    def test_it_applies_the_chosen_profile(self, init_source):
        assert "set_active_profile" in init_source, (
            "the profile is collected and shown in the summary; it must also be applied"
        )

    def test_failures_are_reported_not_swallowed(self, init_source):
        """A wizard that silently fails to apply a choice is the original bug."""
        for marker in ("could not set RAG_SETTINGS_FILE", "could not activate"):
            assert marker in init_source, f"missing a warning for: {marker}"


class TestTheProfileSwitchReachesSearch:
    """`set_active_profile` only helps if it reaches the search path.

    It did not until the session-config write-through was fixed, so this asserts
    the mechanism init now depends on, not just that init calls it.
    """

    def test_activating_speed_disables_reranking(self):
        from lasagna.core.config import rag_config
        from lasagna.core.session_config import get_knowledge_config_manager

        session = "init-profile-probe"
        get_knowledge_config_manager(session_id=session).set_active_profile(
            "speed", session_id=session
        )
        assert rag_config.with_session(session).enable_reranking is False


class TestTheEmbeddingModelIsConfigurableFromTheSDK:
    """`KnowledgeClient(config=KnowledgeConfig(embedding_model=...))` did nothing.

    `embedding_model`, `embedding_dim` and their neighbours were mapped only as
    RUNTIME fields — written as per-session overrides after the backend was
    already constructed. But the `EmbeddingManager` is built at backend init
    from global config, so a session override could not change a model already
    loaded. The SDK's primary configuration surface silently did nothing for the
    one setting that costs 5.4 GB to get wrong, and the README carried it as a
    known limitation with "use the environment instead" as the workaround.

    They are init-time fields now, so the bridge writes them to `rag_config`
    before the backend exists. They stay in RUNTIME_FIELD_MAP as well: the
    evaluation harness sweeps embedding models per config, pairing each with its
    own `collection_name`, and that path still needs the session override.
    """

    def test_the_embedding_fields_are_init_time(self):
        from lasagna.sdk._config_bridge import INIT_TIME_FIELDS

        for field in ("embedding_model", "embedding_dim", "embedding_provider"):
            assert field in INIT_TIME_FIELDS, (
                f"{field} is applied after the EmbeddingManager is constructed, "
                f"so setting it from the SDK cannot take effect"
            )

    def test_a_non_default_model_is_bridged(self):
        from lasagna.sdk._config_bridge import get_init_overrides
        from lasagna.sdk.config import KnowledgeConfig

        overrides = get_init_overrides(
            KnowledgeConfig(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dim=384,
            )
        )
        assert overrides["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert overrides["embedding_dim"] == 384

    def test_the_default_is_not_bridged(self):
        """Only non-defaults are written, so a client that sets nothing does not
        mask a profile or TOML setting."""
        from lasagna.sdk._config_bridge import get_init_overrides
        from lasagna.sdk.config import KnowledgeConfig

        assert "embedding_model" not in get_init_overrides(KnowledgeConfig())

    def test_it_reaches_the_global_config(self, monkeypatch):
        """End to end through the bridge, which is where it broke."""
        from lasagna.core.config import rag_config
        from lasagna.sdk._config_bridge import apply_init_overrides
        from lasagna.sdk.config import KnowledgeConfig

        original = dict(rag_config._env_overrides)
        try:
            apply_init_overrides(
                rag_config._env_overrides,
                KnowledgeConfig(
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    embedding_dim=384,
                ),
            )
            assert rag_config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
            assert rag_config.embedding_dim == 384
        finally:
            rag_config._env_overrides.clear()
            rag_config._env_overrides.update(original)

    def test_the_backend_passes_its_config_to_the_factory(self):
        """`get_embedding_manager()` with no argument cannot detect config
        drift, so a second init after a different model was bridged would keep
        the first manager."""
        import inspect

        from lasagna.api import rag_backend

        source = inspect.getsource(rag_backend.RAGBackend._do_init)
        assert "get_embedding_manager(rag_config)" in source
