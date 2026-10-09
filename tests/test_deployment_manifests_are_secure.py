"""The shipped manifests must configure what the code now enforces.

Nothing in the suite read these files, so they drifted from the code they
deploy. Three defects, all verified before the fix:

* The k8s ConfigMap did not set ``RAG_REQUIRE_AUTH``, so ``X-Session-ID`` was
  accepted as identity by a Service reachable from the whole cluster.
* Neither the ConfigMap nor the Secret set ``RAG_ADMIN_TOKEN``, so all six REST
  admin routes returned 503 permanently — the manifests deployed a service whose
  GC and bulk-delete controls could never be used.
* ``docker-compose.yml`` restricted every service to ``127.0.0.1`` except the
  API, which it published unqualified. Docker writes its own iptables DNAT
  rules, so that reaches the LAN past most host firewalls, and the API is the
  one service in the file that can delete documents.

The bundled ``Dockerfile`` runs gunicorn against the ASGI app rather than
``lasagna serve``, so the CLI's non-loopback refusal never fires in the
deployment that most needs it. That is why the manifests have to be right on
their own, and why these assertions are about files rather than behaviour.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

DEPLOY = pathlib.Path(__file__).resolve().parents[1] / "src" / "lasagna" / "deploy"


def _load(relative):
    path = DEPLOY / relative
    if not path.exists():
        pytest.skip(f"{relative} is not present in this checkout")
    return yaml.safe_load(path.read_text())


class TestKubernetes:
    def test_auth_is_required(self):
        """A Service is reachable by everything else in the cluster. Without
        this, any pod can read or delete any session."""
        assert _load("k8s/base/configmap.yaml")["data"]["RAG_REQUIRE_AUTH"] == "true"

    def test_a_signing_secret_is_provided(self):
        """`RAG_REQUIRE_AUTH` without it is a startup error, by design: the
        secret is per-process otherwise, and this Deployment runs two replicas.
        """
        assert _load("k8s/base/secrets.yaml")["stringData"]["RAG_JWT_SECRET"]

    def test_an_admin_token_is_provided(self):
        """Unset does not mean "open" -- it means the GC and bulk-delete routes
        return 503 forever, which is a broken deployment, not a locked one."""
        assert _load("k8s/base/secrets.yaml")["stringData"]["RAG_ADMIN_TOKEN"]

    def test_the_placeholders_are_still_obvious(self):
        """Guards the guard: satisfying the two tests above by committing a real
        secret would be far worse than the gap they close."""
        secrets = _load("k8s/base/secrets.yaml")["stringData"]
        for key in ("RAG_JWT_SECRET", "RAG_ADMIN_TOKEN"):
            assert secrets[key] == "CHANGE_ME", f"{key} looks like a real value"

    def test_more_than_one_replica_is_why_the_secret_must_be_shared(self):
        """The premise of `_require_a_shared_signing_secret`. If this ever drops
        to 1 the check is still correct — gunicorn forks four workers — but the
        reasoning written down in several places should be revisited rather than
        silently becoming wrong."""
        spec = _load("k8s/base/deployment-api.yaml")["spec"]
        assert spec["replicas"] > 1


class TestDockerCompose:
    @pytest.fixture(scope="class")
    def compose(self):
        return _load("docker-compose.yml")

    def test_the_api_is_published_on_loopback_by_default(self, compose):
        published = compose["services"]["api"]["ports"]
        assert any("127.0.0.1" in str(p) for p in published), (
            "Docker publishes unqualified ports on 0.0.0.0 and bypasses most "
            "host firewalls to do it"
        )

    def test_the_api_is_not_the_only_service_left_exposed(self, compose):
        """The defect stated as an invariant: whatever the API does, it should
        not be the one service in this file that differs."""
        offenders = []
        for name, service in compose["services"].items():
            for port in service.get("ports", []):
                if str(port).count(":") < 2:
                    offenders.append(f"{name}: {port}")
        assert not offenders, f"published to every interface: {offenders}"

    def test_the_bind_is_overridable(self, compose):
        """Loopback by default must not mean "you can never expose it"."""
        assert any("API_BIND" in str(p) for p in compose["services"]["api"]["ports"])

    def test_the_auth_switch_is_present_and_off(self, compose):
        """Compose is the local development path, so the default stays off --
        what makes that safe is the loopback publish above, not the default."""
        env = compose["x-common-env"] if "x-common-env" in compose else {}
        found = str(env) if env else pathlib.Path(DEPLOY / "docker-compose.yml").read_text()
        assert "RAG_REQUIRE_AUTH" in found
        assert "RAG_REQUIRE_AUTH:-false" in found


class TestSecurityMdDescribesTheCode:
    @pytest.fixture(scope="class")
    def text(self):
        return (pathlib.Path(__file__).resolve().parents[1] / "SECURITY.md").read_text()

    def test_it_names_the_switch_that_actually_enforces_identity(self, text):
        """It used to say "Set RAG_ADMIN_TOKEN ... before exposing it", which
        reads as though that variable hardens the service. It guards six of
        roughly forty routes."""
        assert "RAG_REQUIRE_AUTH" in text

    def test_it_says_what_the_admin_token_does_not_cover(self, text):
        assert "not a password for the API" in text

    def test_it_discloses_that_gunicorn_bypasses_the_bind_check(self, text):
        """The bundled Dockerfile is the deployment most people will use, and it
        is the one the CLI's refusal cannot protect."""
        assert "gunicorn" in text

    def test_it_covers_the_mcp_transport_rule(self, text):
        assert "stdio" in text
