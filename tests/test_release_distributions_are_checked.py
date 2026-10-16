"""scripts/check_dist.py refuses distributions that are incomplete or ship too much.

The check lived inline in release.yml, so `scripts/ci-local.sh` could not run
it and nothing tested it. A wheel without the built UI opens a 404 on `lasagna
serve`, and a sdist carrying a `.env` publishes whatever it holds.
"""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_dist.py"
VERSION = "0.1.0"
GOOD_WHEEL = [
    "lasagna/__init__.py",
    "lasagna/settings.lasagna.toml",
    "lasagna/py.typed",
    "lasagna/configurator/frontend/dist/index.html",
    f"lasagna_rag-{VERSION}.dist-info/licenses/LICENSE",
    f"lasagna_rag-{VERSION}.dist-info/licenses/NOTICE",
]
GOOD_SDIST = [f"lasagna_rag-{VERSION}/pyproject.toml", f"lasagna_rag-{VERSION}/.env.example"]


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_dist", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dist(tmp_path, wheel=GOOD_WHEEL, sdist=GOOD_SDIST, extra=()):
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / f"lasagna_rag-{VERSION}-py3-none-any.whl", "w") as zf:
        for name in wheel:
            zf.writestr(name, "x")
    with tarfile.open(dist / f"lasagna_rag-{VERSION}.tar.gz", "w:gz") as tf:
        for name in sdist:
            info = tarfile.TarInfo(name)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    for name in extra:
        (dist / name).write_text("x")
    return dist


def test_a_complete_distribution_passes(check, tmp_path):
    assert check.problems(_dist(tmp_path), VERSION) == []
    assert check.main([str(tmp_path / "dist"), VERSION]) == 0


def test_a_wheel_without_the_built_ui_is_refused(check, tmp_path):
    wheel = [n for n in GOOD_WHEEL if "frontend" not in n]
    (problem,) = check.problems(_dist(tmp_path, wheel=wheel), VERSION)
    assert "configurator/frontend/dist/index.html" in problem


def test_a_sdist_with_a_dotenv_or_tests_is_refused(check, tmp_path):
    sdist = GOOD_SDIST + [f"lasagna_rag-{VERSION}/src/lasagna/deploy/.env", "x/tests/t.py"]
    (problem,) = check.problems(_dist(tmp_path, sdist=sdist), VERSION)
    assert "deploy/.env" in problem and "tests/" in problem


def test_any_other_file_in_the_folder_is_refused(check, tmp_path):
    assert check.main([str(_dist(tmp_path, extra=["old.whl"])), VERSION]) == 1


def test_a_wheel_with_the_operators_k8s_secrets_is_refused(check, tmp_path):
    """Built from a configured checkout, the package-data glob took overlays/local/."""
    wheel = GOOD_WHEEL + ["lasagna/deploy/k8s/overlays/local/secrets.yaml"]
    (problem,) = check.problems(_dist(tmp_path, wheel=wheel), VERSION)
    assert "overlays/local/secrets.yaml" in problem
    (tmp_path / "template").mkdir()
    template = GOOD_WHEEL + ["lasagna/deploy/k8s/overlays/local.example/secrets.yaml"]
    assert check.problems(_dist(tmp_path / "template", wheel=template), VERSION) == []
