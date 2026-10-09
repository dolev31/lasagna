"""The Docker image builds from the repository root and carries its license files.

The compose build context was five levels up, which was right for the layout
this package was extracted from. Here it sent the parent of the checkout (often
the whole home directory) to the Docker daemon, which then could not find the
Dockerfile. The deployment tests validate the merged compose YAML, and
``docker compose config`` never looks inside the context, so none of them
noticed. These need no Docker.
"""

from __future__ import annotations

import pytest
import yaml

from tests._paths import DEPLOY_DIR, PACKAGE_DIR


class TestBuildContext:

    @staticmethod
    def _context():
        if PACKAGE_DIR.parent.name != "src":
            pytest.skip("the image builds from a source checkout")
        with open(DEPLOY_DIR / "docker-compose.yml") as f:
            build = yaml.safe_load(f)["services"]["api"]["build"]
        return (DEPLOY_DIR / build["context"]).resolve(), build["dockerfile"]

    def test_the_build_context_is_the_repository_root(self):
        context, dockerfile = self._context()
        assert (context / dockerfile).is_file(), f"{dockerfile} is not inside {context}"
        assert (context / "pyproject.toml").is_file(), f"no pyproject.toml in {context}"
        assert context == PACKAGE_DIR.parent.parent

    def test_everything_the_dockerfile_copies_is_in_the_context(self):
        context, dockerfile = self._context()
        sources = []
        for line in (context / dockerfile).read_text().splitlines():
            if line.startswith("COPY ") and "--from=" not in line:
                sources += line.split()[1:-1]
        assert sources, "no COPY lines found"
        missing = [s for s in sources if not (context / s).exists()]
        assert not missing, f"COPY sources absent from the build context: {missing}"

    def test_the_image_carries_the_license_files(self):
        copied = " ".join(
            line
            for line in (DEPLOY_DIR / "Dockerfile").read_text().splitlines()
            if line.startswith("COPY ")
        )
        assert "LICENSE" in copied and "NOTICE" in copied
