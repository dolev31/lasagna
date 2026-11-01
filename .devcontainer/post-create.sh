#!/bin/sh
# The dev container's first-run setup (devcontainer.json, postCreateCommand):
# the install CONTRIBUTING.md documents, under the release's constraints.
set -eu

python -m pip install --quiet uv

version=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
constraints="constraints/$version.txt"

# CPU-only torch first, at the version the constraints pin, as the Docker image
# does. From PyPI on Linux x86_64, torch brings several GB of CUDA libraries a
# Codespace has no GPU for, on a disk the Compose stack also needs.
torch=$(grep -E '^torch==' "$constraints")
python -m uv pip install --system --index-url https://download.pytorch.org/whl/cpu "$torch"

python -m uv pip install --system -e ".[all]" --group dev -c "$constraints"
pre-commit install
