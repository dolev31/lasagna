#!/usr/bin/env bash
# Run the checks of .github/workflows/tests.yml on this machine.
#
# The same jobs, in the same order: lint, both secret scans, the hermetic suite
# with the coverage floor and sockets blocked, the in-process vector-backend
# conformance, the OpenSearch, pgvector and Docling unit tests, the license
# check, dependency audit and internal-reference grep, and the plugin template.
# `--full` adds the E2E release gate's quick subset and the release workflow's
# build and wheel smoke test. Only CI runs the tests that load real embedding
# weights, which need the network, and the live compose stack (see
# src/lasagna/deploy/README.md to run it by hand).
#
# It builds its own environment in .venv-ci with uv (installed with the `eval`
# extra and the dev group, as CI does), so the result does not depend on
# whatever happens to be installed in your shell. Nothing here needs network
# access after the first install, except gitleaks if it is not installed.
#
#   scripts/ci-local.sh            # what a branch push runs
#   scripts/ci-local.sh --full     # plus the E2E gate, as a pull request runs
set -uo pipefail

cd "$(dirname "$0")/.."
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

VENV=.venv-ci
PY="$VENV/bin/python"
failed=()

step() { printf '\n== %s\n' "$1"; }
run() {
  local name="$1"; shift
  if "$@"; then echo "   ok: $name"; else echo "   FAILED: $name"; failed+=("$name"); fi
}

if ! command -v uv >/dev/null; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 2
fi

step "environment"
[[ -x "$PY" ]] || uv venv -q --python 3.12 "$VENV"
uv pip install -q --python "$PY" -e ".[eval]" --group dev "qdrant-client>=1.7.0"
# The plugin-template step installs the template into this venv. CI runs the
# suite where it is not installed, and an installed plugin adds a benchmark and
# a reranker that the README and catalogue counts do not include, so remove it
# left over from an earlier run.
uv pip uninstall -q --python "$PY" lasagna-plugin-template

# The suite is hermetic: a throwaway HOME keeps it away from your real
# ~/.rag_knowledge store, the model hub is offline, and sockets are blocked
# except loopback.
HOME_CI="$(mktemp -d)"
export HF_HUB_OFFLINE=1
NO_NETWORK=(--disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,localhost,::1)
# Test order is shuffled (pytest-randomly). Rerun a failing order with
# RANDOMLY_SEED=<seed> scripts/ci-local.sh.
SEED="${RANDOMLY_SEED:-$RANDOM}"

step "lint"
run "ruff check" uvx --quiet ruff@0.14.7 check .
run "ruff format" uvx --quiet ruff@0.14.7 format --check .
run "mypy" uvx --quiet mypy@1.18.2 src/lasagna/sdk src/lasagna/components.py src/lasagna/core/base_components.py

step "secret scans"
if command -v gitleaks >/dev/null; then
  run "gitleaks history" gitleaks detect --source . --config .gitleaks.toml --redact --no-banner
  # CI scans a clean checkout; here, scan only what git tracks, so an ignored
  # local .env is not reported.
  tracked="$(mktemp -d)"
  git ls-files -z | xargs -0 -I{} sh -c 'mkdir -p "$1/$(dirname "$2")" && cp "$2" "$1/$2" 2>/dev/null || true' _ "$tracked" {}
  # From inside the copy: the allowlists in .gitleaks.toml are relative paths.
  run "gitleaks tree" sh -c 'cd "$1" && gitleaks dir . --config "$2/.gitleaks.toml" --redact --no-banner' \
    _ "$tracked" "$PWD"
else
  echo "   skipped: gitleaks is not installed (brew install gitleaks)"
fi
run "detect-secrets" sh -c 'git ls-files -z | xargs -0 uvx --quiet --from detect-secrets detect-secrets-hook --baseline .secrets.baseline'

step "tests with coverage floor (test order seed: $SEED)"
run "pytest" env HOME="$HOME_CI" "$PY" -m pytest tests/ --ignore=tests/live --ignore=tests/stress \
  -q -rs -p no:cacheprovider "${NO_NETWORK[@]}" --randomly-seed="$SEED" \
  --cov --cov-report=term-missing:skip-covered

step "vector backend conformance"
run "conformance" env HOME="$(mktemp -d)" \
  LASAGNA_MILVUS_DB_PATH="$(mktemp -d)/milvus_conformance.db" QDRANT_PATH="$(mktemp -d)/qdrant" \
  "$PY" -m pytest tests/test_vector_backend_conformance.py tests/test_milvus_reopen_in_new_process.py \
  tests/test_corpus_preflight_is_bounded.py tests/test_profile_guard_reads_a_real_store.py \
  -q -p no:cacheprovider

step "OpenSearch, pgvector and Docling unit tests"
# A venv of its own, so .venv-ci keeps matching what CI's main suite runs on.
VENV_EXTRAS=.venv-ci-backends
[[ -x "$VENV_EXTRAS/bin/python" ]] || uv venv -q --python 3.12 "$VENV_EXTRAS"
uv pip install -q --python "$VENV_EXTRAS/bin/python" -e ".[eval,opensearch,pgvector,docling]" --group dev
run "optional backends" env HOME="$(mktemp -d)" "$VENV_EXTRAS/bin/python" -m pytest \
  tests/test_opensearch_backend.py tests/test_pgvector_backend.py tests/test_docling_processor.py \
  tests/test_mock_targets_resolve.py -q -p no:cacheprovider

step "licenses, dependency audit and internal references"
# As CI's compliance job: the core closure only (no extras), installed in a venv
# of its own and read from there. The ignored advisory and its reason are in
# .github/workflows/tests.yml.
VENV_CORE=.venv-ci-core
# --seed: pip-audit reads the environment through its pip.
[[ -x "$VENV_CORE/bin/pip" ]] || uv venv -q --seed --python 3.12 "$VENV_CORE"
uv pip install -q --python "$VENV_CORE/bin/python" -e .
run "licenses" uvx --quiet pip-licenses --python "$VENV_CORE/bin/python" \
  --partial-match --fail-on="GPL;AGPL;SSPL" --ignore-packages lasagna-rag
run "pip-audit" env PIPAPI_PYTHON_LOCATION="$VENV_CORE/bin/python" uvx --quiet pip-audit \
  --skip-editable --progress-spinner off --ignore-vuln PYSEC-2026-3740
run "internal references" sh -c '! grep -rnIE "[a-z0-9-]+\.ibm\.com|/Users/[a-z]+/|/home/[a-z]+/" \
  --include="*.py" --include="*.toml" --include="*.yml" --include="*.yaml" --include="*.md" \
  --include="*.json" src/ tests/ e2e/ docs/ hub/ examples/ README.md CONTRIBUTING.md'

step "plugin template"
uv pip install -q --python "$PY" --no-deps ./examples/lasagna-plugin-template
run "template tests" env HOME="$(mktemp -d)" "$PY" -m pytest examples/lasagna-plugin-template/tests -q -p no:cacheprovider
run "plugins list" env HOME="$(mktemp -d)" "$VENV/bin/lasagna" plugins list
run "template discovered" env HOME="$(mktemp -d)" "$VENV/bin/lasagna" component validate \
  lasagna_plugin_template.reranker:LengthReranker
run "discovery log" env HOME="$(mktemp -d)" "$PY" -c '
from lasagna.core.registry import registry, discovery_log
registry.list_rerankers()
names = {e["name"] for e in discovery_log()}
assert "reranker.template_length" in names, names
assert all(e["ok"] for e in discovery_log()), discovery_log()
'

if [[ $FULL == 1 ]]; then
  step "E2E gate (quick subset)"
  run "e2e" "$PY" -m e2e.run --quick

  step "release build and wheel smoke test"
  # .github/workflows/release.yml, into a temporary folder so a dist/ of your
  # own is left alone.
  VERSION="$("$PY" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
  REL="$(mktemp -d)"
  run "release check" "$PY" scripts/check_release.py "v$VERSION" --allow-placeholder
  run "frontend" sh -c 'cd src/lasagna/configurator/frontend && npm ci --silent && npm run build --silent'
  run "build" uvx --quiet --from build==1.6.1 pyproject-build --installer uv --outdir "$REL/dist" .
  run "twine check" sh -c 'uvx --quiet twine==7.0.0 check --strict "$1"/dist/*' _ "$REL"
  run "distribution contents" "$PY" scripts/check_dist.py "$REL/dist" "$VERSION"
  uv venv -q --python 3.12 "$REL/venv"
  uv pip install -q --python "$REL/venv/bin/python" "$REL/dist/lasagna_rag-$VERSION-py3-none-any.whl"
  run "wheel smoke" sh -c '
    set -e; cd "$1" && mkdir -p smoke home && cd smoke
    export PATH="$1/venv/bin:$PATH" HOME="$1/home"
    test "$(lasagna --version)" = "lasagna $2"
    test "$(python -c "import lasagna; print(lasagna.__version__)")" = "$2"
    printf "def longer_first(query, text):\n    return len(text.split())\n" > smoke_component.py
    lasagna component validate smoke_component:longer_first --stage reranker' _ "$REL" "$VERSION"
fi

echo
if (( ${#failed[@]} )); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "All checks passed."
