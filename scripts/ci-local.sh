#!/usr/bin/env bash
# Run the checks of .github/workflows/tests.yml on this machine.
#
# The same jobs, in the same order: lint, both secret scans, the hermetic suite
# with the coverage floor and sockets blocked, the in-process vector-backend
# conformance, the OpenSearch, pgvector and Docling unit tests, and the plugin
# template. `--full` adds the E2E release gate's quick subset. Only CI runs the
# tests that load real embedding weights, which need the network.
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
  run "gitleaks history" gitleaks git . --config .gitleaks.toml --redact --no-banner
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
  -q -rs -p no:cacheprovider "${NO_NETWORK[@]}" --randomly-seed="$SEED" --cov --cov-report=term:skip-covered

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

step "dependency audit"
# The core closure, as CI's compliance job audits it. This venv also has [eval],
# so the closure is resolved from pyproject.toml instead of read from the venv.
# The ignored advisory and its reason are in .github/workflows/tests.yml.
CORE_REQS="$(mktemp -d)/core.txt"
uv pip compile pyproject.toml --quiet --python "$PY" -o "$CORE_REQS"
run "pip-audit" uvx --quiet pip-audit -r "$CORE_REQS" --no-deps --disable-pip \
  --progress-spinner off --ignore-vuln PYSEC-2026-3740

step "plugin template"
uv pip install -q --python "$PY" --no-deps ./examples/lasagna-plugin-template
run "template tests" env HOME="$(mktemp -d)" "$PY" -m pytest examples/lasagna-plugin-template/tests -q -p no:cacheprovider
run "template discovered" env HOME="$(mktemp -d)" "$VENV/bin/lasagna" component validate \
  lasagna_plugin_template.reranker:LengthReranker

if [[ $FULL == 1 ]]; then
  step "E2E gate (quick subset)"
  run "e2e" "$PY" -m e2e.run --quick
fi

echo
if (( ${#failed[@]} )); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "All checks passed."
