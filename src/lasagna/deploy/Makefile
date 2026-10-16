# =============================================================================
# Lasagna: Deployment Makefile
# =============================================================================
# Usage:
#   make up                         Start with VECTOR_BACKEND from .env
#   VECTOR_BACKEND=pgvector make up Override backend on the fly
#   make down                       Stop all services (keep volumes)
#   make down-v                     Stop all services + delete volumes
#   make logs                       Follow API logs
#   make logs SVC=celery-worker     Follow specific service
#   make build                      Rebuild Docker image
#   make ps                         Show running containers
#   make health                     Check API health endpoint
#   make stress-test                Run stress tests (~6 min, single backend)
#   make stress-test-all            Run stress tests (~27 min, all backends)
#   make validate                   Validate config (backend, secrets, profiles)
#   make help                       Show all commands
# =============================================================================

.DEFAULT_GOAL := help

# Load .env if it exists (VECTOR_BACKEND, COMPOSE_PROFILES, ports, etc.)
-include .env
export

# Assemble compose files: base + backend fragment
FILES := -f docker-compose.yml -f backends/$(VECTOR_BACKEND).yml

# The api queues uploads for Celery only when the celery profile runs a worker.
# Unset, RAG_CELERY_ENABLED follows COMPOSE_PROFILES; `make validate` refuses
# true without the profile. (Started without make, the api checks for a worker
# itself and processes uploads in its own threads when none answers.)
comma := ,
RAG_CELERY_ENABLED ?= $(if $(filter celery,$(subst $(comma), ,$(COMPOSE_PROFILES))),true,false)

.PHONY: up down down-v build logs ps health validate help shell test stress-test stress-test-all

up: validate  ## Start the stack
	docker compose $(FILES) up -d

down:  ## Stop all services (keep volumes)
	docker compose $(FILES) down

down-v:  ## Stop all services + delete volumes
	docker compose $(FILES) down -v

build:  ## Rebuild the Docker image
	docker compose $(FILES) build

logs:  ## Follow logs (SVC=api|celery-worker|mcp|redis|postgres|...)
	docker compose $(FILES) logs -f $(SVC)

ps:  ## Show running containers
	docker compose $(FILES) ps

health:  ## Check API health endpoint
	@curl -sf http://localhost:$${API_PORT:-8000}/api/v1/health | python3 -m json.tool || echo "API not healthy"

validate:  ## Validate configuration
	@test -n "$(VECTOR_BACKEND)" || { echo "ERROR: VECTOR_BACKEND is not set. Set it in .env or pass on command line."; exit 1; }
	@test -f backends/$(VECTOR_BACKEND).yml || { echo "ERROR: Unknown VECTOR_BACKEND='$(VECTOR_BACKEND)'. Available:"; ls -1 backends/*.yml | sed 's|backends/||;s|\.yml||' | sed 's/^/  /'; exit 1; }
	@test -n "$(RAG_JWT_SECRET)" || { echo "ERROR: RAG_JWT_SECRET is not set. Generate one with: openssl rand -hex 32"; exit 1; }
	@test -n "$(POSTGRES_PASSWORD)" || { echo "ERROR: POSTGRES_PASSWORD is not set. Generate one with: openssl rand -hex 16"; exit 1; }
	@if [ "$(VECTOR_BACKEND)" = "milvus-lite" ]; then \
		case ",$$(printf '%s' '$(COMPOSE_PROFILES)' | tr -d ' ')," in \
		*,celery,*|*,mcp,*) echo "ERROR: VECTOR_BACKEND=milvus-lite runs inside the api process, so the celery and mcp profiles cannot reach it (COMPOSE_PROFILES=$(COMPOSE_PROFILES)). Set COMPOSE_PROFILES= (MCP is then served by the api at /mcp/sse), or use a server backend such as VECTOR_BACKEND=milvus."; exit 1;; \
		esac; \
	fi
	@case "$$(printf '%s' '$(RAG_CELERY_ENABLED)' | tr '[:upper:]' '[:lower:]')" in \
	1|true|yes|on) \
		case ",$$(printf '%s' '$(COMPOSE_PROFILES)' | tr -d ' ')," in \
		*,celery,*) ;; \
		*) echo "ERROR: RAG_CELERY_ENABLED=$(RAG_CELERY_ENABLED) queues uploads for a Celery worker, but COMPOSE_PROFILES=$(COMPOSE_PROFILES) starts none. Add celery to COMPOSE_PROFILES, or remove RAG_CELERY_ENABLED from .env so it follows the profile."; exit 1;; \
		esac;; \
	esac
	@# `up` reuses an image that exists, so one built without the vector-backends
	@# extra lacks this backend's client (see the Dockerfile's VECTOR_BACKEND
	@# argument). The extra installs all three clients.
	@case "$(VECTOR_BACKEND)" in qdrant|weaviate|pinecone) \
		for img in $$(docker compose $(FILES) config --images 2>/dev/null | grep -E -- '-(api|celery-worker|celery-beat|mcp)$$'); do \
			built=$$(docker image inspect --format '{{ index .Config.Labels "org.lasagna.vector-backend" }}' "$$img" 2>/dev/null) || continue; \
			case "$$built" in \
			qdrant|weaviate|pinecone|all) ;; \
			*) echo "ERROR: The image $$img has no $(VECTOR_BACKEND) client: it was built for VECTOR_BACKEND=$$(echo "$$built" | sed 's/<no value>//;s/^$$/milvus/'), without the vector-backends extra. Rebuild it with: make build VECTOR_BACKEND=$(VECTOR_BACKEND)"; exit 1;; \
			esac; \
		done;; \
	esac
	@echo "Config OK: VECTOR_BACKEND=$(VECTOR_BACKEND) COMPOSE_PROFILES=$(COMPOSE_PROFILES) RAG_CELERY_ENABLED=$(RAG_CELERY_ENABLED)"

shell:  ## Open a shell in the API container
	docker compose $(FILES) exec api bash

test:  ## Run static deployment tests
	cd ../../.. && python -m pytest tests/test_unified_deployment.py -v -k "not Live"

# Stress test env: lightweight models for fast CPU execution
STRESS_ENV := RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2 \
	RAG_EMBEDDING_DIM=384 \
	RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
	STRESS_NUM_TENANTS=$${STRESS_NUM_TENANTS:-10} \
	STRESS_DOCS_PER_TENANT=$${STRESS_DOCS_PER_TENANT:-5} \
	STRESS_CONCURRENT_SEARCHES=$${STRESS_CONCURRENT_SEARCHES:-20} \
	STRESS_ENDURANCE_MINUTES=$${STRESS_ENDURANCE_MINUTES:-1}

STRESS_ROOT := $(shell cd ../../.. && pwd)

stress-test:  ## Run stress tests (VECTOR_BACKEND from .env, ~6 min)
	cd $(STRESS_ROOT) && $(STRESS_ENV) \
		STRESS_BACKEND_ONLY=$(VECTOR_BACKEND) \
		python -m pytest tests/stress/ -m stress -v

stress-test-all:  ## Run stress tests against ALL backends (~27 min)
	# milvus (standalone) excluded: requires x86_64 architecture
	# pinecone excluded: cloud-only, requires API key
	@for backend in pgvector qdrant weaviate milvus-lite; do \
		echo "\n========== Stress testing: $$backend ==========\n"; \
		cd $(STRESS_ROOT) && $(STRESS_ENV) \
			STRESS_BACKEND_ONLY=$$backend \
			python -m pytest tests/stress/ -m stress -v \
			2>&1 | tee /tmp/stress-$$backend.log; \
		echo "\n========== $$backend complete ==========\n"; \
	done
	@echo "\nResults saved to /tmp/stress-{pgvector,qdrant,weaviate,milvus-lite}.log"

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
