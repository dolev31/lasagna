# Deploying Lasagna

This directory holds the Docker Compose stack and the Kubernetes manifests that run
Lasagna as a service: the REST API, PostgreSQL for metadata, Redis, one vector store,
and optionally a Celery worker, an MCP server and Jaeger. What authentication does and
does not protect is in [SECURITY.md](../../../SECURITY.md); this page says how the files
here apply it.

## Quick start

```bash
cd src/lasagna/deploy
cp .env.example .env

# The two required secrets
sed -i.bak "s/^RAG_JWT_SECRET=$/RAG_JWT_SECRET=$(openssl rand -hex 32)/" .env
sed -i.bak "s/^POSTGRES_PASSWORD=$/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" .env

make up       # make validate, then docker compose up -d
make health   # GET /api/v1/health, pretty-printed
```

With `.env.example` as it ships, this starts Milvus, PostgreSQL, Redis, the API, the
Celery worker and beat, and the MCP server. The first `make up` builds the image, which
downloads the embedding and reranker models into it. The API answers on
`http://127.0.0.1:8000` (OpenAPI page at `/docs`) and the MCP server on
`http://127.0.0.1:8005/sse`. Every published port is bound to 127.0.0.1.

`make up` returns once the containers are created. To return only when every service
passes its healthcheck:

```bash
docker compose -f docker-compose.yml -f backends/milvus.yml up -d --wait
```

## Files

| File | What it is |
| --- | --- |
| [`docker-compose.yml`](docker-compose.yml) | The services every stack shares: api, postgres, redis and the profile services |
| `backends/<name>.yml` | One fragment per vector store: adds its container, if it has one, and points the services at it |
| [`.env.example`](.env.example) | Template for `.env`, which the Makefile, compose and the containers all read |
| [`Makefile`](Makefile) | `make up`, `make validate` and the targets below |
| [`Dockerfile`](Dockerfile) | One image for the api, worker, beat and mcp services, built from the repository root |
| `k8s/` | Kustomize base and one overlay per vector store |

Every line of `.env` reaches the api, celery-worker, celery-beat and mcp containers
(`env_file`), so LLM API keys and `RAG_*` overrides set there take effect. The exception
is a variable that `docker-compose.yml` or the fragment sets to a fixed value under
`environment:`, which wins over `.env`: `RAG_DATABASE_BACKEND`, `RAG_CACHE_BACKEND`,
`RAG_TENANT_ISOLATION`, `RAG_QUOTA_ENABLED`, `RAG_RATE_LIMITING_ENABLED`,
`RAG_ENABLE_PROMETHEUS`, `LASAGNA_UPLOAD_DIR`, the PostgreSQL and Redis URLs, and the
vector store connection.

## Vector backends

`VECTOR_BACKEND` names the fragment: the Makefile runs
`docker compose -f docker-compose.yml -f backends/$(VECTOR_BACKEND).yml`.
`.env.example` sets `VECTOR_BACKEND=milvus`. To use another backend for one command,
pass it as a make argument, `make up VECTOR_BACKEND=qdrant`. An environment variable
(`VECTOR_BACKEND=qdrant make up`) has no effect while `.env` sets it, because the
Makefile includes `.env` and that assignment wins. Pass the same value to `make down`,
`make logs` and `make ps`, which build the same file list.

| `VECTOR_BACKEND` | The fragment adds | and sets on the services |
| --- | --- | --- |
| `milvus` (default) | `milvusdb/milvus:v2.4.24` standalone, ports 19530 and 9091 | `RAG_VECTOR_BACKEND_CONFIG={"uri": "http://milvus:19530"}`, and `RAG_CONFIGURATOR=true` on the api |
| `milvus-lite` | no container: Milvus Lite in the api process, file `/data/milvus/milvus.db` on the `milvus-data` volume | see [below](#milvus-lite-runs-alone) |
| `pgvector` | no container: the postgres service switches to `pgvector/pgvector:pg16` | `PGVECTOR_DSN` to the same PostgreSQL |
| `qdrant` | `qdrant/qdrant:v1.13.2`, ports 6333 and 6334 | `QDRANT_URL=http://qdrant:6333` |
| `weaviate` | `cr.weaviate.io/semitechnologies/weaviate:1.28.4`, ports 8080 and 50051 | `WEAVIATE_URL=http://weaviate:8080` |
| `opensearch` | `opensearchproject/opensearch:2.17.1` with its security plugin disabled, port 9200 | `OPENSEARCH_HOSTS=http://opensearch:9200` |
| `pinecone` | no container (managed service) | `RAG_VECTOR_BACKEND=pinecone`; set `PINECONE_API_KEY` in `.env` |

- **Client libraries in the image.** The Dockerfile installs
  `".[production,pgvector,opensearch,flag-embedding]"`, which covers Milvus, Milvus Lite,
  pgvector and OpenSearch. Its `VECTOR_BACKEND` build argument adds the `vector-backends`
  extra for `qdrant`, `weaviate` and `pinecone` (or `all`), and the build fails if the
  client does not import. The `qdrant`, `weaviate` and `pinecone` fragments set that
  argument, so `make build` builds the right image. `make up` reuses an image that exists,
  so after switching to one of those three, `make validate` refuses an image built
  without the extra (the extra installs all three clients) and names the command:
  `make build VECTOR_BACKEND=qdrant`.
- **External databases.** The connection a fragment sets is under `environment:`, so a
  `QDRANT_URL` in `.env` does not reach the services. To use a vector store outside the
  stack, edit the fragment or write your own.
- **Weaviate** allows anonymous access unless `WEAVIATE_ANONYMOUS_ACCESS=false`; turning
  it off also needs an API key or OIDC provider configured in `backends/weaviate.yml`.

### Milvus Lite runs alone

Milvus Lite is a file that one process opens. With `VECTOR_BACKEND=milvus-lite` the api
runs uvicorn with one worker, serves MCP itself at `/mcp/sse` on port 8000
(`RAG_EMBED_MCP=true`) and processes uploads in its own threads
(`RAG_CELERY_ENABLED=false`). The worker, beat and mcp containers could not open the
file, so `make validate` refuses `milvus-lite` together with the `celery` or `mcp`
profile, and if they are started anyway those services exit at once with the reason.
Run it with no profiles (`tracing` alone is allowed):

```bash
make up VECTOR_BACKEND=milvus-lite COMPOSE_PROFILES=
```

## Services and profiles

`COMPOSE_PROFILES` in `.env` picks the optional services. `.env.example` sets
`COMPOSE_PROFILES=celery,mcp`; the profiles are `celery`, `mcp` and `tracing`.

| Service | Profile | What it does |
| --- | --- | --- |
| `api` | always | REST API: gunicorn with `WEB_CONCURRENCY` uvicorn workers (4), host port `API_PORT` (8000) |
| `postgres` | always | Metadata database (documents, citations, profiles, sessions), `postgres:16-alpine`, port `POSTGRES_PORT` (5432) |
| `redis` | always | Cache, rate-limit buckets, Celery broker (db 0) and results (db 1), `redis:7-alpine`, port `REDIS_PORT` (6379) |
| `celery-worker` | `celery` | Processes uploaded documents, `CELERY_CONCURRENCY` (4) at a time |
| `celery-beat` | `celery` | Schedules the periodic garbage collection and orphan scans; run one per stack |
| `mcp` | `mcp` | Standalone MCP server over SSE, port `MCP_PORT` (8005); see [MCP service](#mcp-service) |
| `jaeger` | `tracing` | Trace UI on 16686, OTLP gRPC on 4317 |

`RAG_CELERY_ENABLED` follows the `celery` profile: `make` sets it to `true` when
`COMPOSE_PROFILES` includes `celery` and to `false` otherwise, and `make validate` refuses
`RAG_CELERY_ENABLED=true` without the profile. Started with `docker compose` directly, it
defaults to `true`; the api then queues an upload only when a Celery worker answers a ping
on the broker (checked at most every 30 seconds), and otherwise logs `RAG_CELERY_ENABLED is
true but no Celery worker answered ...` once and processes uploads in up to
`RAG_MAX_CONCURRENT_UPLOADS` (5) threads of its own, as it does with the setting off.
GC is scheduled the same way: by Celery Beat when a worker answers, by a scheduler inside
the api otherwise. Without the `mcp` profile no MCP server runs, except with milvus-lite.

### Shared uploads volume

The api stores each upload at `/data/uploads/<document_id>/<file>` and hands that path to
the Celery worker, which opens it. Containers do not share a filesystem, so the named
volume `uploads-data` is mounted at `/data/uploads` on the api, celery-worker and mcp
services, and `LASAGNA_UPLOAD_DIR=/data/uploads` is set on all four services
(the `LASAGNA_` name wins over a `RAG_UPLOAD_DIR` in `.env`, which would move uploads off
the volume). celery-beat opens no upload and mounts nothing. The image creates
`/data/uploads` owned by its `rag` user, so the volume is writable. `make down-v` deletes
this volume with the others.

## Make targets

| Target | What it does |
| --- | --- |
| `make up` | `make validate`, then `docker compose <files> up -d` |
| `make validate` | Stops with an error when `VECTOR_BACKEND` is unset; when `backends/<VECTOR_BACKEND>.yml` does not exist (it lists the ones that do); when `RAG_JWT_SECRET` or `POSTGRES_PASSWORD` is empty; when `VECTOR_BACKEND=milvus-lite` is combined with `celery` or `mcp` in `COMPOSE_PROFILES`; when `RAG_CELERY_ENABLED` is true without `celery` in `COMPOSE_PROFILES`; and, for `qdrant`, `weaviate` and `pinecone`, when an image already built for the stack lacks that backend's client. Otherwise prints `Config OK`. It checks that the secrets are set, not how long they are |
| `make down` | Stop the stack, keep the volumes |
| `make down-v` | Stop the stack and delete its volumes (documents, metadata, uploads) |
| `make build` | Rebuild the image |
| `make logs` | Follow every service's log; `make logs SVC=celery-worker` for one |
| `make ps` | List the containers |
| `make health` | `curl` `GET /api/v1/health` on `API_PORT`; prints `API not healthy` when it gets no 2xx answer |
| `make shell` | Open bash in the api container |
| `make test` | From the repository root, `pytest tests/test_unified_deployment.py -k "not Live"`: checks the files and the merged compose config (needs the `docker compose` CLI, starts no container) |
| `make stress-test`, `make stress-test-all` | See [Stress tests](#stress-tests) |
| `make help` | List the targets |

## Healthchecks

| Service | Check | Interval / timeout / retries / start period |
| --- | --- | --- |
| `api` | `GET /api/v1/health/ready` | 15 s / 5 s / 5 / 300 s (loading the models on a cold start) |
| `celery-worker` | `celery inspect ping -d celery@$HOSTNAME --timeout 10` | 60 s / 20 s / 3 / 60 s |
| `celery-beat` | the schedule file `/tmp/celerybeat-schedule*` exists | 30 s / 5 s / 3 / 60 s |
| `mcp` | `GET /health` on port 8005 | 15 s / 5 s / 5 / 300 s |
| `redis` | `redis-cli ping` | 5 s / 3 s / 5 |
| `postgres` | `pg_isready -U rag -d rag` | 5 s / 3 s / 5 |
| `milvus`, `qdrant`, `weaviate`, `opensearch` | the store's own health endpoint | 10 s / 5 s / 10 / 90 s, 15 s, 15 s, 30 s |
| `jaeger` | none | |

The api, worker, beat and mcp start after redis and postgres are healthy; the api,
worker and mcp also wait for the vector store container. `docker compose ... up -d
--wait` returns once every service with a healthcheck is healthy (and jaeger is running),
or fails naming the one that is not. The image's own `HEALTHCHECK` probes
`/api/v1/health/live` and is replaced by the checks above.

## Authentication

`docker-compose.yml` sets `RAG_REQUIRE_AUTH=false` and publishes the API on 127.0.0.1.
In that mode a request names its session with `X-Session-ID` or `?session_id=` and no
credential, so anything that can reach the port can read and delete any session's
documents. To publish the API beyond loopback, set all three in `.env` and restart
(`RAG_API_KEYS` takes several keys, comma-separated):

```bash
cat >> .env <<EOF
API_BIND=0.0.0.0
RAG_REQUIRE_AUTH=true
RAG_API_KEYS=$(openssl rand -hex 32)
EOF
make up

# A client then asks for a session token with one of the keys
KEY=$(grep '^RAG_API_KEYS=' .env | cut -d= -f2)
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions -H "X-API-Key: $KEY"
```

With `RAG_REQUIRE_AUTH=true`:

- `POST /api/v1/sessions` with `X-API-Key: <one of RAG_API_KEYS>` answers 201 with
  `token`, `session_id`, `user_id` and `expires_at`. A missing or wrong key is 401. When
  `RAG_API_KEYS` holds no usable key (unset, a placeholder such as `CHANGE_ME`, or shorter
  than 32 characters) the endpoint answers 503 and issues no token. Naming a `session_id`
  or `user_id` in the body needs `X-Admin-Token` (`RAG_ADMIN_TOKEN`), otherwise 403.
- Every other request sends the token as `X-Session-Token`. A session named by
  `X-Session-ID` or `?session_id=` without a token is refused with 401, the shared
  `default` session included (unless `RAG_ALLOW_ANONYMOUS_SESSION=true`). An invalid or
  expired token is 401 with auth on or off.
- Rate limits are counted per API key, not per session (see [Request limits](#request-limits)).

Tokens are HS256 JWTs signed with `RAG_JWT_SECRET`, valid for
`RAG_SESSION_TOKEN_TTL_SECONDS` (604800, 7 days). The secret must be identical in every
worker and replica and at least 32 characters; a shorter value or a placeholder counts as
unset. Unset with auth on, the app refuses to start. Unset with auth off, each process
signs with a random secret of its own, so a token works only on the gunicorn worker that
issued it and not after a restart. Compose refuses to start without it.

**The bind check.** `lasagna serve` and `lasagna-mcp --transport sse` refuse to start on
a non-loopback address unless `RAG_REQUIRE_AUTH` or `RAG_BIND_IS_PROTECTED` is true; on
loopback they start with a warning when auth is off. Inside a container every process
binds 0.0.0.0, whatever the published address, so compose sets
`RAG_BIND_IS_PROTECTED=true` to say that the loopback port publishing protects it. The
image runs gunicorn directly and never runs this check, which is why `API_BIND` and
`RAG_REQUIRE_AUTH` have to be changed together by you.

Other settings: `RAG_ADMIN_TOKEN` guards the `/api/v1/admin/*` routes (garbage
collection, deleting a session) and returns 503 for them while unset.
`RAG_CORS_ORIGINS` lists the browser origins allowed to call the API (none by default).
`FORWARDED_ALLOW_IPS` names the proxies whose `X-Forwarded-For` gunicorn trusts for the
client address (compose default `127.0.0.1`); `RAG_FORWARDED_ALLOW_IPS` is the name
`lasagna serve` reads. `RAG_ALLOWED_HOSTS` adds `Host` values the embedded `/mcp` mount
accepts, such as a service name.

## Request limits

| Limit | Default | When it is exceeded |
| --- | --- | --- |
| One upload, `POST /api/v1/documents` | `RAG_MAX_FILE_SIZE` = 52428800 (50 MiB) | 413. A `Content-Length` over the limit plus 1 MiB of multipart overhead is refused before the body is read; a file over the limit is refused while it is copied |
| One file inside a batch | `RAG_MAX_FILE_SIZE` | that item reports `status: "error"` in the 201 response |
| Total body of `POST /api/v1/documents/batch` | `RAG_MAX_BATCH_BYTES` = 209715200 (200 MiB), plus 1 MiB | 413, before the body is read |
| Files in one batch | `RAG_MAX_BATCH_FILES` = 20 | 400 |
| Any other request body | 1 MiB, fixed | 413 |
| Search time, `POST /api/v1/search`, `/search/citations`, `/search/multi-query`, `/search/agentic` | `RAG_SEARCH_TIMEOUT` = 120 s | 504; the search thread finishes on its own |
| Processing threads, single upload without Celery | `RAG_MAX_CONCURRENT_UPLOADS` = 5 | 503 with `Retry-After: 5`; nothing is stored, so a retry is a clean upload |
| Request rate | 60 per minute, burst 10, `adaptive` | 429 with `Retry-After` (seconds, rounded up) |
| Quota per session | 1,000 documents, 5 GiB, 100,000 chunks, 600 searches per minute, 200 uploads per hour | 429 |

A chunked body is counted as it arrives and refused once it passes its limit.
`RAG_MAX_BATCH_FILES` (1 to 1000), `RAG_MAX_BATCH_BYTES` (1 to 2^40) and
`RAG_SEARCH_TIMEOUT` (1 to 3600) are checked when the app starts: a value out of range or
not a number stops it with an error naming the variable.

**Rate limits** apply when `RAG_RATE_LIMITING_ENABLED` is true (the code default is false;
compose and k8s turn it on), tuned by `RAG_RATE_LIMITING_RPM`, `RAG_RATE_LIMITING_BURST`
and `RAG_RATE_LIMITING_STRATEGY` (`adaptive` or `fixed`). With the Redis cache the buckets
live in Redis and every worker shares them. A request is counted against its session with
auth off; with auth on, against the API key that minted its token, or against the
client address when it carries no verified token. `/`, `/metrics`, the health routes,
`/docs`, `/openapi.json` and `/redoc` are not counted. **Quotas** apply when
`RAG_QUOTA_ENABLED` is true (on in compose and k8s).

**Shutdown.** On SIGTERM the api waits up to `RAG_PROCESSING_TIMEOUT` seconds (300 in the
code, 20 in compose) for documents still processing in its own threads, then marks the
rest failed with the reason, so none stays "processing" after the process is gone. That
wait must end before gunicorn's `--graceful-timeout` (30 s in the image), which must end
before the container is killed (`stop_grace_period: 40s` for the api). With the `celery`
profile the worker processes documents and these threads are a fallback.

## Observability

### Health endpoints

| Endpoint | What it answers | Used by |
| --- | --- | --- |
| `GET /api/v1/health/live` | 200 `{"status": "alive"}` while the process runs | the image `HEALTHCHECK`, k8s startup and liveness probes |
| `GET /api/v1/health/ready` | Pings the vector store, the metadata database and, when the cache is Redis or Celery is on, Redis, each with a 2 s timeout, and checks that the embedding model is loaded. 200 when all answer; 503 with the names of the ones that did not in `failing` | compose api healthcheck, k8s readiness probe |
| `GET /api/v1/health` | Status of the embedding model, vector store and database, with no timeouts: `healthy`, `degraded` (database down) or `unhealthy`, always with 200 | `make health`, dashboards |

None of them needs a credential or is rate limited. `/health/ready` and `/health` answer
503 while the backend has not loaded, and both report `version` and `require_auth`.

### Logs

Each request gets an id: a client's `X-Request-ID` is kept when it is 1 to 128 characters
of letters, digits and `._:-`, otherwise one is generated. It is returned in the
`X-Request-ID` response header and attached to every log line written while handling the
request. `LASAGNA_LOG_FORMAT=json` makes the api and mcp services write one JSON object
per line (loguru's serialized format, id under `record.extra.request_id`, level from
`RAG_LOG_LEVEL`, default `INFO`); any value other than `json` or `text` stops the app at
startup. Gunicorn's access log keeps its own text format.

### Prometheus

`RAG_ENABLE_PROMETHEUS=true` (on in compose and k8s, off in the code) makes `GET /metrics`
on the api port serve Prometheus text; otherwise it answers 404. It needs the
`prometheus-client` package, which the image has, and is not authenticated. Compose and
k8s set `PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc` so the gunicorn workers'
metrics are collected together; the image empties that directory at start. The k8s api
pods carry `prometheus.io/scrape` annotations for port 8000, path `/metrics`.

| Metric | Labels |
| --- | --- |
| `rag_http_request_duration_ms` (histogram) | `method`, `path` (the route template, `unmatched` when no route matched), `status` |
| `rag_http_rate_limited_total` | `path` |
| `lasagna_llm_calls_total` | `outcome`: `ok` or `error` |
| `lasagna_ingests_total` | `outcome`: `ready` or `error` |
| `lasagna_degraded_fallbacks_total` | `stage` |

Metrics are counted in the process that does the work. With the `celery` profile, documents
are processed in the worker container, which serves no `/metrics`, so the api's
`lasagna_ingests_total` counts only the documents the api processed itself.

`lasagna_degraded_fallbacks_total{stage}` counts a pipeline stage that failed and fell back
to a lesser result (`hyde`, `graphrag`, `hybrid`, `query_expansion`, `agentic`). The
search response names the same stages in `degraded_stages`, empty when every configured
stage ran.

### Tracing

Set `RAG_ENABLE_TRACING=true` and add `tracing` to `COMPOSE_PROFILES`. Spans go over OTLP
gRPC to `RAG_TRACING_ENDPOINT` (compose default `http://jaeger:4317`) under the service
name `rag-knowledge` (`RAG_TRACING_SERVICE_NAME`). The Jaeger UI is at
`http://127.0.0.1:16686`.

## MCP service

The `mcp` profile runs `python -m lasagna.mcp --transport=sse --host=0.0.0.0 --port=8005`
(stdio is the server's default transport; a container serves a port instead), published
on `127.0.0.1:${MCP_PORT:-8005}`. Clients connect to `http://127.0.0.1:8005/sse`; the
healthcheck uses `/health`. On SSE the server loads the backend before it serves.

- **Tools.** `LASAGNA_MCP_TOOLS` defaults to `safe`, which leaves out the seven tools that
  delete data, reset configuration or change the server-wide retention policy
  (`clear_session`, `bulk_delete_session`, `delete_document`, `delete_rag_profile`,
  `rebuild_collection`, `reset_rag_config`, `set_lifecycle_policy`). Set
  `LASAGNA_MCP_TOOLS=all` in `.env` to serve them (`--tools all` outside compose).
- **Auth.** The service runs the bind check above; compose's `RAG_BIND_IS_PROTECTED=true`
  lets it start on 0.0.0.0. If you publish it beyond loopback, set `RAG_REQUIRE_AUTH=true`;
  tools then need a `session_token` argument issued by `POST /api/v1/sessions`.
- **Uploads.** `upload_document` over MCP reads server-side paths only inside
  `RAG_MCP_UPLOAD_ROOT`, which compose leaves unset, so it reads none; upload through the
  REST API. Pointing it at `/data/uploads` would let any MCP client ingest another
  session's stored files.

With milvus-lite the api serves MCP at `/mcp/sse` on port 8000 instead. The server itself
is documented in [src/lasagna/mcp/README.md](../mcp/README.md).

## Kubernetes

`k8s/base` holds the namespace `rag-knowledge`, a ConfigMap and a Secret, the
`rag-knowledge-api` Deployment (2 replicas, image `rag-knowledge:latest`, port 8000), the
`rag-knowledge-celery` worker Deployment (2 replicas, `--concurrency=4`), a ClusterIP
Service on port 80 and two HorizontalPodAutoscalers. Each overlay sets
`RAG_VECTOR_BACKEND` and its connection: `milvus`, `pgvector`, `pinecone`, `qdrant`,
`weaviate`. There is no overlay for OpenSearch or Milvus Lite.

```bash
# From the repository root: build the image the manifests name, then push it where
# your cluster pulls images from.
docker build -t rag-knowledge:latest -f src/lasagna/deploy/Dockerfile .

# From src/lasagna/deploy
kubectl apply -k k8s/overlays/pgvector/
```

Before you apply:

1. **Replace every `CHANGE_ME`** in `base/secrets.yaml` and the overlay:
   `RAG_JWT_SECRET`, `RAG_ADMIN_TOKEN`, `RAG_API_KEYS`, the PostgreSQL password in the
   DSNs and `PINECONE_API_KEY`. A placeholder counts as unset: the api pods do not start
   until `RAG_JWT_SECRET` is real, because the ConfigMap sets `RAG_REQUIRE_AUTH=true`, and
   `POST /api/v1/sessions` answers 503 until `RAG_API_KEYS` is. Prefer sealed-secrets or
   an external secrets operator to a plain manifest.
2. **Provide PostgreSQL, Redis and the vector store.** The manifests deploy none of them;
   the Secret and overlays point at `postgres.rag-knowledge:5432`,
   `redis.rag-knowledge:6379`, `milvus.rag-knowledge:19530`, `qdrant.rag-knowledge:6333`
   and `weaviate.rag-knowledge:8080`. The `qdrant`, `weaviate` and `pinecone` overlays also
   need their client in the image: build it with `--build-arg VECTOR_BACKEND=qdrant` (or
   `weaviate`, `pinecone`, `all`; see [Vector backends](#vector-backends)).
3. **Share the uploads.** The manifests mount no volume and turn Celery on, so a worker
   pod cannot open a file an api pod stored. Mount one ReadWriteMany volume at the same
   path in both Deployments and set `LASAGNA_UPLOAD_DIR` to it, or set
   `RAG_CELERY_ENABLED: "false"` so the api processes its own uploads.
4. **Fit the shutdown drain.** The api pods get `terminationGracePeriodSeconds: 45` and
   gunicorn waits 30 s, but the ConfigMap leaves `RAG_PROCESSING_TIMEOUT` at 300; set it
   below 30, as compose does.
5. **Check tracing.** The ConfigMap turns tracing on and sends it to
   `http://otel-collector.observability:4317`.
6. **Size the pods** for your embedding model: the api requests 500m CPU and 2Gi (limit 2
   CPUs, 6Gi), the worker 1 CPU and 2Gi (limit 4 CPUs, 8Gi).

No Celery beat runs in these manifests, so the periodic garbage collection and orphan
scans that compose schedules do not run. The ConfigMap embeds with
`BAAI/bge-large-en-v1.5` (compose uses `BAAI/bge-m3`) and sets `RAG_DEFAULT_TOP_K=75`,
`RAG_RERANK_TOP_K=10`, partition tenant isolation, quotas and adaptive rate limiting.

Probes: the api has a startup probe on `/api/v1/health/live` (up to 30 tries, 5 s apart),
liveness on `/api/v1/health/live` every 15 s and readiness on `/api/v1/health/ready` every
10 s; the worker's liveness runs `celery inspect ping --timeout=5` every 30 s.

| Autoscaler | Replicas | Scales up on |
| --- | --- | --- |
| api | 2 to 10 | CPU over 70% or memory over 80% |
| worker | 1 to 8 | CPU over 75% |

## Configuration

Each setting resolves from, highest first: the session's overrides, the environment
(`LASAGNA_<NAME>`, then `RAG_<NAME>`), the settings file, the code default
([Configuration](../../../docs/configuration.md#where-a-setting-comes-from)). The server
switches on this page that are not settings are read only by the name shown:
`RAG_REQUIRE_AUTH`, `RAG_API_KEYS`, `RAG_JWT_SECRET`, `RAG_ADMIN_TOKEN`,
`RAG_MAX_BATCH_FILES`, `RAG_MAX_BATCH_BYTES`, `RAG_SEARCH_TIMEOUT`, `RAG_CORS_ORIGINS`,
`RAG_BIND_IS_PROTECTED`, `RAG_EMBED_MCP`, `RAG_SESSION_BACKEND`, `LASAGNA_LOG_FORMAT` and
`LASAGNA_MCP_TOOLS`.

| Setting | Code default | Compose | k8s |
| --- | --- | --- | --- |
| `RAG_DATABASE_BACKEND` | `sqlite` | `postgresql` | `postgresql` |
| `RAG_CACHE_BACKEND` | `memory` | `redis` | `redis` |
| `RAG_CELERY_ENABLED` | `false` | follows the `celery` profile under `make`, else `true` | `true` |
| `RAG_SESSION_BACKEND` | in memory (`persistent` for `lasagna` and `lasagna-mcp`) | `persistent` | `persistent` |
| `RAG_CONNECTION_POOL_SIZE` (vector store) | `1` | `10` | `10` |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | `BAAI/bge-m3` | `BAAI/bge-large-en-v1.5` |
| `RAG_RERANKER_MODEL` | `mixedbread-ai/mxbai-rerank-large-v1` | same | same |
| `RAG_DEFAULT_TOP_K` | `20` | `20` | `75` |
| `RAG_ENABLE_HYBRID_SEARCH`, `RAG_ENABLE_RERANKING` | `true` | `true` | `true` |
| `RAG_ENABLE_PROMETHEUS` | `false` | `true` | `true` |
| `RAG_ENABLE_TRACING` | `false` | `false` | `true` |

The settings file is the packaged `settings.lasagna.toml`. `RAG_SETTINGS_FILE` (or
`LASAGNA_SETTINGS_FILE`) names another, absolute or relative to the installed `lasagna`
package. `RAG_SETTINGS_FILE=settings.lasagna_production.toml` loads the bundled
production file: PostgreSQL, Redis, Celery, partition isolation, quotas, adaptive rate
limiting, Prometheus and tracing, `BAAI/bge-large-en-v1.5`, top-k 75 reranked to 10, and
agentic RAG on (which needs an LLM API key). The environment wins over that file, so in
compose the embedding model, the reranker and the tracing switch stay as
`docker-compose.yml` sets them; set `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL` or
`RAG_ENABLE_TRACING` in `.env` to change them.

At runtime, `PATCH /api/v1/config` changes one session's search settings, and with
`RAG_CONFIGURATOR=true` the api serves a configuration page at `/configurator`.

## Upgrade notes

- **Metadata store version.** SQLite and PostgreSQL record a store version
  (`STORE_VERSION`, 6) in their `schema_meta` table. An older store is migrated forward at
  startup; a store recorded by a newer Lasagna is refused with an error rather than read.
  PostgreSQL creates and migrates its schema inside a transaction holding an advisory
  lock, so replicas and workers starting together do not race.
- **Data directory.** Lasagna keeps its data in `~/.lasagna` (`dbs/` and `uploads/`), or
  in `LASAGNA_DATA_DIR`; `LASAGNA_DB_DIR` and `LASAGNA_UPLOAD_DIR` move each part (the
  `RAG_` names also work). On first use, `~/.rag_knowledge` is renamed to `~/.lasagna`
  when only it exists; two existing directories are never merged. In the image the home
  directory is `/home/rag`, which is not on a volume: compose keeps uploads on
  `uploads-data`, metadata in PostgreSQL (`postgres-data`) and Milvus Lite's file on
  `milvus-data`.
- **One vector collection per embedding model.** Each model and dimension writes to
  `rag_documents_v2__<model>__<dim>`. A collection name other than `rag_documents_v2` is
  used as given. Before 0.1.0 every model wrote to `rag_documents_v2`: the first model to
  open it at the dimension it holds, before that model has a collection of its own, keeps
  using it under that name; other models get their own collections. When no model takes
  it over, it stays in place and `lasagna status` lists it as unused. A collection whose
  vectors another model wrote is refused with an error naming that model.
- **Session tokens** are issued with the issuer `lasagna`. Tokens issued as
  `rag-knowledge` before 0.1.0 stay valid until they expire.

## Stress tests

`make stress-test` runs `tests/stress/` from a source checkout against `VECTOR_BACKEND`;
`make stress-test-all` runs it against `pgvector`, `qdrant`, `weaviate` and `milvus-lite`
in turn and writes `/tmp/stress-<backend>.log` for each. Both use
`all-MiniLM-L6-v2` (384 dimensions) and `cross-encoder/ms-marco-MiniLM-L-6-v2` so they run
on a CPU. The load is set by `STRESS_NUM_TENANTS` (10), `STRESS_DOCS_PER_TENANT` (5),
`STRESS_CONCURRENT_SEARCHES` (20) and `STRESS_ENDURANCE_MINUTES` (1). See
[tests/stress/README.md](../../../tests/stress/README.md).
