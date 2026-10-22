# Docker and Compose

The `deploy` folder holds a Docker Compose stack and Kubernetes manifests. The
stack runs the REST API, PostgreSQL for metadata, Redis, one vector store, and
optionally a Celery worker, an MCP server and Jaeger.

No Docker? `lasagna serve` runs the same REST API in one process. See
[REST API](rest-api.md).

You need Docker with Compose v2, `make`, `openssl` and `git`.

## Start the stack

The image builds from a clone of the repository. The pip package does not carry
the build context.

```bash
git clone https://github.com/dolev31/lasagna.git
cd lasagna/src/lasagna/deploy
cp .env.example .env

# The two required secrets
sed -i.bak "s/^RAG_JWT_SECRET=$/RAG_JWT_SECRET=$(openssl rand -hex 32)/" .env
sed -i.bak "s/^POSTGRES_PASSWORD=$/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" .env

make up       # make validate, then docker compose up -d
make health   # GET /api/v1/health, pretty-printed
```

With `.env.example` as it ships, this starts Milvus, PostgreSQL, Redis, the API,
the Celery worker and beat, and the MCP server. The first `make up` builds the
image, which downloads the embedding and reranker models into it.

| Service | Address |
|---|---|
| REST API | `http://127.0.0.1:8000` (interactive docs at `/docs`) |
| MCP server | `http://127.0.0.1:8005/sse` |

Every published port is bound to `127.0.0.1`.

`make up` returns once the containers are created. To return only when every
service passes its healthcheck:

```bash
docker compose -f docker-compose.yml -f backends/milvus.yml up -d --wait
```

## Choose a vector store

`VECTOR_BACKEND` names a fragment in `backends/`. `.env.example` sets it to
`milvus`. To use another store for one command, pass it to make:

```bash
make up VECTOR_BACKEND=qdrant
```

An environment variable (`VECTOR_BACKEND=qdrant make up`) has no effect while
`.env` sets it, because the Makefile includes `.env` and that assignment wins.
Pass the same value to `make down`, `make logs` and `make ps`.

| `VECTOR_BACKEND` | What the fragment adds |
|---|---|
| `milvus` (default) | A Milvus standalone container. |
| `milvus-lite` | No container. Milvus Lite runs in the API process. |
| `pgvector` | No container. The PostgreSQL service switches to the pgvector image. |
| `qdrant` | A Qdrant container. |
| `weaviate` | A Weaviate container. |
| `opensearch` | An OpenSearch container with its security plugin disabled. |
| `pinecone` | No container. Set `PINECONE_API_KEY` in `.env`. |

The image installs the client libraries for Milvus, Milvus Lite, pgvector and
OpenSearch. The `qdrant`, `weaviate` and `pinecone` fragments add the
`vector-backends` extra at build time, so `make build` builds the right image.
After you switch to one of those three, run `make build VECTOR_BACKEND=qdrant`
(or the one you chose). `make validate` refuses an image built without the
client and names the command.

The connection a fragment sets lives under `environment:`, so a `QDRANT_URL` in
`.env` does not reach the services. To use a store outside the stack, edit the
fragment or write your own.

### Milvus Lite runs alone

Milvus Lite is a file that one process opens. With `VECTOR_BACKEND=milvus-lite`,
the API runs one worker, serves MCP itself at `/mcp/sse` on port 8000 and
processes uploads in its own threads. The worker, beat and MCP containers could
not open the file, so `make validate` refuses `milvus-lite` together with the
`celery` or `mcp` profile. Run it with no profiles:

```bash
make up VECTOR_BACKEND=milvus-lite COMPOSE_PROFILES=
```

## Choose the services

`COMPOSE_PROFILES` in `.env` picks the optional services. `.env.example` sets it
to `celery,mcp`.

| Service | Profile | What it does |
|---|---|---|
| `api` | always | The REST API, on port 8000. |
| `postgres` | always | Metadata: documents, citations, profiles and sessions. |
| `redis` | always | Cache, rate-limit buckets and the Celery broker. Not published on the host. |
| `celery-worker` | `celery` | Processes uploaded documents, four at a time. |
| `celery-beat` | `celery` | Schedules the periodic garbage collection and orphan scans. |
| `mcp` | `mcp` | A standalone [MCP server](mcp.md) over SSE, on port 8005. |
| `jaeger` | `tracing` | Trace UI on port 16686. |

Without the `mcp` profile, no MCP server runs, except with `milvus-lite`.

## Turn on auth before you publish the API

The stack starts with `RAG_REQUIRE_AUTH=false` and publishes the API on
`127.0.0.1`. In that mode, anything that can reach the port can read and delete
any session's documents. To publish the API beyond loopback, set these in `.env`
and restart:

```bash
cat >> .env <<EOF
API_BIND=0.0.0.0
RAG_REQUIRE_AUTH=true
RAG_API_KEYS=$(openssl rand -hex 32)
EOF
make up
```

A client then asks for a session token with one of the keys. See
[Authentication](rest-api.md#authentication). The
[deployment README](https://github.com/dolev31/lasagna/blob/main/src/lasagna/deploy/README.md)
lists what else auth changes. [SECURITY.md](https://github.com/dolev31/lasagna/blob/main/SECURITY.md)
says what authentication does and does not protect.

## CPU or GPU image

The image runs the CPU-only build of torch. For a host with an NVIDIA GPU, build
with the CUDA build instead:

```bash
make build TORCH_VARIANT=cuda
```

`make up` reuses an image that exists, so switching variants needs `make build`.
The API, worker and MCP containers also need GPU access: the NVIDIA Container
Toolkit on the host, and a `gpus: all` entry on those services.

## Make targets

| Target | What it does |
|---|---|
| `make up` | Run `make validate`, then start the stack. |
| `make validate` | Check the config. It stops when a required secret is empty, the backend name is wrong, or the profiles conflict with `milvus-lite`. |
| `make down` | Stop the stack and keep the volumes. |
| `make down-v` | Stop the stack and delete its volumes: documents, metadata and uploads. |
| `make build` | Rebuild the image. |
| `make logs` | Follow every log. `make logs SVC=celery-worker` follows one. |
| `make ps` | List the containers. |
| `make health` | Call `GET /api/v1/health` on the API port. |
| `make shell` | Open bash in the API container. |
| `make test` | Check the files and the merged compose config. It starts no container. |
| `make help` | List the targets. |

## Healthchecks

| Service | Check |
|---|---|
| `api` | `GET /api/v1/health/ready`, with 300 seconds to load the models on a cold start. |
| `mcp` | `GET /health` on port 8005. |
| `redis`, `postgres` | `redis-cli ping` and `pg_isready`. |
| `milvus`, `qdrant`, `weaviate`, `opensearch` | The store's own health endpoint. |

The API, worker, beat and MCP services start after Redis and PostgreSQL are
healthy.

## Kubernetes

The `k8s/` folder holds a Kustomize base and one overlay per vector store.
`overlays/local.example` is the template for the overlay that holds your secrets.
See the
[deployment README](https://github.com/dolev31/lasagna/blob/main/src/lasagna/deploy/README.md)
for the manifests.
