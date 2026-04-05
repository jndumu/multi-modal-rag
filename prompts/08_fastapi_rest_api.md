# Prompt 08 — FastAPI REST API

## Stage
Expose the full RAG pipeline as a REST API with four endpoints.

---

## Prompt

Build the FastAPI application in `src/doc_parser/api/`.

### Schemas (`schemas.py`)

Pydantic v2 models for all request/response types:

**SearchRequest**: query (str), top_k=20, top_n=None, rerank=True,
                   filter_modality=None

**ChunkResult**: chunk_id, text, source_file, page, modality, element_types,
                 bbox, is_atomic, caption, rerank_score, image_base64=None

**SearchResponse**: query, backend, total_candidates, results: list[ChunkResult],
                    latency_ms

**GenerateRequest**: query, top_k=20, top_n=None, rerank=True, filter_modality=None,
                     max_tokens=1024, system_prompt=None

**GenerateResponse**: query, answer, sources: list[ChunkResult],
                      total_candidates, latency_ms

**IngestResponse**: filename, chunks_upserted, latency_ms

**HealthResponse**: status, version, parser_backend, reranker_backend,
                    embedding_provider, collection

---

### Dependencies (`dependencies.py`)

Module-level lazy singletons, initialised once per process:
- `get_store() -> QdrantDocumentStore`
- `get_embedder_dep() -> BaseEmbedder`
- `get_reranker_dep() -> BaseReranker`
- `get_openai_client() -> AsyncOpenAI`

---

### Middleware (`middleware.py`)

`LoggingMiddleware(BaseHTTPMiddleware)`:
Logs method, path, status_code, and latency_ms for every request using loguru.

---

### Routes

**GET /health** (`routes/health.py`):
Returns HealthResponse with status="ok", version="0.1.0", and active backend names
from Settings.

**POST /ingest** (`routes/ingest.py`):
Accept multipart PDF upload (`UploadFile`).
- Save to a NamedTemporaryFile.
- Run parse → chunk → enrich → embed → upsert pipeline.
- Return IngestResponse with chunk count and latency.

**POST /search** (`routes/search.py`):
1. Embed query (dense + sparse).
2. Hybrid search Qdrant (top_k candidates).
3. Optionally rerank with the configured backend.
4. Return SearchResponse with ChunkResult list.
   image_base64 is omitted from the response by default (large payload).
5. Set null rerank_score when rerank=False.

**POST /generate** (`routes/generate.py`):
1. Same retrieve + rerank flow as /search.
2. Build context string: join "[page N] {text}" for each candidate.
3. Call GPT-4o with system prompt + context + question.
4. Default system prompt: "Answer using ONLY the provided context. Cite page numbers.
   If not in context, say 'I don't have enough information.'"
5. Return GenerateResponse with answer + source chunks.

---

### App factory (`app.py`)

```python
def create_app() -> FastAPI
```
- Add LoggingMiddleware.
- Mount routers: health (no prefix), /ingest, /search, /generate.
- Lifespan: call setup_logging on startup, log shutdown.
- Title: "doc-parser RAG API", version: "0.1.0".

Expose `app = create_app()` at module level for uvicorn.

---

## What this produced

- `src/doc_parser/api/app.py`
- `src/doc_parser/api/dependencies.py`
- `src/doc_parser/api/middleware.py`
- `src/doc_parser/api/schemas.py`
- `src/doc_parser/api/routes/health.py`
- `src/doc_parser/api/routes/ingest.py`
- `src/doc_parser/api/routes/search.py`
- `src/doc_parser/api/routes/generate.py`
- `src/doc_parser/api/routes/__init__.py`
- `src/doc_parser/api/__init__.py`
