# Prompt 12 — Test Suite (Unit + Integration)

## Stage
Write comprehensive tests covering every module in the package.

---

## Prompt

Write the test suite in `tests/`.

pytest config: `asyncio_mode = "auto"` (from pyproject.toml) so async tests
need no explicit @pytest.mark.asyncio decorator.

---

### tests/conftest.py

Shared fixtures:
- `mock_settings` — a Settings-like object with all fields set to test values
  (in-memory, no .env reads).
- `sample_elements` — a list of ParsedElement-like objects covering multiple
  label types (document_title, paragraph, table, image, formula).
- `sample_chunks` — pre-built Chunk objects for text, image, table, formula.
- `tmp_pdf` — path to a minimal PDF in test_data/ (or skip if missing).

---

### tests/unit/test_chunker.py

- `test_atomic_elements_get_own_chunks`: table and formula always become solo chunks.
- `test_title_attaches_forward`: a paragraph_title + paragraph become one chunk.
- `test_figure_title_joins_image_chunk`: figure_title prepended to image atomic chunk.
- `test_token_limit_splits_text`: long paragraph split into sub-chunks.
- `test_orphan_title_at_end`: title with no following content emits as solo chunk.
- `test_cross_page_heading`: heading at bottom of page N attaches to content page N+1
  (document_aware_chunking only).
- `test_modality_inference`: image elements → modality="image", etc.

---

### tests/unit/test_embedder.py

- `test_embed_texts_batching` (mock AsyncOpenAI): verify correct number of API calls
  for a list larger than batch_size.
- `test_embed_texts_replaces_empty_strings`: empty strings become "[empty]".
- `test_compute_sparse_vectors_sorted`: indices in returned SparseVector are sorted.
- `test_compute_sparse_vectors_normalised`: values sum to ≤1.0 (normalised TF).
- `test_compute_sparse_vectors_empty_string`: returns SparseVector(indices=[], values=[]).
- `test_get_embedder_unknown_provider_raises`.

---

### tests/unit/test_vector_store.py

Mock `AsyncQdrantClient`:
- `test_create_collection_new`: verify create_collection called with correct vector config.
- `test_create_collection_skip_existing`: no-op when collection already exists.
- `test_create_collection_overwrite`: delete then recreate.
- `test_upsert_chunks_builds_correct_points`: verify uuid5 IDs, payload keys.
- `test_upsert_chunks_batches`: two batches when len(chunks) > batch_size.
- `test_upsert_length_mismatch_raises`: ValueError when chunks/dense/sparse lengths differ.
- `test_search_builds_rrf_query`: verify Prefetch+FusionQuery structure.

---

### tests/unit/test_reranker.py

- `test_openai_reranker_sorts_by_score` (mock AsyncOpenAI): verify top_n returned in
  descending score order.
- `test_openai_reranker_handles_parse_failure`: returns score=0.0 on bad LLM response.
- `test_jina_reranker_calls_api` (mock httpx): verify correct payload structure.
- `test_bge_reranker_not_importable_raises`: ImportError if FlagEmbedding missing.
- `test_get_reranker_unknown_backend_raises`: ValueError for bad backend name.

---

### tests/unit/test_post_processor.py

- `test_assemble_markdown_ordering`: elements sorted by reading_order in output.
- `test_assemble_markdown_skips_labels`: image/seal/page_number not in output.
- `test_assemble_markdown_formula_wrapped`: formula text wrapped in $$.
- `test_assemble_markdown_empty`: returns "" for empty input.
- `test_save_to_json_creates_files` (tmp_path): verify .md and .json files created.
- `test_save_to_json_prefers_full_markdown`: uses full_markdown over per-page assembly.

---

### tests/unit/test_api_schemas.py

- `test_search_request_defaults`: top_k=20, rerank=True, filter_modality=None.
- `test_generate_request_requires_query`: missing query raises ValidationError.
- `test_chunk_result_optional_fields`: image_base64 and bbox can be None.
- `test_search_response_structure`: results is list[ChunkResult], latency_ms is float.

---

### tests/integration/test_ingest_e2e.py

Skip if QDRANT_URL is not reachable (use httpx to probe it first).
- `test_full_ingest_pipeline`: parse a sample PDF → chunk → embed (mocked) → upsert →
  verify point count in collection > 0.

---

### tests/integration/test_pipeline_e2e.py

Skip if Z_AI_API_KEY not set in environment.
- `test_parse_real_pdf`: parse the test_data/Docling_Technical_Report.pdf,
  verify pages > 0, total_elements > 0, full_markdown is non-empty string.

---

## What this produced

- `tests/__init__.py`
- `tests/conftest.py`
- `tests/unit/__init__.py`
- `tests/unit/test_chunker.py`
- `tests/unit/test_embedder.py`
- `tests/unit/test_vector_store.py`
- `tests/unit/test_reranker.py`
- `tests/unit/test_post_processor.py`
- `tests/unit/test_api_schemas.py`
- `tests/integration/__init__.py`
- `tests/integration/test_ingest_e2e.py`
- `tests/integration/test_pipeline_e2e.py`
- `tests/integration/.gitkeep`
- `tests/unit/.gitkeep`
