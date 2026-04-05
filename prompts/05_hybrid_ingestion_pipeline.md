# Prompt 05 — Hybrid Ingestion Pipeline (Embeddings + Vector Store)

## Stage
Embed chunks with dense + sparse vectors and upsert them into Qdrant.

---

## Prompt A — Embedder

Write `src/doc_parser/ingestion/embedder.py`.

I need:

1. `embed_texts(texts, client, model, dimensions, batch_size=100) -> list[list[float]]`
   Async function using OpenAI embeddings API.
   - Replace empty strings with "[empty]" (API rejects blank inputs).
   - Batch calls in groups of `batch_size` (OpenAI limit is 2048 inputs per call).
   - Preserve response order (API guarantees this per batch).

2. BM25-proxy sparse vectors via the feature-hashing trick:
   `compute_sparse_vectors(texts, n_features=2**17) -> list[SparseVector]`
   - Tokenise with `re.findall(r"\b\w+\b", text.lower())`.
   - Hash each unique term: `idx = abs(hash(term)) % n_features`.
   - Weight = normalised term frequency (count / total_terms).
   - Sort by index (Qdrant requires sorted sparse vectors).
   - Return `qdrant_client.models.SparseVector` objects.
   - n_features=2^17=131072 is large enough to minimise collisions.

3. `BaseEmbedder` ABC with abstract `embed(texts) -> list[list[float]]`.

4. `OpenAIEmbedder(BaseEmbedder)`:
   Uses `AsyncOpenAI` client, reads `openai_api_key`, `embedding_model`,
   `embedding_dimensions` from Settings.

5. `GeminiEmbedder(BaseEmbedder)`:
   Uses `google.genai.Client`. Model: `gemini-embedding-2-preview`.
   Gemini's API is synchronous — run it in `loop.run_in_executor(None, ...)`.
   Raises ImportError if google-genai is not installed.
   Raises ValueError if GEMINI_API_KEY is missing.

6. `get_embedder(settings) -> BaseEmbedder` factory: dispatches by EMBEDDING_PROVIDER.

7. `embed_chunks(chunks, embedder, settings) -> tuple[list[list[float]], list[SparseVector]]`
   Embeds all chunks' text fields with both dense and sparse encoding.

---

## Prompt B — Vector Store

Write `src/doc_parser/ingestion/vector_store.py`.

I need `QdrantDocumentStore` using `AsyncQdrantClient`:

Two named vector spaces in the collection:
- `text_dense`:  VectorParams(size=embedding_dimensions, distance=COSINE,
                  hnsw_config=HnswConfigDiff(m=16, ef_construct=100))
- `bm25_sparse`: SparseVectorParams(index=SparseIndexParams(on_disk=False))

Methods:
1. `create_collection(overwrite=False)`:
   Check existing collections. Skip if already exists (unless overwrite=True).
   If overwrite: delete then recreate.

2. `delete_collection(collection_name) -> bool`:
   Returns False if not found, True after deletion.

3. `upsert_chunks(chunks, dense_embeddings, sparse_vectors, batch_size=64) -> int`:
   Build PointStruct for each chunk:
   - id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))   ← deterministic IDs
   - vector = {"text_dense": dense, "bm25_sparse": sparse}
   - payload = all chunk fields (text, chunk_id, source_file, page, element_types,
                bbox, is_atomic, modality, image_base64, caption)
   Upsert in batches of batch_size. Return total points upserted.

4. `search(query_text, embedder, settings, top_k=10, filter_modality=None) -> list[dict]`:
   Embed query (dense + sparse), then run Qdrant hybrid query:
   - prefetch=[Prefetch(dense, limit=top_k*2), Prefetch(sparse, limit=top_k*2)]
   - query=FusionQuery(fusion=Fusion.RRF)   ← Reciprocal Rank Fusion
   - Optional FieldCondition filter on "modality" payload field.
   Return list of point.payload dicts.

---

## What this produced

- `src/doc_parser/ingestion/embedder.py`
- `src/doc_parser/ingestion/vector_store.py`
- `src/doc_parser/ingestion/__init__.py`
