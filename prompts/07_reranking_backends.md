# Prompt 07 — Post-Retrieval Reranking (4 Backends)

## Stage
After hybrid vector search returns candidate chunks, rerank them by true
relevance before feeding them to the LLM.

---

## Prompt

Write `src/doc_parser/retrieval/reranker.py`.

I want a pluggable reranking system with four backends, selected by RERANKER_BACKEND.

All backends share a `BaseReranker` ABC:
```python
async def rerank(query, candidates: list[dict], top_n=5) -> list[dict]
```
Each candidate is a payload dict from Qdrant with keys: text, modality,
image_base64 (optional), source_file, page, etc.
Return the top-n candidates with a `rerank_score` key added.

---

### Backend 1 — OpenAI (RERANKER_BACKEND=openai)

Use GPT-4o-mini as an async cross-encoder.
- For text chunks: send a prompt "Rate relevance 1-10, reply with ONLY the integer."
- For image chunks: pass image_base64 inline as a vision message with the caption.
- Fire all candidates in parallel with asyncio.gather.
- Parse the integer score; return 0.0 on parse failure.
- Sort descending, return top_n.

Cost: ~$0.03–0.10 per call (20 candidates).  Latency ~800ms–2s.

---

### Backend 2 — Jina (RERANKER_BACKEND=jina) [DEFAULT]

Use Jina Reranker M0 cloud API at https://api.jina.ai/v1/rerank.
Model: `jina-reranker-m0`.
- Build documents list: text-only chunks send {text: ...},
  image chunks send {text: ..., images: [base64_string]}.
- Single batch call with all documents.
- Map results back to original candidates by index.
- Requires JINA_API_KEY.

Cost: ~$0.01–0.02 per call.  Latency ~500ms–2s.

---

### Backend 3 — BGE (RERANKER_BACKEND=bge)

Use BAAI/bge-reranker-v2-minicpm-layerwise (local model, text-only).
- Import from FlagEmbedding: `LayerWiseFlagLLMReranker`.
- cutoff_layers=[28].
- Use MPS device on Apple Silicon, fallback to CPU.
- The compute_score call is synchronous — run it in loop.run_in_executor(None, ...).
- Requires: uv pip install 'doc-parser[bge]'

Cost: free (local).  Latency ~50–100ms on Apple Silicon CPU.

---

### Backend 4 — Qwen VL (RERANKER_BACKEND=qwen)

Use Qwen3-VL-Reranker-2B (local, multimodal, ranked #1 on MMEB-V2).
- Load with transformers AutoProcessor + AutoModelForSequenceClassification.
- For image chunks: decode image_base64, pass PIL Image to processor.
- For text chunks: text-only input.
- Runs synchronously in thread pool via run_in_executor.
- MPS device on Apple Silicon, CPU fallback.
- Requires: uv pip install 'doc-parser[qwen]'

Cost: free (local).  Latency ~400–800ms on MPS, ~1–2s on CPU.

---

## Factory

```python
def get_reranker(settings) -> BaseReranker
```
Dispatch by settings.reranker_backend to one of the four classes above.
Raise ValueError for unknown backend names.

---

## What this produced

- `src/doc_parser/retrieval/reranker.py`
- `src/doc_parser/retrieval/__init__.py`
