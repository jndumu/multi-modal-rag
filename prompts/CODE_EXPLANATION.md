# Code Explanation — Multimodal RAG System

A step-by-step walkthrough of every layer of the codebase, from raw PDF to
answered question.

---

## Table of Contents

1. [Big Picture — What This System Does](#1-big-picture)
2. [Configuration Layer](#2-configuration-layer)
3. [Document Parsing Pipeline](#3-document-parsing-pipeline)
4. [Post-Processor — Markdown Assembly](#4-post-processor)
5. [Structure-Aware Chunker](#5-structure-aware-chunker)
6. [Ingestion — Embeddings](#6-ingestion--embeddings)
7. [Ingestion — Multimodal Enrichment](#7-ingestion--multimodal-enrichment)
8. [Ingestion — Vector Store (Qdrant)](#8-ingestion--vector-store)
9. [Retrieval — Reranking Backends](#9-retrieval--reranking)
10. [FastAPI REST API](#10-fastapi-rest-api)
11. [CLI Scripts](#11-cli-scripts)
12. [Ollama Self-Hosted Mode](#12-ollama-self-hosted-mode)
13. [Streamlit Visualizer](#13-streamlit-visualizer)
14. [Data Flow — End to End](#14-data-flow-end-to-end)

---

## 1. Big Picture

This is a **multimodal RAG** system.  "Multimodal" means it understands not
only the text in a PDF, but also its tables, figures, formulas, and algorithms.

The system has two phases:

### Phase 1 — Ingestion (offline, one time per document)
```
PDF file
  ↓  GLM-OCR (layout detection + text recognition)
ParseResult (structured elements per page)
  ↓  Structure-aware chunker
list[Chunk]  (text/image/table/formula chunks)
  ↓  GPT-4o enrichment (captions + descriptions)
Enriched Chunks  (every chunk has useful text now)
  ↓  Dense embedder (OpenAI) + Sparse encoder (BM25)
(dense_vectors, sparse_vectors)
  ↓  Qdrant upsert
Vector Database  (searchable by meaning AND keywords)
```

### Phase 2 — Query (real-time, per user question)
```
User question
  ↓  Embed query (dense + sparse)
  ↓  Hybrid search Qdrant (RRF fusion of dense + sparse results)
Top-20 candidate chunks
  ↓  Reranker (Jina / OpenAI / BGE / Qwen)
Top-5 most relevant chunks
  ↓  GPT-4o with context
Answer  (grounded in the document, with page citations)
```

---

## 2. Configuration Layer

**File:** [src/doc_parser/config.py](../src/doc_parser/config.py)

### How it works

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    parser_backend: str = "cloud"
    z_ai_api_key: SecretStr | None = None
    ...
```

`pydantic-settings` automatically reads values from:
1. Environment variables (highest priority)
2. `.env` file
3. Field defaults (lowest priority)

`SecretStr` wraps secrets so they don't accidentally appear in logs or stack
traces — you must call `.get_secret_value()` to access the real string.

### The singleton pattern

```python
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()   # reads .env once
    return _settings
```

Every module calls `get_settings()`.  The `.env` file is only read on the
first call.  All subsequent calls return the same object — this is called
the **singleton pattern**.

### Validation

```python
@model_validator(mode="after")
def _validate_backend(self) -> Settings:
    if self.parser_backend == "cloud":
        if self.z_ai_api_key is None:
            raise ValueError("Z_AI_API_KEY is required when PARSER_BACKEND=cloud")
    elif self.parser_backend == "ollama":
        if self.config_yaml_path == "config.yaml":
            self.config_yaml_path = "ollama/config.yaml"  # auto-set
```

This runs after all fields are populated.  It enforces that you can't run
in cloud mode without an API key, and automatically switches the config file
path when using Ollama.

---

## 3. Document Parsing Pipeline

**File:** [src/doc_parser/pipeline.py](../src/doc_parser/pipeline.py)

### The dataclasses

```
ParsedElement         ← one detected element (a paragraph, table row, figure, etc.)
  label               "paragraph", "table", "image", "formula", ...
  text                recognized text content
  bbox                [x1, y1, x2, y2] in 0–1000 normalised coords
  score               detection confidence (GLM-OCR always returns 1.0)
  reading_order       int — sequence position for correct left-to-right, top-to-bottom order

PageResult            ← all elements on one PDF page
  page_num            1-based
  elements            list[ParsedElement]
  markdown            page text assembled as Markdown

ParseResult           ← entire document
  source_file         "report.pdf"
  pages               list[PageResult]
  total_elements      sum of elements across all pages
  full_markdown       full document Markdown from SDK (higher quality than per-page assembly)
```

### What GLM-OCR returns

The glmocr SDK's `GlmOcr.parse()` returns a `PipelineResult` with:
- `json_result`: a list of lists — `[page][element]`, each element is a dict:
  ```json
  {"index": 0, "label": "paragraph", "content": "text here", "bbox_2d": [10, 20, 500, 80]}
  ```
- `markdown_result`: one big Markdown string for the whole document.

### The bbox coordinate system

**bbox_2d values are NOT pixels.**  They are normalised to a 0–1000 scale.
This means `bbox_2d = [100, 200, 600, 400]` means:
- left edge = 10% from left (100/1000)
- top edge = 20% from top (200/1000)
- right edge = 60% from right (600/1000)
- bottom edge = 40% from top (400/1000)

To convert to pixels when you have the rendered image:
```python
pixel_x = bbox_value * image_width_px / 1000
pixel_y = bbox_value * image_height_px / 1000
```

### Why separate cloud vs Ollama parsing logic

```python
if settings.parser_backend == "cloud":
    parse_kwargs["start_page_id"] = 0
    parse_kwargs["end_page_id"] = total_pages - 1
```

The cloud SDK defaults to parsing only page 1 if no page range is given.
For a 50-page PDF this would silently skip 49 pages — a serious bug.

```python
if settings.parser_backend == "ollama":
    parse_kwargs["save_layout_visualization"] = False
```

In Ollama mode we turn off saving debug visualization images because:
1. They accumulate in the current directory.
2. The Ollama SDK ignores start/end page ids — it uses its own pypdfium2
   loader internally, which may count pages slightly differently to PyMuPDF.

### `from_sdk_result` classmethod

```python
@classmethod
def from_sdk_result(cls, raw, source_file) -> ParseResult:
    raw_pages = getattr(raw, "json_result", [])
    full_markdown = getattr(raw, "markdown_result", "") or ""

    for page_idx, raw_elements in enumerate(raw_pages):
        elements = [
            ParsedElement(
                label=el.get("label", "paragraph"),
                text=el.get("content", ""),
                bbox=[float(v) for v in el.get("bbox_2d", [0,0,1,1])],
                score=1.0,
                reading_order=el.get("index", len(elements)),
            )
            for el in raw_elements
        ]
        ...
```

`getattr(raw, "json_result", [])` instead of `raw.json_result` is defensive
programming — if the SDK changes its response format, this returns an empty
list rather than crashing with AttributeError.

---

## 4. Post-Processor

**File:** [src/doc_parser/post_processor.py](../src/doc_parser/post_processor.py)

### The Protocol

```python
@runtime_checkable
class ElementLike(Protocol):
    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int
```

A **Protocol** is Python's duck-typing mechanism.  Any object that has these
attributes satisfies `ElementLike`, without inheriting from it.  This means
tests can pass simple `SimpleNamespace` objects instead of real `ParsedElement`
instances.  `@runtime_checkable` allows `isinstance(obj, ElementLike)` checks.

### Markdown assembly

```python
PROMPT_MAP = {
    "document_title":  lambda t: f"# {t}",        # H1 heading
    "paragraph_title": lambda t: f"## {t}",        # H2 heading
    "formula":         lambda t: f"\n$$\n{t}\n$$\n",  # LaTeX block
    "code_block":      lambda t: f"```\n{t}\n```",
    ...
}

def assemble_markdown(elements):
    sorted_elements = sorted(elements, key=lambda e: e.reading_order)
    parts = []
    for element in sorted_elements:
        if element.label in SKIP_LABELS:
            continue                              # skip images, page numbers, seals
        transform = PROMPT_MAP.get(element.label)
        parts.append(transform(element.text) if transform else element.text)
    return "\n\n".join(parts).strip()
```

Elements are sorted by `reading_order` (the `index` from GLM-OCR), which
represents the correct reading sequence — left column before right column,
top before bottom, caption before or after its figure.

### Why skip images in Markdown?

Images have no text to put in Markdown (their text content is usually empty
or "[figure]").  They are handled separately in the chunker as atomic chunks
that receive GPT-4o captions during enrichment.

---

## 5. Structure-Aware Chunker

**File:** [src/doc_parser/chunker.py](../src/doc_parser/chunker.py)

### Why chunking matters

Language models have context length limits.  A 50-page PDF might have 100,000
tokens.  We can only send a few thousand tokens to GPT-4o as context.
Chunking splits the document into pieces that fit in context and can be
retrieved individually.

### The Chunk dataclass

```python
@dataclass
class Chunk:
    text: str           # content that gets embedded (or AI caption for images)
    chunk_id: str       # "report.pdf_3_12" = page 3, chunk index 12
    page: int           # which page this chunk came from
    element_types: list[str]   # ["paragraph_title", "paragraph"]
    bbox: list[float] | None   # for atomic elements; None for merged text chunks
    source_file: str
    is_atomic: bool     # True for tables, formulas, images
    modality: str       # "text" | "image" | "table" | "formula" | "algorithm"
    image_base64: str | None   # set by image_captioner for images
    caption: str | None        # short caption for images/tables
```

### Three types of elements

**Atomic elements** (table, formula, image, figure, algorithm):
Each always gets its own dedicated chunk.  They are never merged with other
elements, and never split.  A table that exceeds max_chunk_tokens is still
one chunk — splitting it would destroy its tabular structure.

**Title elements** (document_title, paragraph_title, figure_title):
Titles do not become standalone chunks if they can be avoided.  They "attach
forward" — meaning they are prepended to the next content element's chunk.
This keeps headings with their content, which is critical for retrieval
quality (you want to retrieve "# Introduction\n\nThis paper proposes..." not
just "This paper proposes...").

**`figure_title`** is special: it is the figure caption label from PP-DocLayout-V3.
It attaches not to the next text element but to the next image/figure atomic
chunk.  This co-locates the caption text with the figure's visual description
in one chunk.

**Regular content** (paragraph, text, abstract, reference, etc.):
Accumulates into chunks up to `max_chunk_tokens`.  When a new element would
exceed the limit, the current accumulator is flushed and a new chunk starts.

### Token estimation

```python
_TOKEN_WORD_RATIO: float = 1.3

def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOKEN_WORD_RATIO)
```

Word count multiplied by 1.3 is a fast BPE token approximation for English.
Actual subword tokenizers (tiktoken cl100k_base) produce ~1.2–1.4 tokens
per word depending on vocabulary coverage.  This avoids a tiktoken import
which would require downloading a vocabulary file.

### Cross-page chunking (`document_aware_chunking`)

```python
def document_aware_chunking(pages, source_file, max_chunk_tokens=512):
    # Flatten all pages into one stream
    all_pairs = [(page_num, el) for page_num, elements in pages for el in elements]
    all_pairs.sort(key=lambda x: (x[0], x[1].reading_order))
    ...
```

By flattening across pages and processing in one pass, a section heading on
the last line of page 5 correctly attaches to the paragraph that starts at
the top of page 6.  With per-page chunking this heading would become an
orphan chunk with no content, wasting an embedding and causing poor retrieval.

### The accumulator + flush pattern

```python
current_texts: list[str] = []   # text segments accumulating
current_labels: list[str] = []  # their labels
current_tokens: int = 0         # running token count

def flush_current():
    if not current_texts and pending_title is None:
        return
    # build Chunk from accumulated state, reset accumulators
    ...
```

This is a classic "buffer and flush" pattern.  The buffer (`current_texts`)
fills up as elements are processed.  When it hits the token limit, or a
heading signals a section break, it is flushed to produce a Chunk object.

---

## 6. Ingestion — Embeddings

**File:** [src/doc_parser/ingestion/embedder.py](../src/doc_parser/ingestion/embedder.py)

### Dense embeddings (semantic meaning)

```python
async def embed_texts(texts, client, model, dimensions, batch_size=100):
    sanitised = [t if t.strip() else "[empty]" for t in texts]
    all_embeddings = []
    for i in range(0, len(sanitised), batch_size):
        batch = sanitised[i : i + batch_size]
        response = await client.embeddings.create(model=model, input=batch, dimensions=dimensions)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings
```

`text-embedding-3-large` produces a 3072-dimensional vector for each text
input.  Vectors that are "close" in this 3072-dimensional space have similar
meaning.  This is how "What is the accuracy of the model?" can retrieve a
chunk that says "The system achieves 94.7% precision on the test set."

Batching is necessary because the API has a maximum of 2048 inputs per call.

### Sparse embeddings (keyword matching — BM25 proxy)

```python
def compute_sparse_vectors(texts, n_features=2**17):
    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())  # tokenize
        tf = Counter(tokens)                              # term frequencies
        total_terms = len(tokens)

        bucket_weights = {}
        for term, count in tf.items():
            idx = abs(hash(term)) % n_features           # hash to bucket
            bucket_weights[idx] = count / total_terms    # normalised TF

        sorted_items = sorted(bucket_weights.items())
        vectors.append(SparseVector(indices=[i for i,_ in sorted_items],
                                    values=[v for _,v in sorted_items]))
```

Sparse vectors represent documents as "bags of words" — each word maps to a
position (index) and its frequency is its value.  Most positions are 0 (hence
"sparse").  This enables **keyword matching**: searching "BM25" will find
chunks that literally contain "BM25" even if no semantically similar text
exists in the embedding space.

**Feature hashing trick:**  Instead of building a vocabulary (which requires
seeing all documents first), we hash each word to a bucket in a fixed-size
array.  n_features=2^17=131,072 gives enough buckets that collisions are rare.

### Why BOTH dense and sparse?

- Dense alone: "What is the accuracy?" finds the right section. But searching
  for "table 3" or "equation 7" fails — numbers and proper nouns don't embed well.
- Sparse alone: Exact keyword matching works for technical terms, but "neural
  network performance" doesn't find "deep learning metrics."
- **Hybrid (RRF):** Gets the best of both worlds.

### The embedder ABC

```python
class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class OpenAIEmbedder(BaseEmbedder): ...
class GeminiEmbedder(BaseEmbedder): ...
```

The Abstract Base Class (`ABC`) defines the interface.  Any code that
uses an embedder only depends on `BaseEmbedder`, not on the specific
provider.  Swapping OpenAI → Gemini requires only changing EMBEDDING_PROVIDER
in `.env` — no code changes.

---

## 7. Ingestion — Multimodal Enrichment

**File:** [src/doc_parser/ingestion/image_captioner.py](../src/doc_parser/ingestion/image_captioner.py)

### The problem

After chunking, an image chunk has `text="[figure]"` because GLM-OCR
recognizes text *inside* images but doesn't describe the image itself.
Embedding "[figure]" produces a useless vector — no query will ever find it.

### The solution: GPT-4o as a multimodal captioner

For **image chunks:**
```python
# 1. Render the page at 150 DPI
page_img = pdf_page_to_image(pdf_path, chunk.page - 1, dpi=150)
w, h = page_img.size

# 2. Crop the exact bounding box region
x1 = int(bbox[0] * w / 1000)   # convert from 0–1000 scale to pixels
y1 = int(bbox[1] * h / 1000)
x2 = int(bbox[2] * w / 1000)
y2 = int(bbox[3] * h / 1000)
crop = page_img.crop((x1, y1, x2, y2))

# 3. Encode as base64 PNG
buf = io.BytesIO()
crop.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

# 4. Send to GPT-4o vision API
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": _IMAGE_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image_url",
                                       "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
    ],
)
```

The structured system prompt asks GPT-4o to respond in exactly:
```
CAPTION: <1-2 sentence description>
FLOW: <numbered steps of the process depicted>
STRUCTURE: <containment and grouping relationships>
```

After enrichment:
- `chunk.text` = the full structured description (this is what gets embedded)
- `chunk.caption` = just the CAPTION line (shown in UI responses)
- `chunk.image_base64` = the base64 PNG (passed to the reranker for visual scoring)

For **tables, formulas, and algorithms:** similar approach but text-only API calls.

### Concurrency control

```python
semaphore = asyncio.Semaphore(max_concurrent=5)

async def _enrich_image_single(chunk, pdf_path, client, semaphore, model):
    async with semaphore:   # at most 5 concurrent enrichments
        ...

tasks = [_enrich_image_single(chunk, ...) for chunk in image_chunks]
await asyncio.gather(*tasks)
```

`asyncio.Semaphore(5)` ensures at most 5 concurrent GPT-4o API calls,
avoiding rate limit errors while still processing enrichments in parallel.

---

## 8. Ingestion — Vector Store

**File:** [src/doc_parser/ingestion/vector_store.py](../src/doc_parser/ingestion/vector_store.py)

### Qdrant collection structure

```python
await client.create_collection(
    collection_name=self._collection,
    vectors_config={
        "text_dense": VectorParams(
            size=3072,           # OpenAI text-embedding-3-large dimensions
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
    },
    sparse_vectors_config={
        "bm25_sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)  # keep in RAM for speed
        )
    },
)
```

The collection holds two separate vector spaces for each point.  When
searching, Qdrant runs both searches independently then fuses the results.

### Deterministic Point IDs

```python
id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
```

`uuid5` generates a deterministic UUID from the chunk_id string.  If you
re-ingest the same document, the same chunks get the same IDs → **upsert
(not insert)**: existing points are updated in place rather than duplicated.

### Hybrid search with RRF

```python
results = await client.query_points(
    collection_name=self._collection,
    prefetch=[
        Prefetch(query=query_dense, using="text_dense", limit=top_k * 2),   # dense candidates
        Prefetch(query=query_sparse, using="bm25_sparse", limit=top_k * 2), # sparse candidates
    ],
    query=FusionQuery(fusion=Fusion.RRF),   # combine with Reciprocal Rank Fusion
    limit=top_k,
)
```

**Reciprocal Rank Fusion (RRF):**  Each result gets a score of `1/(rank + k)`.
A chunk that ranks #1 in dense AND #1 in sparse gets a very high RRF score.
A chunk that ranks #50 in dense but #1 in sparse still gets a decent score.
RRF is robust and doesn't require tuning weights between dense and sparse.

---

## 9. Retrieval — Reranking

**File:** [src/doc_parser/retrieval/reranker.py](../src/doc_parser/retrieval/reranker.py)

### Why rerank after vector search?

Vector search (ANN — Approximate Nearest Neighbour) is optimised for speed,
not perfect precision.  It retrieves the top-20 **approximate** neighbours.
A reranker does an exhaustive cross-attention comparison between the query
and each of the 20 candidates, giving much more accurate relevance scores.

The tradeoff: vector search is O(log N) over millions of documents; reranking
is O(N_candidates) but only over the 20 candidates already retrieved.

### OpenAI reranker — async cross-encoder

```python
async def _score_one(self, query, candidate):
    prompt = f"Rate relevance 1-10, reply ONLY with the integer.\n\nQuery: {query}\n\nDocument: {text}"
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4,       # just enough for "10"
    )
    return float(response.choices[0].message.content.strip())

# All 20 candidates scored IN PARALLEL
scores = await asyncio.gather(*[self._score_one(query, c) for c in candidates])
```

GPT-4o-mini understands context deeply and can correctly judge "this chunk
discusses something related to the query" vs "this chunk just shares a keyword."

### Jina Reranker M0 — true multimodal reranker

```python
documents = [
    {"text": c["text"], "images": [c["image_base64"]]}  # image chunks
    if c["modality"] == "image" else
    {"text": c["text"]}                                   # text chunks
    for c in candidates
]
payload = {"model": "jina-reranker-m0", "query": query, "documents": documents, "top_n": top_n}
response = await httpx_client.post("https://api.jina.ai/v1/rerank", json=payload)
```

Jina M0 is a proper cross-modal model trained to score (query, image) and
(query, text) relevance jointly.  This means a query like "architecture
diagram" can correctly rank an image chunk above a text chunk that just
says "the architecture is described in Figure 3."

### BGE reranker — local, text-only, very fast

```python
from FlagEmbedding import LayerWiseFlagLLMReranker
reranker = LayerWiseFlagLLMReranker("BAAI/bge-reranker-v2-minicpm-layerwise",
                                     use_fp16=True, cutoff_layers=[28])
pairs = [[query, chunk_text] for c in candidates]
scores = reranker.compute_score(pairs, cutoff_layers=[28])
```

BGE is a transformer reranker that runs entirely locally.  `cutoff_layers=[28]`
means it exits the transformer at layer 28 (of ~32 total), sacrificing a tiny
bit of quality for ~30% speed improvement.  Runs synchronously so it's
offloaded to a thread pool with `run_in_executor`.

### Qwen3-VL-Reranker-2B — local, multimodal

Ranked #1 on MMEB-V2 (multimodal retrieval benchmark).  Accepts raw PIL images
directly alongside text, giving it true visual understanding without relying
on AI-generated captions.

---

## 10. FastAPI REST API

**Files:** [src/doc_parser/api/](../src/doc_parser/api/)

### Request flow

```
HTTP POST /generate
  → middleware logs request start
  → route handler:
      1. get_store() → QdrantDocumentStore (lazy singleton)
      2. get_embedder_dep() → OpenAIEmbedder (lazy singleton)
      3. get_reranker_dep() → JinaReranker (lazy singleton)
      4. store.search(query, embedder, top_k=20) → 20 candidates
      5. reranker.rerank(query, candidates, top_n=5) → top 5
      6. build context = "\n\n".join(f"[page {p}] {text}" for each)
      7. openai.chat.completions.create(GPT-4o, context + question)
      8. return GenerateResponse(answer, sources)
  → middleware logs status + latency
```

### Lazy singletons in dependencies.py

```python
_store: QdrantDocumentStore | None = None

def get_store() -> QdrantDocumentStore:
    global _store
    if _store is None:
        _store = QdrantDocumentStore(get_settings())
    return _store
```

The store, embedder, and reranker are created once on first request (not at
import time).  This means the FastAPI app starts up instantly — it doesn't
try to connect to Qdrant or load model weights until the first request arrives.

### The /ingest endpoint

```python
@router.post("")
async def ingest(file: UploadFile = File(...)):
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    parser = DocumentParser()
    parse_result = parser.parse_file(tmp_path)
    chunks = [...]   # chunking
    await enrich_chunks(chunks, tmp_path, client)
    dense, sparse = await embed_chunks(chunks, embedder, settings)
    await store.upsert_chunks(chunks, dense, sparse)
    return IngestResponse(chunks_upserted=len(chunks), ...)
```

The file is written to a temp file because `DocumentParser.parse_file` needs
a real filesystem path (the glmocr SDK opens the file directly).

### Pydantic schemas enforce contract

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    top_n: int | None = None
    rerank: bool = True
    filter_modality: str | None = None
```

FastAPI automatically validates the request body against `SearchRequest`.
If `query` is missing or `top_k` is not an integer, FastAPI returns a 422
error with a detailed explanation — before your handler even runs.

---

## 11. CLI Scripts

**Files:** [scripts/](../scripts/)

### scripts/ingest.py — the main ingestion entry point

```
python scripts/ingest.py path/to/docs/ --overwrite --collection my-docs
```

Five-step pipeline for each file:
1. **Parse** — `DocumentParser.parse_file(file_path)` → `ParseResult`
2. **Chunk** — `structure_aware_chunking(page.elements, ...)` for each page
3. **Enrich** — `enrich_chunks(chunks, pdf_path, openai_client)` (GPT-4o)
4. **Embed** — `embed_chunks(chunks, embedder, settings)` → dense + sparse
5. **Upsert** — `store.upsert_chunks(chunks, dense, sparse)`

Rich progress bars show which step is running and how long it has been.

### scripts/search.py — query from the terminal

```
python scripts/search.py "what is the architecture of the model?" --top-n 3
```

Embeds the query, searches Qdrant, reranks, prints a rich table with source
file, page number, modality, rerank score, and a text excerpt.

### scripts/serve.py — start the API

```
python scripts/serve.py
```

Launches uvicorn with host/port from Settings.  The FastAPI docs UI is then
available at http://localhost:8000/docs.

---

## 12. Ollama Self-Hosted Mode

**Files:** [ollama/](../ollama/)

### The difference from cloud mode

In cloud mode, GLM-OCR runs on Z.AI's servers.  You send a PDF, they process
it and return JSON.  This requires an API key and internet access.

In Ollama mode, all processing runs on your machine.  GLM-OCR models are
pulled via Ollama and served locally.  No API key, no internet required after
setup.

### ollama/config.yaml

```yaml
maas:
  enabled: false        # disables cloud mode
ollama:
  endpoint: http://localhost:11434
```

This is the config file the glmocr SDK reads.  When PARSER_BACKEND=ollama,
`Settings._validate_backend` auto-sets `config_yaml_path = "ollama/config.yaml"`.

### Critical: don't pass api_key in Ollama mode

```python
api_key = (
    settings.z_ai_api_key.get_secret_value()
    if settings.parser_backend == "cloud" and settings.z_ai_api_key
    else None
)
self._parser = GlmOcr(config_path=settings.config_yaml_path, api_key=api_key)
```

The glmocr SDK treats ANY non-None api_key as a signal to use cloud mode,
even if the config says `maas.enabled: false`.  Passing `api_key=None` lets
the config file take precedence.

### ollama/visualize.py

Renders parse results visually to validate that Ollama mode is detecting
elements in the same locations as cloud mode.  This is a debugging tool to
compare detection quality between the two backends.

---

## 13. Streamlit Visualizer

**File:** [app.py](../app.py)

### What it does

Upload any PDF → click "Parse Document" → the app calls GLM-OCR and shows
you every detected element as a colored bounding box overlaid on the page.

### How bbox rendering works

```python
RENDER_DPI = 150
BBOX_SCALE = 1000   # GLM-OCR normalises to 0–1000

def draw_bboxes(img, elements):
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for el in elements:
        color = get_color(el.label)
        x1 = int(el.bbox[0] * w / BBOX_SCALE)   # scale from 0–1000 to pixels
        y1 = int(el.bbox[1] * h / BBOX_SCALE)
        x2 = int(el.bbox[2] * w / BBOX_SCALE)
        y2 = int(el.bbox[3] * h / BBOX_SCALE)

        draw.rectangle([x1, y1, x2, y2],
                       fill=(*color, 35),        # semi-transparent fill
                       outline=(*color, 220),    # solid outline
                       width=2)
        # Small badge: label text in a colored pill in the top-left corner
        draw.text((x1+3, y1-18), el.label.replace("_", " "), fill=(255,255,255))
```

### Session state prevents re-parsing

```python
if st.session_state.uploaded_filename != uploaded.name:
    st.session_state.result = None   # reset only on NEW file
```

Streamlit reruns the whole script on every interaction (slider move, checkbox
toggle).  Without session state caching, it would re-call GLM-OCR (30+ seconds)
every time you drag the page slider.  The session state check ensures the API
is only called once per uploaded file.

---

## 14. Data Flow — End to End

Here is a complete trace of what happens when you ingest a PDF and then ask
a question about it.

### Ingestion trace (scripts/ingest.py)

```
report.pdf (50 pages, 200 figures, 15 tables)
│
├─ DocumentParser.parse_file(report.pdf)
│   ├─ fitz.open() → 50 pages counted
│   ├─ GlmOcr.parse(start_page_id=0, end_page_id=49)
│   │   └─ Z.AI cloud API processes all 50 pages
│   └─ ParseResult.from_sdk_result(raw)
│       ├─ 50 PageResult objects
│       └─ 1,847 ParsedElement objects total
│
├─ structure_aware_chunking (per page)
│   ├─ Atomic: 200 image chunks + 15 table chunks + 12 formula chunks
│   ├─ Text:   320 text/paragraph chunks (some span heading + content)
│   └─ Total: ~547 Chunk objects
│
├─ enrich_chunks (GPT-4o, 5 concurrent)
│   ├─ 200 image chunks → GPT-4o vision → CAPTION/FLOW/STRUCTURE descriptions
│   ├─ 15 table chunks → SUMMARY/DETAIL descriptions
│   ├─ 12 formula chunks → plain-English meaning + symbol definitions
│   └─ 320 text chunks → unchanged
│
├─ embed_chunks
│   ├─ 547 texts → OpenAI text-embedding-3-large → 547 × 3072-dim vectors
│   └─ 547 texts → feature-hashing BM25 → 547 SparseVector objects
│
└─ store.upsert_chunks (64 per batch → 9 batches)
    └─ Qdrant: 547 points with text_dense + bm25_sparse vectors + full payload
```

### Query trace (POST /generate)

```
Question: "What performance improvements does the model achieve on Table 3?"
│
├─ embed query → 1 × 3072-dim dense vector + 1 sparse vector
│
├─ Qdrant hybrid search (top_k=20)
│   ├─ Dense prefetch: top 40 nearest dense neighbours
│   ├─ Sparse prefetch: top 40 keyword-matching results for "performance", "table", "3"
│   └─ RRF fusion → top 20 candidates
│       (this likely includes the table chunk from page with Table 3,
│        and paragraphs discussing the results)
│
├─ JinaReranker.rerank(query, 20 candidates, top_n=5)
│   └─ Jina M0 API → 5 reranked results (Table 3 chunk likely ranked #1)
│
├─ Build context:
│   "[page 12] SUMMARY: Table 3 compares our method with baselines on CIFAR-10..."
│   "[page 11] The proposed approach achieves 94.7% top-1 accuracy..."
│   "[page 12] ## Results on Image Classification..."
│   ...
│
└─ GPT-4o (gpt-4o):
    System: "Answer using ONLY the provided context. Cite page numbers."
    User:   "Context: {context}\n\nQuestion: {question}"
    →
    Answer: "According to Table 3 (page 12), the proposed model achieves 94.7%
             top-1 accuracy on CIFAR-10, outperforming the baseline by 3.2
             percentage points and the previous state-of-the-art by 1.1 points."
```

---

## Key Design Decisions Summary

| Decision | Why |
|---|---|
| GLM-OCR for parsing | Returns structured element types + bounding boxes, not just text. This enables modality-aware chunking and visual retrieval. |
| 0–1000 bbox scale | SDK-normalised coordinates that are resolution-independent. Always multiply by `image_width/1000` to get pixels. |
| Hybrid dense+sparse search | Dense catches semantic similarity; sparse catches exact keywords like "Table 3" or "Algorithm 1". Neither alone is sufficient. |
| RRF fusion | Parameter-free method for combining dense and sparse rankings. Robust to score scale differences. |
| GPT-4o enrichment before embedding | An unenriched image chunk embeds as "[figure]" — useless. After enrichment it embeds as a rich description that can be retrieved semantically. |
| Atomic chunks for tables/formulas | Splitting a table across chunks destroys its structure. Atomic chunks ensure tables are always retrieved (and reranked) as a unit. |
| figure_title joined to image chunk | Caption and figure must be co-located. A caption-only chunk tells you what the figure shows; the figure-only chunk shows it. Together they enable both text and visual retrieval. |
| Pluggable reranker backends | Cloud (Jina), API (OpenAI), local-text (BGE), local-multimodal (Qwen) — different cost/latency/quality tradeoffs for different deployment scenarios. |
| Pydantic v2 settings with SecretStr | Prevents API keys from appearing in repr(), logs, or error messages. |
| Deterministic UUIDs for Qdrant points | Re-ingesting the same document updates existing points rather than creating duplicates. |
