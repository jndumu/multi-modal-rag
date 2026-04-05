# Prompt 01 — Initial Project Setup

## Stage
Project bootstrap: define scope, choose tooling, scaffold the package.

---

## Prompt

I want to build a multimodal RAG (Retrieval-Augmented Generation) system that can
ingest PDF documents, understand them visually (not just as plain text), and answer
questions about them with a language model.

The parsing backend I want to use is GLM-OCR, specifically the `glmocr` Python SDK
that talks to the Z.AI MaaS (Model-as-a-Service) cloud API.  GLM-OCR uses
PP-DocLayout-V3 for layout detection and GLM-OCR 0.9B for recognition — it returns
structured JSON with element labels (paragraph, table, formula, image, etc.) and
bounding box coordinates.

Requirements:
- Python 3.12, managed with `uv` (not pip/poetry).
- Package name: `doc-parser`, installable as `uv pip install -e .`.
- Core runtime deps: glmocr, PyMuPDF (fitz), Pillow, pyyaml, pydantic v2,
  pydantic-settings, python-dotenv, rich, tqdm, streamlit, openai, qdrant-client,
  tiktoken, httpx, fastapi, uvicorn[standard], loguru, python-multipart.
- Optional extras:
    - `bge`    → FlagEmbedding (local BGE reranker)
    - `qwen`   → transformers + torch (local Qwen VL reranker)
    - `gemini` → google-genai (Gemini embedding provider)
    - `layout` → glmocr[layout] + torch + torchvision + transformers + sentencepiece
                  + accelerate + opencv-python (for running layout detection locally)
- Dev extras: pytest, pytest-asyncio (asyncio_mode=auto), ruff, mypy.
- Ruff: target py312, line-length 100, select E/W/F/I/B/UP, double quotes.
- All secrets (API keys) go in `.env` — never committed. Provide `.env.example`.
- `config.yaml` for GLM-OCR MaaS mode (needed by the glmocr SDK).
- `docker-compose.yml` that starts a local Qdrant instance on port 6333.
- `.gitignore` that excludes: .env, .env.* (but allows .env.example), .venv/,
  __pycache__/, *.pyc, *.egg-info/, dist/, build/, test caches, output/,
  data/raw/, data/processed/, models/, test_data/, ollama/output/, notebooks
  checkpoints, frontend/, .DS_Store, IDE configs, *.log, *.tmp, CLAUDE.md,
  uv.lock, .claude/.

Scaffold the `src/doc_parser/` package with empty `__init__.py` files for the
subpackages: `api/`, `api/routes/`, `ingestion/`, `retrieval/`, `utils/`.

---

## What this produced

- `pyproject.toml` with all dependencies and tool configs
- `.gitignore` with comprehensive exclusions
- `.env.example` documenting every variable
- `config.yaml` pointing to the Z.AI GLM-OCR MaaS endpoint
- `docker-compose.yml` for local Qdrant
- `src/doc_parser/` package skeleton with subpackage directories
