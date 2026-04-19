# Deployment Guide — Multimodal RAG System

This folder contains everything needed to deploy and run the full pipeline locally
using **Ollama + GLM-OCR** (no cloud API required) or the **Z.AI cloud API**.

---

## Stack Overview

| Service | Role | Port |
|---|---|---|
| Qdrant | Vector database (stores embeddings) | 6333 |
| Ollama | Local LLM runtime (GLM-OCR model) | 11434 |
| FastAPI | REST API for ingest + search | 8000 |
| Streamlit | UI for PDF parsing + bbox visualization | 8501 |

---

## Quick Start (Local / Ollama Mode)

### Step 1 — Install Prerequisites

**Python 3.12 + uv**
```bash
# Install uv (fast Python package manager)
pip install uv

# Create virtual environment and install all dependencies
uv venv --python 3.12
uv pip install -e ".[dev]"
```

**Docker** (for Qdrant)
- Download and install Docker Desktop from https://www.docker.com/products/docker-desktop/

**Ollama** (for local GLM-OCR)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download installer from https://ollama.com/download
```

---

### Step 2 — Pull the GLM-OCR Model

```bash
ollama pull glm-ocr:latest
```

This downloads the GLM-OCR 0.9B vision model (~600 MB).

Verify it downloaded:
```bash
ollama list
# Should include: glm-ocr:latest
```

---

### Step 3 — Install Layout Detection Dependencies

PP-DocLayout-V3 (document layout detection) requires extra packages:

```bash
uv pip install "glmocr[layout]"
```

This installs: `torch`, `torchvision`, `transformers`, `sentencepiece`, `accelerate`, `opencv-python`

> The PP-DocLayout-V3 model weights (~400 MB) are downloaded automatically from
> HuggingFace Hub on first use and cached in `~/.cache/huggingface/`.

---

### Step 4 — Configure Environment

```bash
cp .env.example .env
```

For **Ollama mode** (no cloud keys needed), set:
```dotenv
PARSER_BACKEND=ollama
OPENAI_API_KEY=sk-...    # Still needed for embeddings + image captions
```

For **Cloud mode** (Z.AI API), set:
```dotenv
PARSER_BACKEND=cloud
Z_AI_API_KEY=...          # Z.AI MaaS API key
OPENAI_API_KEY=sk-...
```

---

### Step 5 — Start Services

Run the startup script for your platform:

```bash
# Linux / macOS
bash deployment/start.sh

# Windows
deployment\start.bat
```

Or start each service manually (see sections below).

---

## Manual Service Startup

### Start Qdrant (Docker)

```bash
docker compose up -d qdrant
```

Verify it is running:
```bash
curl http://localhost:6333
# → {"title":"qdrant - vector search engine","version":"..."}
```

Stop Qdrant:
```bash
docker compose down
```

---

### Start Ollama

```bash
ollama serve
```

Ollama runs at `http://localhost:11434`. Leave this terminal open.

Verify GLM-OCR model is loaded:
```bash
curl http://localhost:11434/api/tags
# → {"models":[{"name":"glm-ocr:latest",...}]}
```

---

### Start the FastAPI Server

```bash
uv run uvicorn doc_parser.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Or using the convenience script:
```bash
uv run python scripts/serve.py --reload
```

API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

### Start the Streamlit UI

```bash
uv run streamlit run app.py
```

Opens at http://localhost:8501

For the Ollama-specific visualizer:
```bash
uv run streamlit run ollama/visualize.py
```

---

## Running the Pipeline

### Parse a Document

```bash
# Ollama backend (local, free)
PARSER_BACKEND=ollama uv run python scripts/parse.py data/raw/document.pdf --chunks

# Cloud backend (Z.AI API)
PARSER_BACKEND=cloud uv run python scripts/parse.py data/raw/document.pdf --chunks
```

### Ingest into Qdrant

```bash
uv run python scripts/ingest.py data/raw/document.pdf
```

### Search

```bash
uv run python scripts/search.py "document layout detection" --top-k 10
```

---

## Verify Everything is Working

```bash
bash deployment/verify.sh
```

Or check manually:
```bash
# 1. Qdrant
curl http://localhost:6333

# 2. Ollama
curl http://localhost:11434/api/tags

# 3. GLM-OCR parsing (quick test)
uv run python ollama/test_parse.py data/raw/test_page1.pdf

# 4. API health
curl http://localhost:8000/health
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Connection refused` on port 6333 | Qdrant not running | `docker compose up -d qdrant` |
| `Connection refused` on port 11434 | Ollama not running | `ollama serve` |
| `model not found` | Model not pulled | `ollama pull glm-ocr:latest` |
| `AttributeError: 'LayoutConfig' object has no attribute 'id2label'` | SDK bug in glmocr 0.1.3 | Set `id2label: null` in `ollama/config.yaml` (already done) |
| `ModuleNotFoundError: No module named 'torch'` | Layout deps missing | `uv pip install "glmocr[layout]"` |
| Slow parsing | First run loads model into VRAM | Subsequent runs are faster; GPU recommended |
| `write operation timed out` | Large PDF to cloud API | Extract one page: `uv run python deployment/extract_page.py doc.pdf 0` |

---

## Cloud vs Ollama Comparison

| Feature | Cloud API (Z.AI) | Ollama (Local) |
|---|---|---|
| API key required | Yes (`Z_AI_API_KEY`) | No |
| Speed | Fast (cloud GPU) | 5–30s per page |
| Privacy | Data sent to Z.AI | Fully local |
| Cost | API credits | Free |
| Layout detection | PP-DocLayout-V3 | PP-DocLayout-V3 (same) |
| OCR model | GLM-OCR 0.9B | GLM-OCR 0.9B (same) |
| Output quality | Equivalent | Equivalent |

---

## Config Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | Qdrant container setup |
| `config.yaml` | Main pipeline config (cloud mode) |
| `ollama/config.yaml` | Ollama-specific config (local mode) |
| `.env` | API keys and runtime settings |
| `.env.example` | Template for `.env` |
