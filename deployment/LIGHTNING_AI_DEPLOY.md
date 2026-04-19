# Deploying to Lightning AI

This guide walks through deploying the full multimodal RAG pipeline on
[Lightning AI](https://lightning.ai) — GPU-backed cloud compute with persistent
storage, managed studios, and easy team sharing.

---

## What You Will Deploy

| Component | How it runs on Lightning AI |
|---|---|
| Qdrant | Docker container inside a Studio |
| Ollama + GLM-OCR | Studio with GPU (or CPU) |
| FastAPI server | Studio process, exposed via port forwarding |
| Streamlit UI | Studio process, exposed via port forwarding |

---

## Prerequisites

1. **Lightning AI account** — sign up at https://lightning.ai
2. **Lightning CLI** installed locally:
   ```bash
   pip install lightning
   lightning login
   ```
3. Your project code pushed to GitHub (`main` branch):
   ```bash
   git push origin main
   ```

---

## Step 1 — Create a Lightning Studio

1. Go to https://lightning.ai and click **New Studio**
2. Choose a **GPU machine** (recommended: L4 or A10G for GLM-OCR + layout detection)
   - Minimum: CPU-only (slower but works)
   - Recommended: L4 GPU ($0.70/hr) for real-time parsing
3. Name it: `multimodal-rag`
4. Click **Create**

---

## Step 2 — Clone the Repository Inside the Studio

In the Studio terminal:

```bash
git clone https://github.com/jndumu/multi-modal-rag.git
cd multi-modal-rag
```

---

## Step 3 — Install Dependencies

```bash
# Install uv
pip install uv

# Create venv and install all packages
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install layout detection deps (PP-DocLayout-V3)
uv pip install "glmocr[layout]"
```

> **GPU note:** `torch` will auto-detect the Studio GPU. No extra flags needed.

---

## Step 4 — Install and Start Ollama

Lightning Studios support `apt` and `curl` — install Ollama the standard way:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama in the background:

```bash
ollama serve &
sleep 3

# Pull the GLM-OCR model (~600 MB, downloaded once and cached)
ollama pull glm-ocr:latest

# Verify
ollama list
```

---

## Step 5 — Install and Start Qdrant

Install Docker in the Studio (if not already present):

```bash
# Check if Docker is available
docker --version

# If not, install (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
```

Start Qdrant:

```bash
docker compose up -d qdrant

# Verify
curl http://localhost:6333
```

> **Alternative (no Docker):** Run Qdrant as a binary directly:
> ```bash
> wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
> tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
> ./qdrant &
> ```

---

## Step 6 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```dotenv
# Parser backend
PARSER_BACKEND=ollama           # Use local Ollama (no Z_AI_API_KEY needed)
# PARSER_BACKEND=cloud          # Use Z.AI cloud API (faster)
# Z_AI_API_KEY=your-key-here   # Only needed for cloud mode

# OpenAI — required for embeddings and image captions
OPENAI_API_KEY=sk-...

# Qdrant — running locally in the Studio
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                 # Leave empty for local Qdrant

# Reranker
RERANKER_BACKEND=openai         # openai | jina | bge | qwen
```

---

## Step 7 — Start the FastAPI Server

```bash
uv run uvicorn doc_parser.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Lightning Studios automatically expose ports. To access the API:
1. In the Studio sidebar, click **Ports**
2. Find port `8000` and click **Open in browser**
3. Append `/docs` to the URL for the Swagger UI

---

## Step 8 — Start the Streamlit UI (Optional)

In a second terminal tab:

```bash
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Expose via port `8501` in the Studio ports panel.

---

## Step 9 — One-Shot Startup Script

Save this as a Studio startup script so everything starts automatically:

```bash
# In Studio terminal — run once to start everything
bash deployment/start.sh
```

Or manually:

```bash
# Terminal 1 — Qdrant
docker compose up -d qdrant

# Terminal 2 — Ollama
ollama serve

# Terminal 3 — FastAPI
source .venv/bin/activate
uv run uvicorn doc_parser.api.app:app --host 0.0.0.0 --port 8000

# Terminal 4 (optional) — Streamlit
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Step 10 — Ingest a Document and Test

Upload a PDF via the API or copy one into the Studio filesystem:

```bash
# Ingest a document
uv run python scripts/ingest.py data/raw/test_page1.pdf

# Search
uv run python scripts/search.py "document layout detection"
```

Via the REST API (from your local machine, using the Studio's exposed URL):

```bash
# Replace <studio-url> with the URL from the Ports panel
curl -X POST https://<studio-url>/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/raw/test_page1.pdf"}'
```

---

## Persistent Storage

Lightning AI Studios have `/teamspace/` for persistent storage that survives
Studio restarts. Move your data there:

```bash
# Move raw documents to persistent storage
mkdir -p /teamspace/studios/this_studio/data/raw
cp data/raw/*.pdf /teamspace/studios/this_studio/data/raw/

# Symlink back so scripts find them
ln -s /teamspace/studios/this_studio/data data
```

Qdrant data is stored in a Docker volume (`qdrant_data`). To persist across
restarts, mount it to `/teamspace/`:

In `docker-compose.yml`, change the volume:
```yaml
volumes:
  qdrant_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /teamspace/studios/this_studio/qdrant_storage
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `PARSER_BACKEND` | Yes | `ollama` (local) or `cloud` (Z.AI) |
| `Z_AI_API_KEY` | Cloud only | Z.AI MaaS API key |
| `OPENAI_API_KEY` | Yes | Used for embeddings + image captions |
| `QDRANT_URL` | Yes | `http://localhost:6333` (local Studio) |
| `QDRANT_API_KEY` | No | Leave empty for local Qdrant |
| `RERANKER_BACKEND` | Yes | `openai` \| `jina` \| `bge` \| `qwen` |
| `JINA_API_KEY` | Jina only | Required if `RERANKER_BACKEND=jina` |
| `EMBEDDING_MODEL` | Yes | Default: `text-embedding-3-large` |
| `EMBEDDING_DIMENSIONS` | Yes | Default: `3072` |
| `IMAGE_CAPTION_ENABLED` | No | `true` to caption images with GPT-4o |

---

## GPU Recommendations

| Use Case | Recommended GPU | Notes |
|---|---|---|
| Parsing only (Ollama OCR) | L4 (24 GB) | GLM-OCR 0.9B + PP-DocLayout-V3 |
| Full pipeline (parse + embed + rerank) | A10G (24 GB) | Comfortable for all models |
| Local reranker (BGE or Qwen VL) | L4 or A10G | BGE is CPU-ok; Qwen VL needs GPU |
| CPU-only (dev/testing) | No GPU | Parsing is slow (5–30s/page) |

---

## Troubleshooting

### `CUDA out of memory`
Too many models loaded simultaneously. Options:
- Use `RERANKER_BACKEND=openai` (cloud, no local GPU needed)
- Use `RERANKER_BACKEND=jina` (cloud, no local GPU needed)
- Use a larger GPU (A10G or A100)

### Ollama model not found after Studio restart
Ollama model cache may be in `/tmp`. Pull again:
```bash
ollama pull glm-ocr:latest
```
To persist, move Ollama's model directory to `/teamspace/`:
```bash
export OLLAMA_MODELS=/teamspace/studios/this_studio/ollama_models
ollama pull glm-ocr:latest
```
Add `export OLLAMA_MODELS=...` to `~/.bashrc` to persist across sessions.

### Port not accessible externally
Make sure you are using `--host 0.0.0.0` (not `127.0.0.1`) when starting
FastAPI and Streamlit. Lightning AI only forwards ports bound to `0.0.0.0`.

### `Connection refused` on port 6333
Qdrant container stopped after Studio restart. Run:
```bash
docker compose up -d qdrant
```

---

## Cost Estimate (Lightning AI)

| Resource | Approx. cost |
|---|---|
| L4 GPU Studio (active) | ~$0.70/hr |
| A10G GPU Studio (active) | ~$1.10/hr |
| CPU-only Studio (active) | ~$0.10/hr |
| Storage (teamspace) | ~$0.03/GB/month |

> **Tip:** Pause the Studio when not in use — you are only charged while the Studio is running.
> Qdrant data in Docker volumes is preserved when the Studio is paused.
