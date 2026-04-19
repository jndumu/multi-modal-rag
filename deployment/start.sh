#!/usr/bin/env bash
# deployment/start.sh — Start all services for the multimodal RAG pipeline
# Usage: bash deployment/start.sh [--no-api] [--no-ui]

set -e

START_API=true
START_UI=false

for arg in "$@"; do
  case $arg in
    --no-api) START_API=false ;;
    --ui) START_UI=true ;;
  esac
done

echo "============================================================"
echo " Multimodal RAG — Service Startup"
echo "============================================================"

# ── 1. Qdrant ────────────────────────────────────────────────────
echo ""
echo "[1/3] Starting Qdrant (Docker)..."
docker compose up -d qdrant
sleep 2
if curl -sf http://localhost:6333 > /dev/null; then
  echo "      ✓ Qdrant is up at http://localhost:6333"
else
  echo "      ✗ Qdrant failed to start. Check: docker compose logs qdrant"
  exit 1
fi

# ── 2. Ollama ────────────────────────────────────────────────────
echo ""
echo "[2/3] Checking Ollama..."
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "      ✓ Ollama already running at http://localhost:11434"
else
  echo "      Starting Ollama in background..."
  ollama serve &
  sleep 4
  if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "      ✓ Ollama is up at http://localhost:11434"
  else
    echo "      ✗ Ollama failed to start. Run 'ollama serve' manually."
  fi
fi

# Check GLM-OCR model
echo "      Checking glm-ocr:latest model..."
if ollama list 2>/dev/null | grep -q "glm-ocr"; then
  echo "      ✓ glm-ocr:latest model is present"
else
  echo "      ✗ glm-ocr model not found. Pulling now..."
  ollama pull glm-ocr:latest
fi

# ── 3. FastAPI ───────────────────────────────────────────────────
if [ "$START_API" = true ]; then
  echo ""
  echo "[3/3] Starting FastAPI server..."
  uv run uvicorn doc_parser.api.app:app --host 0.0.0.0 --port 8000 --reload &
  sleep 3
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "      ✓ API is up at http://localhost:8000"
    echo "      ✓ Docs at http://localhost:8000/docs"
  else
    echo "      API starting up... check http://localhost:8000/health in a few seconds"
  fi
fi

# ── 4. Streamlit (optional) ──────────────────────────────────────
if [ "$START_UI" = true ]; then
  echo ""
  echo "[4/4] Starting Streamlit UI..."
  uv run streamlit run app.py &
  echo "      ✓ UI starting at http://localhost:8501"
fi

echo ""
echo "============================================================"
echo " All services started."
echo ""
echo " Qdrant  : http://localhost:6333"
echo " Ollama  : http://localhost:11434"
if [ "$START_API" = true ]; then
echo " API     : http://localhost:8000"
echo " API docs: http://localhost:8000/docs"
fi
if [ "$START_UI" = true ]; then
echo " UI      : http://localhost:8501"
fi
echo ""
echo " To stop Qdrant: docker compose down"
echo " To stop Ollama: pkill ollama"
echo "============================================================"
