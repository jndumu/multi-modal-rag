#!/usr/bin/env bash
# deployment/setup_ollama.sh — One-time setup for Ollama + GLM-OCR local pipeline
# Run this once before using the Ollama backend for the first time.
# Usage: bash deployment/setup_ollama.sh

set -e

echo "============================================================"
echo " GLM-OCR via Ollama — One-Time Setup"
echo "============================================================"
echo ""

# ── Check Ollama installed ────────────────────────────────────────
echo "[1/4] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
  echo "      ✓ Ollama installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
  echo "      ✗ Ollama not found."
  echo ""
  echo "      Install Ollama:"
  echo "        macOS:  brew install ollama"
  echo "        Linux:  curl -fsSL https://ollama.com/install.sh | sh"
  echo "        Win:    https://ollama.com/download"
  echo ""
  exit 1
fi

# ── Pull GLM-OCR model ────────────────────────────────────────────
echo ""
echo "[2/4] Pulling GLM-OCR model (glm-ocr:latest, ~600 MB)..."
if ollama list 2>/dev/null | grep -q "glm-ocr"; then
  echo "      ✓ glm-ocr:latest already present"
else
  ollama pull glm-ocr:latest
  echo "      ✓ glm-ocr:latest downloaded"
fi

# ── Install layout detection deps ─────────────────────────────────
echo ""
echo "[3/4] Installing layout detection dependencies..."
echo "      (torch, torchvision, transformers, sentencepiece, accelerate, opencv-python)"
uv pip install "glmocr[layout]"
echo "      ✓ Layout deps installed"

# ── Verify imports ────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying Python imports..."

uv run python -c "import glmocr; print('      ✓ glmocr')"
uv run python -c "import torch; print(f'      ✓ torch {torch.__version__}')"
uv run python -c "import transformers; print(f'      ✓ transformers {transformers.__version__}')"
uv run python -c "import cv2; print(f'      ✓ opencv-python {cv2.__version__}')"

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Start Ollama:   ollama serve"
echo "   2. Start Qdrant:   docker compose up -d qdrant"
echo "   3. Test parsing:   uv run python ollama/test_parse.py data/raw/test_page1.pdf"
echo ""
echo " Or run everything at once:"
echo "   bash deployment/start.sh"
echo "============================================================"
