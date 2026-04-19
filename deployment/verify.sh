#!/usr/bin/env bash
# deployment/verify.sh — Check all services are running correctly
# Usage: bash deployment/verify.sh

PASS=0
FAIL=0

check() {
  local name="$1"
  local cmd="$2"
  if eval "$cmd" > /dev/null 2>&1; then
    echo "  ✓  $name"
    PASS=$((PASS + 1))
  else
    echo "  ✗  $name"
    FAIL=$((FAIL + 1))
  fi
}

echo ""
echo "============================================================"
echo " Multimodal RAG — Service Verification"
echo "============================================================"
echo ""

echo "── Infrastructure ──────────────────────────────────────────"
check "Qdrant accessible (port 6333)"     "curl -sf http://localhost:6333"
check "Ollama accessible (port 11434)"    "curl -sf http://localhost:11434/api/tags"
check "glm-ocr:latest model present"     "ollama list 2>/dev/null | grep -q glm-ocr"
echo ""

echo "── Application ─────────────────────────────────────────────"
check "FastAPI server (port 8000)"        "curl -sf http://localhost:8000/health"
check "API docs accessible"              "curl -sf http://localhost:8000/docs"
echo ""

echo "── Python Environment ───────────────────────────────────────"
check "glmocr installed"                 "uv run python -c 'import glmocr'"
check "qdrant-client installed"          "uv run python -c 'import qdrant_client'"
check "openai installed"                 "uv run python -c 'import openai'"
check "Layout deps (torch)"             "uv run python -c 'import torch'"
check "Layout deps (transformers)"      "uv run python -c 'import transformers'"
echo ""

echo "── Quick Parse Test ─────────────────────────────────────────"
if [ -f "data/raw/test_page1.pdf" ]; then
  check "Ollama GLM-OCR parse (test_page1.pdf)" \
    "PARSER_BACKEND=ollama uv run python scripts/parse.py data/raw/test_page1.pdf --format markdown 2>&1 | grep -q 'page'"
else
  echo "  -  Skipping parse test (data/raw/test_page1.pdf not found)"
fi
echo ""

echo "============================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================================"
echo ""

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
