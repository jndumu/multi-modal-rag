@echo off
REM deployment\start.bat — Start all services on Windows
REM Usage: deployment\start.bat

echo ============================================================
echo  Multimodal RAG -- Service Startup (Windows)
echo ============================================================

REM ── 1. Qdrant ────────────────────────────────────────────────────
echo.
echo [1/3] Starting Qdrant (Docker)...
docker compose up -d qdrant
timeout /t 3 /nobreak >nul

curl -sf http://localhost:6333 >nul 2>&1
if %errorlevel%==0 (
    echo       OK  Qdrant is up at http://localhost:6333
) else (
    echo       FAIL  Qdrant failed to start. Check: docker compose logs qdrant
    pause
    exit /b 1
)

REM ── 2. Ollama ────────────────────────────────────────────────────
echo.
echo [2/3] Checking Ollama...
curl -sf http://localhost:11434/api/tags >nul 2>&1
if %errorlevel%==0 (
    echo       OK  Ollama already running at http://localhost:11434
) else (
    echo       Starting Ollama in background...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    curl -sf http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel%==0 (
        echo       OK  Ollama is up at http://localhost:11434
    ) else (
        echo       NOTE: Ollama may still be starting. Check http://localhost:11434 manually.
    )
)

echo       Checking glm-ocr:latest model...
ollama list 2>nul | findstr "glm-ocr" >nul
if %errorlevel%==0 (
    echo       OK  glm-ocr:latest model is present
) else (
    echo       Pulling glm-ocr:latest (this may take a few minutes)...
    ollama pull glm-ocr:latest
)

REM ── 3. FastAPI ───────────────────────────────────────────────────
echo.
echo [3/3] Starting FastAPI server...
start /B uv run uvicorn doc_parser.api.app:app --host 0.0.0.0 --port 8000 --reload
timeout /t 4 /nobreak >nul
echo       API starting at http://localhost:8000
echo       Docs at http://localhost:8000/docs

echo.
echo ============================================================
echo  Services started:
echo.
echo  Qdrant   : http://localhost:6333
echo  Ollama   : http://localhost:11434
echo  API      : http://localhost:8000
echo  API docs : http://localhost:8000/docs
echo.
echo  To stop Qdrant: docker compose down
echo  To stop Ollama: taskkill /IM ollama.exe /F
echo ============================================================
pause
