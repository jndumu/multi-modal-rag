# Build Prompts — Multimodal RAG System

This folder contains the exact sequence of prompts used to build this project.
Pass them to Claude in order to reconstruct the entire codebase from scratch.

## How to use

1. Start a fresh Claude conversation.
2. Paste the contents of each prompt file in order (01 → 12).
3. After each prompt, let Claude generate the files before moving to the next.
4. Use `CODE_EXPLANATION.md` as a reference to understand what each part does.

## Files

| File | What it builds |
|---|---|
| 01_initial_project_setup.md | pyproject.toml, .gitignore, .env.example, config.yaml, docker-compose.yml |
| 02_config_and_settings.md | config.py (pydantic-settings singleton), logging_config.py |
| 03_document_parsing_pipeline.md | pipeline.py (GLM-OCR wrapper), utils/pdf_utils.py |
| 04_post_processor_and_chunker.md | post_processor.py (Markdown assembly), chunker.py (Chunk dataclass + chunking logic) |
| 05_hybrid_ingestion_pipeline.md | ingestion/embedder.py (dense + BM25 sparse), ingestion/vector_store.py (Qdrant) |
| 06_multimodal_enrichment.md | ingestion/image_captioner.py (GPT-4o captions for images/tables/formulas) |
| 07_reranking_backends.md | retrieval/reranker.py (OpenAI, Jina, BGE, Qwen VL backends) |
| 08_fastapi_rest_api.md | api/ — schemas, middleware, dependencies, all four routes |
| 09_cli_scripts.md | scripts/ — ingest, search, parse, serve, debug_raw |
| 10_ollama_self_hosted.md | ollama/ — local backend config and utilities |
| 11_streamlit_visualizer.md | app.py — PDF upload + bbox visualization UI |
| 12_test_suite.md | tests/ — unit and integration tests for every module |
| CODE_EXPLANATION.md | Full step-by-step explanation of how every file works |

## Key design decisions (quick reference)

- **GLM-OCR** returns bbox_2d in 0–1000 normalised coords (NOT pixels).
  Convert: `pixel = bbox_value * image_dimension / 1000`
- **Hybrid search** = dense (OpenAI embeddings) + sparse (BM25 feature-hashing) fused with RRF.
- **Atomic chunks**: tables, formulas, images are never split across chunks.
- **figure_title** is co-located with its image chunk (caption + visual in one chunk).
- **GPT-4o enrichment** must run BEFORE embedding — images without captions embed as "[figure]".
- **PARSER_BACKEND=ollama**: never pass `api_key` to GlmOcr, and never pass `start_page_id`/`end_page_id`.
- **PARSER_BACKEND=cloud**: always pass `start_page_id=0, end_page_id=N-1` or the SDK only parses page 1.
