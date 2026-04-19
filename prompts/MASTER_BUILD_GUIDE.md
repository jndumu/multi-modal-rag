# Master Build Guide — Multimodal RAG System
# How to rebuild this entire project from scratch using Claude

---

## BEFORE YOU START — Read This First

### What we are building
A multimodal RAG (Retrieval-Augmented Generation) system that:
- Ingests PDF documents using GLM-OCR (PP-DocLayout-V3 + GLM-OCR 0.9B)
- Understands text, tables, figures, formulas, and algorithms
- Embeds content with dense (OpenAI) + sparse (BM25) vectors into Qdrant
- Answers questions with GPT-4o, grounded in the document with page citations

### How to use this guide
- Open a **fresh Claude conversation** for each session below
- Paste the **CONTEXT** block first (so Claude knows what already exists)
- Then paste the **PROMPT** block
- Let Claude generate ALL files before starting the next session
- The sessions must be done **in order** — each builds on the previous

### Tools you need
- Python 3.12
- `uv` package manager (`pip install uv`)
- Docker (for local Qdrant)
- Z.AI API key (for GLM-OCR cloud) OR Ollama (for local mode)
- OpenAI API key (for embeddings, enrichment, generation)
- Jina API key (for reranking — free tier available at jina.ai)

---

## SESSION 1 — Project Bootstrap

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
This is session 1 — nothing exists yet. We are starting from scratch.
```

### PROMPT
```
Create the full project skeleton for a Python 3.12 package called doc-parser
managed with uv (not pip or poetry).

Create ALL of these files exactly as specified:

─── pyproject.toml ───────────────────────────────────────────────────────────
[project]
name = "doc-parser"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "glmocr>=0.1.0",
    "pymupdf>=1.27.2",
    "Pillow>=12.1.1",
    "pyyaml>=6.0",
    "pydantic>=2.12.0",
    "pydantic-settings>=2.8.0",
    "python-dotenv>=1.0.0",
    "rich>=14.0.0",
    "tqdm>=4.67.0",
    "streamlit>=1.40.0",
    "openai>=2.24.0",
    "qdrant-client>=1.17.0",
    "tiktoken>=0.9.0",
    "httpx>=0.28.0",
    "fastapi>=0.120.0",
    "uvicorn[standard]>=0.34.0",
    "loguru>=0.7.0",
    "python-multipart>=0.0.20",
]
[project.optional-dependencies]
dev     = ["pytest>=8.4.0", "pytest-asyncio>=0.25.0", "ruff>=0.11.0", "mypy>=1.15.0"]
bge     = ["FlagEmbedding>=1.3.0"]
qwen    = ["transformers>=4.51.0", "torch>=2.7.0"]
gemini  = ["google-genai>=1.0.0"]
layout  = ["glmocr[layout]", "torch>=2.10", "torchvision>=0.25",
           "transformers>=5.3", "sentencepiece>=0.2", "accelerate>=1.13",
           "opencv-python>=4.10.0"]
[tool.ruff]
target-version = "py312"
line-length = 100
[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP"]
[tool.ruff.format]
quote-style = "double"
[tool.pytest.ini_options]
asyncio_mode = "auto"

─── .gitignore ───────────────────────────────────────────────────────────────
Exclude: .env and .env.* (but allow .env.example), .venv/, __pycache__/,
*.pyc, *.pyo, *.pyd, *.egg-info/, dist/, build/, *.so, *.egg,
pip-wheel-metadata/, .mypy_cache/, .ruff_cache/, .pytest_cache/, .coverage,
coverage.xml, htmlcov/, data/raw/, data/processed/, output/, models/,
test_data/, ollama/output/, *.ipynb_checkpoints/, frontend/, .DS_Store,
**/.DS_Store, .idea/, .vscode/, *.swp, *.swo, *~, *.log, output.txt,
*.tmp, CLAUDE.md, uv.lock, docling_report.pdf, .claude/, prompts/

─── .env.example ─────────────────────────────────────────────────────────────
# Parser backend
PARSER_BACKEND=cloud
Z_AI_API_KEY=your-z-ai-key-here

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o

# Embedding
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
GEMINI_API_KEY=

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents

# Reranker
RERANKER_BACKEND=jina
RERANKER_TOP_N=5
JINA_API_KEY=

# Features
IMAGE_CAPTION_ENABLED=true

# API server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Logging
LOG_LEVEL=INFO
LOG_JSON=false

─── config.yaml (GLM-OCR cloud mode) ────────────────────────────────────────
maas:
  enabled: true
  endpoint: "https://open.bigmodel.cn/api/paas/v4/"

─── docker-compose.yml ───────────────────────────────────────────────────────
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
volumes:
  qdrant_data:

─── Package skeleton ─────────────────────────────────────────────────────────
Create empty __init__.py files in:
  src/doc_parser/__init__.py
  src/doc_parser/api/__init__.py
  src/doc_parser/api/routes/__init__.py
  src/doc_parser/ingestion/__init__.py
  src/doc_parser/retrieval/__init__.py
  src/doc_parser/utils/__init__.py

Create empty .gitkeep files in:
  scripts/.gitkeep
  notebooks/.gitkeep
  tests/unit/.gitkeep
  tests/integration/.gitkeep
  tests/__init__.py
  tests/unit/__init__.py
  tests/integration/__init__.py
```

---

## SESSION 2 — Configuration & Logging

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Session 1 is complete: pyproject.toml, .gitignore, .env.example, config.yaml,
docker-compose.yml, and the src/doc_parser/ package skeleton all exist.
Now we build the configuration singleton and logging setup.
```

### PROMPT
```
Create src/doc_parser/config.py and src/doc_parser/logging_config.py.

─── src/doc_parser/config.py ─────────────────────────────────────────────────
from __future__ import annotations
import logging
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )
    # Parser
    parser_backend: str = "cloud"          # "cloud" | "ollama"
    z_ai_api_key: SecretStr | None = None
    log_level: str = "INFO"
    output_dir: str = "./output"
    config_yaml_path: str = "config.yaml"
    # OpenAI
    openai_api_key: SecretStr | None = None
    openai_llm_model: str = "gpt-4o"
    # Embedding
    embedding_provider: str = "openai"     # "openai" | "gemini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    gemini_api_key: SecretStr | None = None
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "documents"
    # Reranker
    reranker_backend: str = "jina"         # "jina" | "openai" | "bge" | "qwen"
    reranker_top_n: int = 5
    jina_api_key: SecretStr | None = None
    # Feature flags
    image_caption_enabled: bool = True
    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    # Logging
    log_json: bool = False

    @model_validator(mode="after")
    def _validate_backend(self) -> "Settings":
        if self.parser_backend == "cloud":
            if self.z_ai_api_key is None:
                raise ValueError("Z_AI_API_KEY is required when PARSER_BACKEND=cloud")
        elif self.parser_backend == "ollama":
            if self.config_yaml_path == "config.yaml":
                self.config_yaml_path = "ollama/config.yaml"
        else:
            raise ValueError(
                f"PARSER_BACKEND must be 'cloud' or 'ollama', got: {self.parser_backend!r}"
            )
        return self

_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

─── src/doc_parser/logging_config.py ─────────────────────────────────────────
from __future__ import annotations
import sys
from loguru import logger

def setup_logging(level: str = "INFO", log_json: bool = False) -> None:
    logger.remove()
    if log_json:
        logger.add(sys.stdout, level=level.upper(), serialize=True)
    else:
        logger.add(
            sys.stdout,
            level=level.upper(),
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
```

---

## SESSION 3 — Document Parsing Pipeline

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-2 complete. We have: project skeleton, config.py (pydantic-settings
singleton with PARSER_BACKEND cloud/ollama, SecretStr keys, model_validator),
logging_config.py (loguru).

CRITICAL FACTS about GLM-OCR:
- bbox_2d values are normalised 0-1000 scale (NOT pixels)
  Convert: pixel = bbox_value * image_dimension / 1000
- In cloud mode MUST pass start_page_id=0 and end_page_id=N-1 or SDK only parses page 1
- In ollama mode MUST NOT pass api_key (any non-None key forces cloud mode in SDK)
- SDK returns: json_result (list[list[dict]]) and markdown_result (str)
- Each element dict: {"index": int, "label": str, "content": str, "bbox_2d": [f,f,f,f]}
```

### PROMPT
```
Create src/doc_parser/utils/pdf_utils.py and src/doc_parser/pipeline.py.

─── src/doc_parser/utils/pdf_utils.py ────────────────────────────────────────
"""PyMuPDF helpers for PDF page counting and rendering."""
from __future__ import annotations
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

def count_pdf_pages(path: Path | str) -> int:
    doc = fitz.open(str(path))
    try:
        return len(doc)
    finally:
        doc.close()

def pdf_page_to_image(path: Path | str, page_index: int, dpi: int = 150) -> Image.Image:
    """Render a PDF page as a PIL Image. page_index is 0-based."""
    doc = fitz.open(str(path))
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()

─── src/doc_parser/pipeline.py ───────────────────────────────────────────────
"""Main document parsing pipeline wrapping the GLM-OCR SDK."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from tqdm import tqdm
from doc_parser.config import get_settings
from doc_parser.post_processor import assemble_markdown, save_to_json
from doc_parser.utils.pdf_utils import count_pdf_pages

logger = logging.getLogger(__name__)

try:
    from glmocr import GlmOcr
    _GLMOCR_AVAILABLE = True
except ImportError:
    _GLMOCR_AVAILABLE = False

@dataclass
class ParsedElement:
    label: str           # "paragraph", "table", "image", "formula", etc.
    text: str            # recognized text
    bbox: list[float]    # [x1,y1,x2,y2] normalised to 0-1000 (NOT pixels)
    score: float         # always 1.0 (SDK does not return confidence)
    reading_order: int   # sequence index for correct reading order

@dataclass
class PageResult:
    page_num: int                              # 1-based
    elements: list[ParsedElement] = field(default_factory=list)
    markdown: str = ""

@dataclass
class ParseResult:
    source_file: str
    pages: list[PageResult] = field(default_factory=list)
    total_elements: int = 0
    full_markdown: str = ""   # full document Markdown from SDK (preferred over per-page)

    @classmethod
    def from_sdk_result(cls, raw: Any, source_file: str) -> "ParseResult":
        raw_pages: list[list[dict]] = getattr(raw, "json_result", [])
        full_markdown: str = getattr(raw, "markdown_result", "") or ""
        pages = []
        for page_idx, raw_elements in enumerate(raw_pages):
            elements = []
            for raw_el in raw_elements:
                elements.append(ParsedElement(
                    label=raw_el.get("label", "paragraph"),
                    text=raw_el.get("content", ""),
                    bbox=[float(v) for v in raw_el.get("bbox_2d", [0, 0, 1, 1])],
                    score=1.0,
                    reading_order=raw_el.get("index", len(elements)),
                ))
            markdown = assemble_markdown(elements)
            pages.append(PageResult(page_num=page_idx + 1, elements=elements, markdown=markdown))
        return cls(
            source_file=source_file,
            pages=pages,
            total_elements=sum(len(p.elements) for p in pages),
            full_markdown=full_markdown,
        )

    def save(self, output_dir: Path) -> None:
        save_to_json(self, output_dir)


class DocumentParser:
    def __init__(self) -> None:
        if not _GLMOCR_AVAILABLE:
            raise ImportError("glmocr not installed. Run: uv pip install glmocr")
        settings = get_settings()
        # CRITICAL: never pass api_key in ollama mode — any non-None key forces cloud mode
        api_key = (
            settings.z_ai_api_key.get_secret_value()
            if settings.parser_backend == "cloud" and settings.z_ai_api_key
            else None
        )
        self._parser = GlmOcr(config_path=settings.config_yaml_path, api_key=api_key)

    def parse_file(self, file_path: str | Path) -> ParseResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        settings = get_settings()
        parse_kwargs: dict[str, Any] = {}
        if file_path.suffix.lower() == ".pdf":
            total_pages = count_pdf_pages(file_path)
            if settings.parser_backend == "cloud":
                # CRITICAL: without these the SDK silently parses ONLY page 1
                parse_kwargs["start_page_id"] = 0
                parse_kwargs["end_page_id"] = total_pages - 1
            # ollama mode: do NOT pass start/end — SDK ignores them (uses pypdfium2)
        if settings.parser_backend == "ollama":
            parse_kwargs["save_layout_visualization"] = False
        raw = self._parser.parse(str(file_path), **parse_kwargs)
        result = ParseResult.from_sdk_result(raw, source_file=str(file_path))
        if file_path.suffix.lower() == ".pdf" and len(result.pages) != total_pages:
            logger.warning("Page count mismatch: SDK=%d PyMuPDF=%d for %s",
                           len(result.pages), total_pages, file_path.name)
        return result

    def parse_batch(self, file_paths: list[Path], output_dir: Path) -> list[ParseResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for fp in tqdm(file_paths, desc="Parsing", unit="file"):
            result = self.parse_file(fp)
            result.save(output_dir)
            results.append(result)
        return results
```

---

## SESSION 4 — Post-Processor & Chunker

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-3 complete. We have config, logging, pdf_utils, and pipeline.py
(ParsedElement, PageResult, ParseResult, DocumentParser).

Now we build:
1. post_processor.py — converts elements to Markdown, saves JSON
2. chunker.py — splits elements into RAG-ready chunks

Chunking rules (MUST follow exactly):
- ATOMIC labels {table, formula, inline_formula, algorithm, image, figure}:
  always their own chunk, never merged, never split
- TITLE labels {document_title, paragraph_title, figure_title}:
  attach FORWARD to the next content element, never backward
- figure_title is a figure caption: must join the NEXT image/figure atomic chunk
  (co-locates caption text + visual in one chunk)
- Regular text: accumulates until max_chunk_tokens (default 512)
- Token estimate: word_count * 1.3
- chunk.page = page of the FIRST element in the chunk
- chunk_id format: "{source_file}_{page}_{index}"
```

### PROMPT
```
Create src/doc_parser/post_processor.py and src/doc_parser/chunker.py.

─── src/doc_parser/post_processor.py ─────────────────────────────────────────
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

@runtime_checkable
class ElementLike(Protocol):
    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int

SKIP_LABELS: frozenset[str] = frozenset({"image", "seal", "page_number"})

PROMPT_MAP: dict[str, Any] = {
    "document_title":  lambda t: f"# {t}",
    "paragraph_title": lambda t: f"## {t}",
    "abstract":        lambda t: f"**Abstract:** {t}",
    "table":           lambda t: t,
    "formula":         lambda t: f"\n$$\n{t}\n$$\n",
    "inline_formula":  lambda t: f"\n$$\n{t}\n$$\n",
    "code_block":      lambda t: f"```\n{t}\n```",
    "footnotes":       lambda t: f"\n---\n{t}",
    "algorithm":       lambda t: f"```\n{t}\n```",
}

def assemble_markdown(elements: list[ElementLike]) -> str:
    if not elements:
        return ""
    parts = []
    for el in sorted(elements, key=lambda e: e.reading_order):
        if el.label in SKIP_LABELS:
            continue
        transform = PROMPT_MAP.get(el.label)
        parts.append(transform(el.text) if transform else el.text)
    return "\n\n".join(parts).strip()

def save_to_json(result: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source_file).stem
    full_markdown = getattr(result, "full_markdown", "") or ""
    if not full_markdown:
        full_markdown = "\n\n".join(p.markdown for p in result.pages if p.markdown)
    (output_dir / f"{stem}.md").write_text(full_markdown, encoding="utf-8")
    json_data = {
        "source_file": result.source_file,
        "total_elements": result.total_elements,
        "pages": [
            {
                "page_num": page.page_num,
                "elements": [
                    {"label": el.label, "text": el.text, "bbox": el.bbox,
                     "score": el.score, "reading_order": el.reading_order}
                    for el in page.elements
                ],
                "markdown": page.markdown,
            }
            for page in result.pages
        ],
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

─── src/doc_parser/chunker.py ────────────────────────────────────────────────
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from doc_parser.post_processor import ElementLike

logger = logging.getLogger(__name__)

_TOKEN_WORD_RATIO: float = 1.3

ATOMIC_LABELS: frozenset[str] = frozenset(
    {"table", "formula", "inline_formula", "algorithm", "image", "figure"}
)
TITLE_LABELS: frozenset[str] = frozenset(
    {"document_title", "paragraph_title", "figure_title"}
)
_IMAGE_TYPES   = frozenset({"image", "figure"})
_TABLE_TYPES   = frozenset({"table"})
_FORMULA_TYPES = frozenset({"formula", "inline_formula"})
_ALGO_TYPES    = frozenset({"algorithm"})

def _infer_modality(element_types: list[str]) -> str:
    t = frozenset(element_types)
    if t & _IMAGE_TYPES:   return "image"
    if t & _TABLE_TYPES:   return "table"
    if t & _FORMULA_TYPES: return "formula"
    if t & _ALGO_TYPES:    return "algorithm"
    return "text"

@dataclass
class Chunk:
    text: str
    chunk_id: str            # "{source_file}_{page}_{idx}"
    page: int
    element_types: list[str]
    bbox: list[float] | None
    source_file: str
    is_atomic: bool
    modality: str = field(default="text")
    image_base64: str | None = field(default=None)
    caption: str | None = field(default=None)

def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOKEN_WORD_RATIO)

def _split_text_into_sub_chunks(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    wpch = max(1, int(max_tokens / _TOKEN_WORD_RATIO))
    return [" ".join(words[i: i + wpch]) for i in range(0, len(words), wpch)]

def document_aware_chunking(
    pages: list[tuple[int, list[ElementLike]]],
    source_file: str,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Process ALL pages as one stream so headings at page N attach to content at page N+1."""
    all_pairs = sorted(
        [(pn, el) for pn, els in pages for el in els],
        key=lambda x: (x[0], x[1].reading_order),
    )
    if not all_pairs:
        return []

    chunks: list[Chunk] = []
    chunk_idx = 0
    cur_texts: list[str] = []
    cur_labels: list[str] = []
    cur_tokens: int = 0
    cur_page: int = all_pairs[0][0]
    pending_title: str | None = None
    pending_label: str | None = None
    pending_page: int = cur_page

    def flush() -> None:
        nonlocal cur_texts, cur_labels, cur_tokens, chunk_idx
        nonlocal pending_title, pending_label, pending_page, cur_page
        if not cur_texts and pending_title is None:
            return
        texts, labels = [], []
        page_use = pending_page if (pending_title and not cur_texts) else cur_page
        if pending_title is not None:
            texts.append(pending_title)
            labels.append(pending_label or "paragraph_title")
            pending_title = pending_label = None
        texts.extend(cur_texts); labels.extend(cur_labels)
        if not texts:
            return
        chunks.append(Chunk(
            text="\n\n".join(texts),
            chunk_id=f"{source_file}_{page_use}_{chunk_idx}",
            page=page_use, element_types=labels, bbox=None,
            source_file=source_file, is_atomic=False,
            modality=_infer_modality(labels),
        ))
        chunk_idx += 1
        cur_texts.clear(); cur_labels.clear()
        cur_tokens = 0

    for page_num, element in all_pairs:
        label = element.label
        text = element.text.strip()

        if label in ATOMIC_LABELS:
            # figure_title directly before image/figure → prepend as caption
            fig_caption: str | None = None
            if pending_title is not None and pending_label == "figure_title":
                fig_caption = pending_title
                pending_title = pending_label = None
            flush()
            if fig_caption:
                atomic_text = f"{fig_caption}\n\n{text}" if text else fig_caption
                atomic_labels = ["figure_title", label]
            else:
                atomic_text = text
                atomic_labels = [label]
            chunks.append(Chunk(
                text=atomic_text,
                chunk_id=f"{source_file}_{page_num}_{chunk_idx}",
                page=page_num, element_types=atomic_labels, bbox=element.bbox,
                source_file=source_file, is_atomic=True,
                modality=_infer_modality(atomic_labels),
            ))
            chunk_idx += 1
            continue

        if not text:
            continue

        if label in TITLE_LABELS:
            if cur_texts:
                flush()
            elif pending_title is not None:
                flush()   # two consecutive titles → flush orphan first
            pending_title = text
            pending_label = label
            pending_page = page_num
            continue

        # Regular content
        tok = _estimate_tokens(text)
        p_tok = _estimate_tokens(pending_title) if pending_title else 0

        if tok > max_chunk_tokens:
            flush()
            for sub in _split_text_into_sub_chunks(text, max_chunk_tokens):
                chunks.append(Chunk(
                    text=sub, chunk_id=f"{source_file}_{page_num}_{chunk_idx}",
                    page=page_num, element_types=[label], bbox=None,
                    source_file=source_file, is_atomic=False,
                    modality=_infer_modality([label]),
                ))
                chunk_idx += 1
            continue

        if cur_texts and (cur_tokens + tok + p_tok > max_chunk_tokens):
            flush()

        if pending_title is not None:
            if not cur_texts:
                cur_page = pending_page
            cur_texts.append(pending_title)
            cur_labels.append(pending_label or "paragraph_title")
            cur_tokens += _estimate_tokens(pending_title)
            pending_title = pending_label = None

        if not cur_texts:
            cur_page = page_num
        cur_texts.append(text)
        cur_labels.append(label)
        cur_tokens += tok
        if cur_tokens >= max_chunk_tokens:
            flush()

    flush()
    return chunks

def structure_aware_chunking(
    elements: list[ElementLike],
    source_file: str,
    page: int,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Single-page convenience wrapper around document_aware_chunking."""
    return document_aware_chunking([(page, elements)], source_file, max_chunk_tokens)
```

---

## SESSION 5 — Hybrid Ingestion (Embeddings + Vector Store)

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-4 complete. We have: config, logging, pdf_utils, pipeline
(ParsedElement/PageResult/ParseResult/DocumentParser), post_processor
(assemble_markdown/save_to_json/ElementLike), chunker (Chunk dataclass,
document_aware_chunking, structure_aware_chunking).

Now we build the ingestion layer:
- embedder.py: dense embeddings (OpenAI/Gemini) + BM25 sparse via feature hashing
- vector_store.py: Qdrant async wrapper with hybrid search

KEY FACTS:
- Dense: text-embedding-3-large, 3072 dimensions, COSINE distance
- Sparse: feature hashing with 2^17=131072 buckets, normalised TF weights
  term → idx = abs(hash(term)) % 131072, weight = count/total_terms
- Qdrant collection has TWO named vector spaces: "text_dense" and "bm25_sparse"
- Point IDs: uuid5(NAMESPACE_DNS, chunk_id) — deterministic, safe to re-ingest
- Hybrid search: Prefetch both, fuse with FusionQuery(fusion=Fusion.RRF)
- batch_size=64 for upsert, batch_size=100 for embedding API calls
```

### PROMPT
```
Create src/doc_parser/ingestion/embedder.py and src/doc_parser/ingestion/vector_store.py.

─── src/doc_parser/ingestion/embedder.py ─────────────────────────────────────
from __future__ import annotations
import asyncio, logging, re
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
from openai import AsyncOpenAI
from qdrant_client.models import SparseVector
if TYPE_CHECKING:
    from doc_parser.chunker import Chunk
    from doc_parser.config import Settings

logger = logging.getLogger(__name__)
_BM25_N_FEATURES: int = 2**17   # 131072 hash buckets

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

async def embed_texts(
    texts: list[str],
    client: AsyncOpenAI,
    model: str = "text-embedding-3-large",
    dimensions: int = 3072,
    batch_size: int = 100,
) -> list[list[float]]:
    sanitised = [t if t.strip() else "[empty]" for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(sanitised), batch_size):
        response = await client.embeddings.create(
            model=model, input=sanitised[i: i + batch_size], dimensions=dimensions
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings

def compute_sparse_vectors(
    texts: list[str], n_features: int = _BM25_N_FEATURES
) -> list[SparseVector]:
    vectors: list[SparseVector] = []
    for text in texts:
        tokens = _tokenize(text)
        if not tokens:
            vectors.append(SparseVector(indices=[], values=[]))
            continue
        tf = Counter(tokens)
        total = len(tokens)
        buckets: dict[int, float] = {}
        for term, count in tf.items():
            buckets[abs(hash(term)) % n_features] = count / total
        sorted_items = sorted(buckets.items())
        vectors.append(SparseVector(
            indices=[i for i, _ in sorted_items],
            values=[v for _, v in sorted_items],
        ))
    return vectors

class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, settings: "Settings") -> None:
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self._client = AsyncOpenAI(api_key=key)
        self._model = settings.embedding_model
        self._dim = settings.embedding_dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts, self._client, self._model, self._dim)

class GeminiEmbedder(BaseEmbedder):
    _MODEL = "gemini-embedding-2-preview"

    def __init__(self, settings: "Settings") -> None:
        if settings.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY required when EMBEDDING_PROVIDER=gemini")
        try:
            from google import genai as _genai
        except ImportError as exc:
            raise ImportError("Install: uv pip install 'doc-parser[gemini]'") from exc
        self._client = _genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        sanitised = [t if t.strip() else "[empty]" for t in texts]
        result = self._client.models.embed_content(model=self._MODEL, contents=sanitised)
        return [e.values for e in result.embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.get_running_loop().run_in_executor(None, self._embed_sync, texts)

_PROVIDERS = {"openai": OpenAIEmbedder, "gemini": GeminiEmbedder}

def get_embedder(settings: "Settings") -> BaseEmbedder:
    provider = settings.embedding_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Choose: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](settings)

async def embed_chunks(
    chunks: list["Chunk"], embedder: "BaseEmbedder", settings: "Settings"
) -> tuple[list[list[float]], list[SparseVector]]:
    texts = [c.text for c in chunks]
    return await embedder.embed(texts), compute_sparse_vectors(texts)

─── src/doc_parser/ingestion/vector_store.py ─────────────────────────────────
from __future__ import annotations
import logging, uuid
from typing import TYPE_CHECKING
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, Fusion, FusionQuery, HnswConfigDiff, PointStruct,
    Prefetch, SparseIndexParams, SparseVector, SparseVectorParams, VectorParams,
)
from doc_parser.ingestion.embedder import BaseEmbedder, compute_sparse_vectors
if TYPE_CHECKING:
    from doc_parser.chunker import Chunk
    from doc_parser.config import Settings

logger = logging.getLogger(__name__)

class QdrantDocumentStore:
    def __init__(self, settings: "Settings") -> None:
        key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        self._client = AsyncQdrantClient(url=settings.qdrant_url, api_key=key)
        self._collection = settings.qdrant_collection_name
        self._settings = settings

    async def create_collection(self, overwrite: bool = False) -> None:
        existing = {c.name for c in (await self._client.get_collections()).collections}
        if self._collection in existing:
            if not overwrite:
                return
            await self._client.delete_collection(self._collection)
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={"text_dense": VectorParams(
                size=self._settings.embedding_dimensions,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )},
            sparse_vectors_config={"bm25_sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )},
        )

    async def delete_collection(self, collection_name: str) -> bool:
        existing = {c.name for c in (await self._client.get_collections()).collections}
        if collection_name not in existing:
            return False
        await self._client.delete_collection(collection_name)
        return True

    async def upsert_chunks(
        self,
        chunks: list["Chunk"],
        dense_embeddings: list[list[float]],
        sparse_vectors: list[SparseVector],
        batch_size: int = 64,
    ) -> int:
        if not (len(chunks) == len(dense_embeddings) == len(sparse_vectors)):
            raise ValueError("Length mismatch: chunks, dense, and sparse must be same length")
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector={"text_dense": dense, "bm25_sparse": sparse},
                payload={
                    "text": chunk.text, "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file, "page": chunk.page,
                    "element_types": chunk.element_types, "bbox": chunk.bbox,
                    "is_atomic": chunk.is_atomic, "modality": chunk.modality,
                    "image_base64": chunk.image_base64, "caption": chunk.caption,
                },
            )
            for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_vectors)
        ]
        total = 0
        for i in range(0, len(points), batch_size):
            await self._client.upsert(collection_name=self._collection, points=points[i: i + batch_size])
            total += len(points[i: i + batch_size])
        return total

    async def search(
        self,
        query_text: str,
        embedder: "BaseEmbedder",
        settings: "Settings",
        top_k: int = 10,
        filter_modality: str | None = None,
    ) -> list[dict]:
        query_dense = (await embedder.embed([query_text]))[0]
        query_sparse = compute_sparse_vectors([query_text])[0]
        query_filter = None
        if filter_modality:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            query_filter = Filter(must=[FieldCondition(
                key="modality", match=MatchValue(value=filter_modality)
            )])
        results = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=query_dense, using="text_dense", limit=top_k * 2),
                Prefetch(query=query_sparse, using="bm25_sparse", limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [p.payload for p in results.points]
```

---

## SESSION 6 — Multimodal Enrichment (GPT-4o Captioning)

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-5 complete. Now we build image_captioner.py.

PURPOSE: After chunking, image chunks have text="" or "[figure]". Tables have
raw OCR text that embeds poorly. This module calls GPT-4o BEFORE embedding to
replace that content with rich structured descriptions.

KEY FACTS:
- bbox_2d coords are 0-1000 normalised → pixel = bbox_value * image_dim / 1000
- page_index for pdf_page_to_image is 0-based (chunk.page - 1)
- Skip crops smaller than 50x50 pixels (detection noise)
- Store: chunk.text = enriched description (gets embedded)
         chunk.caption = short caption (shown in API responses)
         chunk.image_base64 = base64 PNG (passed to reranker for visual scoring)
- Semaphore(5): limits concurrent GPT-4o calls to avoid rate limits
- Tables: truncate input to 3000 chars. max_tokens=400
- Formulas: max_tokens=350. Algorithms: max_tokens=450. Images: max_tokens=512
```

### PROMPT
```
Create src/doc_parser/ingestion/image_captioner.py with these exact system prompts
and exact field assignments.

The four system prompts (copy exactly — the response parsers depend on their format):

_IMAGE_SYSTEM_PROMPT asks for: CAPTION: / FLOW: / STRUCTURE: sections
_TABLE_SYSTEM_PROMPT asks for: SUMMARY: / DETAIL: sections
_FORMULA_SYSTEM_PROMPT asks for: SUMMARY: / DETAIL: sections
_ALGORITHM_SYSTEM_PROMPT asks for: SUMMARY: / DETAIL: sections

Response parsers:
  _parse_image_response(text) -> (caption_str, full_text_str)
    Extract the CAPTION: line as the short caption.
    Return full response as full_text. Fall back to text[:200] if no CAPTION: line.

  _parse_text_response(raw_original, enriched) -> (raw_for_caption, enriched_for_text)
    caption = original raw OCR text, text = enriched description

Per-modality async helpers (each takes semaphore):
  _enrich_image_single: render page at dpi=150, crop bbox, encode PNG as base64,
    call GPT-4o vision. Set chunk.caption, chunk.text, chunk.image_base64.
  _enrich_table_single: text call with _TABLE_SYSTEM_PROMPT, max_tokens=400
  _enrich_formula_single: text call with _FORMULA_SYSTEM_PROMPT, max_tokens=350
  _enrich_algorithm_single: text call with _ALGORITHM_SYSTEM_PROMPT, max_tokens=450

Public functions:
  async def enrich_chunks(chunks, pdf_path, client, model="gpt-4o", max_concurrent=5)
    Dispatch by chunk.modality. Use asyncio.Semaphore(max_concurrent).
    Use asyncio.gather(*tasks). Return chunks (mutated in place).

  async def enrich_image_chunks(chunks, pdf_path, client, max_concurrent=5)
    Backward-compat alias → calls enrich_chunks(...)

All enrichment helpers must catch all exceptions and set chunk.text="[figure]"
on failure for images, or leave chunk unchanged on failure for text types.
```

---

## SESSION 7 — Reranking Backends

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-6 complete. Now we build the four reranker backends.

PURPOSE: After Qdrant returns top-20 candidates, a reranker cross-encodes each
(query, candidate) pair for much more accurate relevance scoring.

The four backends (selected by RERANKER_BACKEND env var):
1. openai   → GPT-4o-mini as async cross-encoder (all candidates in parallel)
2. jina     → Jina Reranker M0 cloud API (multimodal, single batch call)
3. bge      → BAAI/bge-reranker-v2-minicpm-layerwise (local, text-only, thread pool)
4. qwen     → Qwen3-VL-Reranker-2B (local, multimodal, thread pool)

All backends: add "rerank_score" key to each returned candidate dict.
All backends: sort descending by score, return top_n.
BGE uses cutoff_layers=[28]. Uses MPS if available, else CPU.
Qwen: AutoProcessor + AutoModelForSequenceClassification from "Qwen/Qwen3-VL-Reranker-2B"
Jina API URL: https://api.jina.ai/v1/rerank, model: jina-reranker-m0
```

### PROMPT
```
Create src/doc_parser/retrieval/reranker.py with BaseReranker ABC and four backends.

BaseReranker ABC:
  abstract async def rerank(query, candidates: list[dict], top_n=5) -> list[dict]

Each candidate dict has keys: text, modality, image_base64 (optional), source_file, page.

OpenAIReranker:
  _SCORE_PROMPT: "Rate the relevance of the following document to the query on a scale
  of 1 to 10. Reply with ONLY the integer score (e.g. '7'), nothing else.\n\n
  Query: {query}\n\nDocument: {text}"
  For image chunks with image_base64: use vision API message instead.
  model="gpt-4o-mini", temperature=0.0, max_tokens=4
  On parse failure return 0.0.

JinaReranker:
  POST to https://api.jina.ai/v1/rerank
  model: "jina-reranker-m0"
  documents: image chunks → {"text": text, "images": [base64_str]}
             text chunks  → {"text": text}
  timeout=30.0 with httpx.AsyncClient

BGEReranker:
  from FlagEmbedding import LayerWiseFlagLLMReranker
  model: "BAAI/bge-reranker-v2-minicpm-layerwise", use_fp16=True, cutoff_layers=[28]
  pairs = [[query, text[:2000]] for each candidate]
  run _compute_scores_sync in thread pool via run_in_executor

QwenVLReranker:
  "Qwen/Qwen3-VL-Reranker-2B"
  For image chunks: decode base64 → PIL Image → pass to processor with images=[image]
  For text chunks: processor(text=[[query, text[:2000]]])
  torch.no_grad() → model(**inputs).logits[0].item()
  Each candidate scored in separate thread pool call, then gathered

Factory:
  _BACKENDS = {"openai": OpenAIReranker, "jina": JinaReranker,
               "bge": BGEReranker, "qwen": QwenVLReranker}
  def get_reranker(settings) -> BaseReranker
```

---

## SESSION 8 — FastAPI REST API

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-7 complete. Now we build the full FastAPI application.

Four endpoints:
  GET  /health   → HealthResponse
  POST /ingest   → multipart PDF upload → parse+chunk+enrich+embed+upsert
  POST /search   → hybrid search + optional rerank
  POST /generate → search + rerank + GPT-4o answer

IMPORTANT:
- dependencies.py uses module-level lazy singletons (initialised on first request)
- /ingest writes uploaded file to NamedTemporaryFile (glmocr needs a real path)
- /search omits image_base64 from ChunkResult response (too large)
- /generate context format: "[page N] {text}" joined with \n\n
- Default system prompt: "Answer using ONLY the provided context.
  If not in context, say I don't have enough information. Cite page numbers."
- LoggingMiddleware logs: method, path, status_code, latency_ms
- app = create_app() at module level for uvicorn
```

### PROMPT
```
Create all files in src/doc_parser/api/:
  schemas.py, dependencies.py, middleware.py, app.py
  routes/__init__.py, routes/health.py, routes/ingest.py,
  routes/search.py, routes/generate.py

schemas.py — Pydantic v2 models:
  SearchRequest:  query(str), top_k=20, top_n=None, rerank=True, filter_modality=None
  ChunkResult:    chunk_id, text, source_file, page, modality, element_types,
                  bbox, is_atomic, caption, rerank_score, image_base64=None
  SearchResponse: query, backend, total_candidates, results:list[ChunkResult], latency_ms
  GenerateRequest: query, top_k=20, top_n=None, rerank=True, filter_modality=None,
                   max_tokens=1024, system_prompt=None
  GenerateResponse: query, answer, sources:list[ChunkResult], total_candidates, latency_ms
  IngestResponse: filename, chunks_upserted, latency_ms
  HealthResponse: status, version, parser_backend, reranker_backend,
                  embedding_provider, collection

dependencies.py — lazy singletons:
  get_store() -> QdrantDocumentStore
  get_embedder_dep() -> BaseEmbedder
  get_reranker_dep() -> BaseReranker
  get_openai_client() -> AsyncOpenAI

middleware.py — LoggingMiddleware(BaseHTTPMiddleware):
  log: "{method} {path} {status_code} {latency_ms:.1f}ms" via loguru

routes/health.py: GET "" → HealthResponse(status="ok", version="0.1.0", ...)
routes/ingest.py: POST "" file:UploadFile → full pipeline → IngestResponse
routes/search.py: POST "" SearchRequest → candidates → optional rerank → SearchResponse
routes/generate.py: POST "" GenerateRequest → candidates → rerank → GPT-4o → GenerateResponse

app.py:
  create_app() → FastAPI with lifespan (setup_logging on startup, log shutdown)
  add LoggingMiddleware, mount all routers with correct prefixes/tags
  app = create_app() at module level
```

---

## SESSION 9 — CLI Scripts

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-8 complete. Now we build 5 CLI scripts in scripts/.
Each script adds sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
so it works without installing the package.
```

### PROMPT
```
Create these five scripts in scripts/:

scripts/ingest.py
  Args: input (file or dir), --no-captions, --collection NAME,
        --overwrite, --max-chunk-tokens 512
  Supported extensions: .pdf .png .jpg .jpeg .tiff .bmp
  Pipeline per file: parse → chunk (structure_aware_chunking per page) →
    enrich (unless --no-captions) → embed → upsert
  Use rich Progress with SpinnerColumn, TextColumn, TimeElapsedColumn
  Show step name in progress description: "filename — parsing", "filename — chunking", etc.
  Print final summary: total chunks per modality
  --overwrite: store.create_collection(overwrite=True) before first file

scripts/search.py
  Args: query (str), --top-k 20, --top-n 5, --no-rerank, --modality
  Embed query → search → rerank → print rich table:
  Rank | Source | Page | Modality | Score | Text (first 120 chars)

scripts/parse.py
  Args: input (file or dir), --output-dir ./output
  Parse and save .md + .json per file to output_dir
  Print per-file: pages, elements, output paths

scripts/serve.py
  import uvicorn, print local URL + /docs URL, then uvicorn.run(app, ...)
  host/port/workers from settings

scripts/debug_raw.py
  Args: pdf_path
  Call GlmOcr.parse() directly, print raw json_result (first 2 pages)
  and first 500 chars of markdown_result
  Useful for inspecting raw SDK output
```

---

## SESSION 10 — Ollama Self-Hosted Mode

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-9 complete. Now we add the Ollama self-hosted alternative to
the Z.AI cloud API. When PARSER_BACKEND=ollama the Settings validator
automatically sets config_yaml_path to "ollama/config.yaml".
```

### PROMPT
```
Create these files in ollama/:

ollama/config.yaml:
  maas:
    enabled: false
  ollama:
    endpoint: "http://localhost:11434"
    layout_model: "glm-ocr:latest"
    ocr_model: "glm-ocr:latest"

ollama/api_parse.py:
  Direct HTTP debugging client (bypasses glmocr SDK).
  Accept PDF path as CLI arg. For each page: render at 150 DPI with PyMuPDF,
  encode as base64 PNG, POST to http://localhost:11434/api/generate with
  model="glm-ocr:latest", stream=false. Parse and print elements per page.

ollama/test_parse.py:
  Smoke test. Instantiate DocumentParser() (reads PARSER_BACKEND=ollama from .env).
  Parse a PDF (path from sys.argv[1]). Print pages, total elements,
  first 3 elements per page (label + first 60 chars of text).

ollama/visualize.py:
  Render bounding box overlays. For each page: pdf_page_to_image at 150 DPI,
  draw bbox rectangles (0-1000 coords → pixels), color by label using same
  LABEL_COLORS dict as app.py. Save annotated images to ollama/output/.

ollama/README.md:
  Document: install Ollama, ollama pull glm-ocr:latest, set PARSER_BACKEND=ollama,
  run test_parse.py, run visualize.py, known differences vs cloud mode
  (pypdfium2 vs PyMuPDF page counts, no page range params in ollama mode).
```

---

## SESSION 11 — Streamlit Visualizer

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-10 complete. Now we build app.py — a Streamlit UI at the project root.

CRITICAL coordinate fact: GLM-OCR bbox_2d coords are 0-1000 normalised (NOT pixels).
Convert: pixel = bbox_value * image_dimension / 1000

Session state trick: compare uploaded.name to prevent re-parsing on every
slider/checkbox interaction. Only re-parse when the user uploads a DIFFERENT file.
```

### PROMPT
```
Create app.py in the project root.

Page config: title="GLM-OCR Document Visualizer", icon="📄", layout="wide"
Title: "📄 GLM-OCR Document Visualizer"
Caption: "Upload a PDF to detect and visualize document elements (PP-DocLayout-V3 + GLM-OCR 0.9B)"

RENDER_DPI = 150
BBOX_SCALE = 1000

LABEL_COLORS dict (exact values):
  document_title:  (220, 50, 50)    paragraph_title: (30, 100, 220)
  abstract:        (20, 160, 160)   paragraph:       (40, 160, 40)
  text:            (40, 160, 40)    table:           (230, 120, 0)
  formula:         (150, 50, 220)   inline_formula:  (180, 80, 220)
  figure_caption:  (0, 180, 200)    caption:         (0, 180, 200)
  code_block:      (200, 180, 0)    algorithm:       (220, 0, 180)
  footnotes:       (200, 80, 120)   reference:       (140, 80, 40)
  header:          (160, 160, 160)  footer:          (160, 160, 160)
  page_number:     (120, 120, 120)  image:           (100, 100, 100)
  seal:            (60, 60, 60)
DEFAULT_COLOR = (180, 180, 0)

render_page(pdf_path, page_num) → PIL Image at RENDER_DPI (use fitz)

draw_bboxes(img, elements) → annotated PIL Image:
  Use ImageDraw.Draw(img.copy(), "RGBA")
  For each element: fill=(*color, 35), outline=(*color, 220), width=2
  Draw badge label pill: colored rectangle + white text in top-left of box
  Skip if x2<=x1 or y2<=y1

build_legend(labels_present: set[str]):
  st.markdown("**Legend**"), st.columns(4)
  Colored HTML spans, 4 per row, sorted alphabetically

Session state keys: result, pdf_path, uploaded_filename
Compare by filename to avoid re-parsing on widget interaction.

Sidebar: file_uploader, "Parse Document" button, show_text checkbox, show_markdown checkbox
Main: page slider, col_img(3)+col_detail(2), full document markdown expander

col_img: st.image(annotated, use_container_width=True) + build_legend
col_detail: Counter of element labels as colored badge spans × count
  If show_text: expandable list sorted by reading_order, show label + text + bbox
If show_markdown: st.markdown(page_result.markdown)
Bottom expander: full document Markdown
```

---

## SESSION 12 — Test Suite

### CONTEXT (paste this first)
```
We are building a multimodal RAG system called doc-parser.
Sessions 1-11 complete. The full system exists. Now we write tests.

pytest asyncio_mode="auto" is set in pyproject.toml — no @pytest.mark.asyncio needed.
Use unittest.mock (not pytest-mock). Use SimpleNamespace for ElementLike fixtures.
Integration tests auto-skip when infrastructure is unavailable.
```

### PROMPT
```
Create the full test suite:

tests/conftest.py:
  Fixture sample_elements: list of 5 SimpleNamespace objects with
  label/text/bbox/score/reading_order attributes covering:
  document_title, paragraph, table, image, formula

  Fixture sample_chunks: list of 3 Chunk objects (text, image, table modalities)

tests/unit/test_chunker.py — test these cases exactly:
  test_atomic_elements_get_own_chunks: table+image+formula → 3 atomic chunks
  test_title_attaches_forward: paragraph_title + paragraph → 1 chunk containing both
  test_figure_title_joins_image_chunk: figure_title + image → 1 atomic chunk with both texts, "figure_title" in element_types
  test_token_limit_splits_text: 1000-word paragraph → multiple chunks each within limit
  test_orphan_title_at_end: title with no following content still emits as chunk
  test_cross_page_heading_attaches: heading on page 1, content on page 2 → 1 chunk

tests/unit/test_embedder.py:
  test_embed_texts_replaces_empty_strings: mock client, verify "[empty]" sent not ""
  test_compute_sparse_vectors_indices_are_sorted: indices must be ascending
  test_compute_sparse_vectors_values_are_normalised: all values > 0 and <= 1
  test_compute_sparse_vectors_empty_text: returns indices=[], values=[]
  test_get_embedder_unknown_provider_raises: ValueError with "Unknown embedding provider"

tests/unit/test_post_processor.py:
  test_assemble_markdown_sorts_by_reading_order: title before paragraph in output
  test_assemble_markdown_skips_image_label: image text not in Markdown output
  test_assemble_markdown_wraps_formula: $$ present in output
  test_assemble_markdown_empty_input: returns ""

tests/unit/test_api_schemas.py:
  test_search_request_defaults: top_k=20, rerank=True, filter_modality=None
  test_generate_request_missing_query_raises: ValidationError when query absent
  test_generate_request_defaults: max_tokens=1024, system_prompt=None

tests/integration/test_pipeline_e2e.py:
  Skip if Z_AI_API_KEY not in os.environ
  test_parse_real_pdf: parse test_data/Docling_Technical_Report.pdf,
  assert pages > 0, total_elements > 0, full_markdown is non-empty str

tests/integration/test_ingest_e2e.py:
  Skip if Qdrant not reachable (probe with httpx.get(QDRANT_URL, timeout=2))
  test_create_and_delete_collection: create_collection(overwrite=True),
  delete_collection(name) → returns True
```

---

## FINAL VERIFICATION

After all 12 sessions, run these checks:

```bash
# Install the package
uv pip install -e ".[dev]"

# Run unit tests (should all pass without API keys or Qdrant)
pytest tests/unit/ -v

# Start Qdrant
docker compose up -d

# Start the API
python scripts/serve.py

# Check health
curl http://localhost:8000/health

# Run the Streamlit visualizer
streamlit run app.py
```

---

## CRITICAL FACTS TO REMEMBER (tell Claude these if anything goes wrong)

1. GLM-OCR bbox_2d is 0-1000 normalised — NOT pixels.
   pixel = bbox_value * image_dimension / 1000

2. Cloud mode MUST pass start_page_id=0, end_page_id=N-1.
   Without it the SDK silently parses ONLY page 1.

3. Ollama mode MUST NOT pass api_key to GlmOcr().
   Any non-None api_key forces cloud mode regardless of config.yaml.

4. figure_title chunks must join the NEXT image/figure atomic chunk.
   They must NOT become standalone chunks or join text chunks.

5. GPT-4o enrichment MUST run before embedding.
   Unenriched image chunks embed as "[figure]" — useless for retrieval.

6. Qdrant point IDs use uuid5(NAMESPACE_DNS, chunk_id) — deterministic.
   Re-ingesting the same document updates points, not duplicates.

7. RRF fusion: FusionQuery(fusion=Fusion.RRF) — no weight tuning needed.

8. SecretStr: always use .get_secret_value() to access API key strings.

9. Lazy singletons in dependencies.py: never initialise at import time.

10. chunk.page = page of the FIRST element that entered the accumulator.
