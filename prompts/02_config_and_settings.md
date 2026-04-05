# Prompt 02 — Configuration & Settings Management

## Stage
Build the settings singleton and logging setup that every other module imports.

---

## Prompt

Build `src/doc_parser/config.py` — a pydantic-settings `BaseSettings` class that
loads all configuration from environment variables / `.env` file.

The settings I need:

```
PARSER_BACKEND        "cloud" | "ollama"   (default "cloud")
Z_AI_API_KEY          SecretStr | None     required when backend=cloud
LOG_LEVEL             str                  default "INFO"
OUTPUT_DIR            str                  default "./output"
CONFIG_YAML_PATH      str                  default "config.yaml"

OPENAI_API_KEY        SecretStr | None
OPENAI_LLM_MODEL      str                  default "gpt-4o"

EMBEDDING_PROVIDER    "openai" | "gemini"  default "openai"
EMBEDDING_MODEL       str                  default "text-embedding-3-large"
EMBEDDING_DIMENSIONS  int                  default 3072
GEMINI_API_KEY        SecretStr | None

QDRANT_URL            str                  default "http://localhost:6333"
QDRANT_API_KEY        SecretStr | None
QDRANT_COLLECTION_NAME str                 default "documents"

RERANKER_BACKEND      "jina"|"openai"|"bge"|"qwen"  default "jina"
RERANKER_TOP_N        int                  default 5
JINA_API_KEY          SecretStr | None

IMAGE_CAPTION_ENABLED bool                 default True

API_HOST              str                  default "0.0.0.0"
API_PORT              int                  default 8000
API_WORKERS           int                  default 1

LOG_JSON              bool                 default False
```

Validation rules (model_validator after):
- If PARSER_BACKEND="cloud" and Z_AI_API_KEY is None → raise ValueError.
- If PARSER_BACKEND="ollama" and CONFIG_YAML_PATH is still the default
  "config.yaml" → auto-set it to "ollama/config.yaml".
- Any other PARSER_BACKEND value → raise ValueError.

Expose a module-level `get_settings() -> Settings` singleton so every caller
gets the same object without re-reading `.env`.

Also add `configure_logging(level: str)` that sets up Python's root logger
with a timestamped format string.

Also write `src/doc_parser/logging_config.py` — a loguru-based alternative to
configure structured JSON or human-readable logging, controlled by LOG_JSON.
Export `setup_logging(level, log_json)`.

---

## What this produced

- `src/doc_parser/config.py`  — Settings singleton with full validation
- `src/doc_parser/logging_config.py` — loguru JSON/pretty logging setup
