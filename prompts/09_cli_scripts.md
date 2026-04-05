# Prompt 09 — CLI Scripts

## Stage
Build command-line tools for the five common developer workflows.

---

## Prompt

Write five scripts in `scripts/` that can be run from the project root
(`python scripts/<name>.py ...`) without installing the package.
Each script adds `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))`.

---

### scripts/ingest.py — End-to-end ingestion CLI

```
Usage: python scripts/ingest.py <input> [--no-captions] [--collection NAME]
                                 [--overwrite] [--max-chunk-tokens 512]
```

- `input` can be a single file or a directory (recursive search for
  .pdf, .png, .jpg, .jpeg, .tiff, .bmp).
- For each file: parse → chunk (structure_aware_chunking per page) →
  enrich image/table/formula/algorithm chunks (unless --no-captions) →
  embed (dense + sparse) → upsert to Qdrant.
- Show rich progress bars with spinner, file name, current step, and elapsed time.
- Print a final summary table with chunk counts by modality.
- --overwrite: delete and recreate the Qdrant collection before ingestion.

---

### scripts/search.py — Hybrid search CLI

```
Usage: python scripts/search.py "<query>" [--top-k 10] [--top-n 5]
                                           [--no-rerank] [--modality text|image|table]
```

- Embed the query, hybrid-search Qdrant, optionally rerank.
- Print results as a rich table: rank | source | page | modality | score | text excerpt.

---

### scripts/parse.py — Standalone parse to output/

```
Usage: python scripts/parse.py <file_or_dir> [--output-dir ./output]
```

- Parse documents with DocumentParser.
- Save Markdown + JSON for each file to output_dir.
- Print a summary with page and element counts.

---

### scripts/serve.py — Start the FastAPI server

```
Usage: python scripts/serve.py
```

- Calls `uvicorn doc_parser.api.app:app` with host/port/workers from Settings.
- Prints the local URL before starting.

---

### scripts/debug_raw.py — Dump raw GLM-OCR SDK output

```
Usage: python scripts/debug_raw.py <pdf_path>
```

- Calls GlmOcr.parse() directly (no SDK wrapper).
- Prints the raw json_result as pretty-printed JSON and the first 500 chars
  of markdown_result.
- Useful for verifying what the SDK actually returns.

---

## What this produced

- `scripts/ingest.py`
- `scripts/search.py`
- `scripts/parse.py`
- `scripts/serve.py`
- `scripts/debug_raw.py`
- `scripts/.gitkeep`
