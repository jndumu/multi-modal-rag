# Prompt 03 — Document Parsing Pipeline (GLM-OCR)

## Stage
Wrap the glmocr SDK into clean, typed Python dataclasses.

---

## Prompt

Build the core parsing pipeline in `src/doc_parser/pipeline.py`.

The glmocr SDK exposes a `GlmOcr` class.  When you call `GlmOcr.parse(file_path)`
it returns a `PipelineResult` with two attributes:
- `json_result`: `list[list[dict]]` — one inner list per page; each dict has:
  `index` (int), `label` (str), `content` (str), `bbox_2d` (list[float, 4]).
  The bbox_2d values are normalised to a 0–1000 scale, NOT pixels.
- `markdown_result`: `str` — full document markdown for the whole doc.

I need three dataclasses:

1. `ParsedElement` — label, text, bbox (list[float]), score (float), reading_order (int).
   The SDK doesn't give a confidence score, so default score=1.0.
   reading_order comes from the `index` key.

2. `PageResult` — page_num (int, 1-based), elements (list[ParsedElement]), markdown (str).

3. `ParseResult` — source_file (str), pages (list[PageResult]), total_elements (int),
   full_markdown (str).
   - classmethod `from_sdk_result(raw, source_file)` that builds from a PipelineResult.
     Per-page markdown should be assembled by calling `assemble_markdown(elements)`
     (from post_processor.py — will be built separately).
   - method `save(output_dir)` that delegates to `save_to_json(self, output_dir)`.

4. `DocumentParser` class:
   - `__init__`: reads Settings from `get_settings()`.
     In cloud mode pass `api_key` to GlmOcr; in ollama mode do NOT pass api_key
     (passing any key forces MaaS mode regardless of the YAML config).
     Pass `config_path` from settings.config_yaml_path.
   - `parse_file(file_path) -> ParseResult`:
     - Raise FileNotFoundError if missing.
     - For PDFs in cloud mode: call `count_pdf_pages(file_path)` (PyMuPDF util)
       then pass `start_page_id=0, end_page_id=total_pages-1` to `parser.parse()`.
       The SDK defaults to page 1 only if no range is given.
     - In ollama mode: do NOT pass start/end page ids (SDK ignores them and uses
       its own pypdfium2 loader internally).
     - In ollama mode: pass `save_layout_visualization=False`.
     - After parsing, warn if `len(result.pages) != total_pages` (PyMuPDF and
       pypdfium2 can disagree on page count for some PDFs).
   - `parse_batch(file_paths, output_dir) -> list[ParseResult]`:
     Iterates with tqdm, calls parse_file + result.save, logs errors with exc_info.

The glmocr import should be wrapped in a try/except ImportError so the rest of the
package can be imported even when glmocr is not installed (useful in test environments).

Also write `src/doc_parser/utils/pdf_utils.py` with:
- `count_pdf_pages(path) -> int` using fitz.open().
- `pdf_page_to_image(path, page_index, dpi=150) -> PIL.Image.Image` using PyMuPDF.

---

## What this produced

- `src/doc_parser/pipeline.py` — ParsedElement, PageResult, ParseResult, DocumentParser
- `src/doc_parser/utils/pdf_utils.py` — PyMuPDF helpers
- `src/doc_parser/utils/__init__.py`
