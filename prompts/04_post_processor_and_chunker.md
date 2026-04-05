# Prompt 04 — Post-Processor & Structure-Aware Chunker

## Stage
Convert raw parsed elements into Markdown, then split into RAG-ready chunks.

---

## Prompt A — Post-Processor

Write `src/doc_parser/post_processor.py`.

I need:
1. An `ElementLike` runtime-checkable Protocol with: label, text, bbox, score, reading_order.
   This allows duck-typing so tests can pass simple objects without importing ParsedElement.

2. `SKIP_LABELS` frozenset: {"image", "seal", "page_number"} — skip these in Markdown output.

3. `PROMPT_MAP` dict mapping label → lambda that transforms text to Markdown:
   - "document_title"  → f"# {t}"
   - "paragraph_title" → f"## {t}"
   - "abstract"        → f"**Abstract:** {t}"
   - "table"           → passthrough (t)
   - "formula"         → f"\n$$\n{t}\n$$\n"
   - "inline_formula"  → f"\n$$\n{t}\n$$\n"
   - "code_block"      → f"```\n{t}\n```"
   - "footnotes"       → f"\n---\n{t}"
   - "algorithm"       → f"```\n{t}\n```"
   Everything else (paragraph, text, reference, etc.) → plain passthrough.

4. `assemble_markdown(elements) -> str`:
   Sort by reading_order, skip SKIP_LABELS, apply PROMPT_MAP, join with "\n\n".

5. `save_to_json(result, output_dir)`:
   - Create output_dir if needed.
   - Prefer result.full_markdown (SDK output) over assembling from per-page markdown.
   - Write {stem}.md and {stem}.json.
   - JSON structure: {source_file, total_elements, pages: [{page_num, elements: [
     {label, text, bbox, score, reading_order}], markdown}]}

---

## Prompt B — Structure-Aware Chunker

Write `src/doc_parser/chunker.py`.

I need a `Chunk` dataclass:
  text, chunk_id (format: "{source_file}_{page}_{idx}"), page, element_types (list[str]),
  bbox (list[float] | None), source_file, is_atomic (bool), modality (str, default "text"),
  image_base64 (str | None), caption (str | None).

Rules for chunking:
- ATOMIC_LABELS = {table, formula, inline_formula, algorithm, image, figure}
  → each always gets its own chunk, never merged.
- TITLE_LABELS = {document_title, paragraph_title, figure_title}
  → a title attaches FORWARD to the next content element, not backward.
  → figure_title specifically is a figure caption — it must be prepended to the
    next image/figure atomic chunk, co-locating caption + visual in one chunk.
- Regular text elements accumulate until max_chunk_tokens is reached.
- Token estimation: word_count * 1.3 (BPE heuristic — no dependency on tiktoken).
- If a single element exceeds max_chunk_tokens, split it on whitespace boundaries.
- Modality inference from element_types: image/figure → "image", table → "table",
  formula/inline_formula → "formula", algorithm → "algorithm", else → "text".

Write TWO public functions:

1. `structure_aware_chunking(elements, source_file, page, max_chunk_tokens=512) -> list[Chunk]`
   Single-page chunker. Internally delegates to document_aware_chunking.

2. `document_aware_chunking(pages: list[tuple[int, list[ElementLike]]], source_file,
                             max_chunk_tokens=512) -> list[Chunk]`
   Multi-page chunker that processes all pages as one flat element stream.
   This prevents a heading at the bottom of page N becoming an orphan — it
   correctly attaches to content at the top of page N+1.
   The chunk's `page` field is set to the page of its FIRST element.

The flush_current() inner function should handle orphan titles (a title with no
following content before end-of-document) by emitting them as their own chunk.

---

## What this produced

- `src/doc_parser/post_processor.py` — Markdown assembly and JSON serialisation
- `src/doc_parser/chunker.py` — Chunk dataclass, structure_aware_chunking,
  document_aware_chunking
