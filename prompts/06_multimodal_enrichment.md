# Prompt 06 — Multimodal Chunk Enrichment (GPT-4o Captioning)

## Stage
Use GPT-4o vision to generate searchable descriptions for images, tables,
formulas, and algorithms before they are embedded.

---

## Prompt

Write `src/doc_parser/ingestion/image_captioner.py`.

The problem with multimodal documents is that images, tables, and formulas
don't embed well as raw text.  I want to enrich every non-text chunk by
calling GPT-4o to generate a structured description that can be embedded and
retrieved semantically.

For IMAGES:
- Crop the exact PDF region using `pdf_page_to_image(pdf_path, page-1, dpi=150)`
  then `PIL.Image.crop((x1, y1, x2, y2))` where coords are scaled from the
  0–1000 bbox_2d normalisation to pixel space.
- Skip crops smaller than 50×50 pixels (likely detection noise) — set text="[figure]".
- Encode cropped PNG as base64 and send as an OpenAI vision message.
- System prompt asks for a response in EXACTLY this format:
    CAPTION: <1-2 sentence description for semantic search>
    FLOW: <numbered step-by-step description of process/sequence>
    STRUCTURE: <grouping and containment relationships>
- Store: chunk.caption = short CAPTION line, chunk.text = full structured response,
  chunk.image_base64 = base64 PNG (for the reranker and API responses).

For TABLES:
- Send the raw text content (truncated to 3000 chars).
- System prompt asks for SUMMARY + DETAIL bullet points about what the table measures.
- chunk.caption = original raw text, chunk.text = enriched description.

For FORMULAS:
- Send the LaTeX/text content.
- System prompt asks for SUMMARY (plain English meaning) + DETAIL (symbol definitions).
- Same caption/text swap.

For ALGORITHMS:
- Send the pseudocode.
- System prompt asks for SUMMARY (what it does) + DETAIL (inputs/outputs, steps, complexity).
- Same caption/text swap.

For TEXT chunks: no enrichment.

All four enrichment helpers run concurrently via `asyncio.gather` with a shared
`asyncio.Semaphore(max_concurrent=5)` to avoid overwhelming the OpenAI rate limiter.

Public API:
- `enrich_chunks(chunks, pdf_path, client, model="gpt-4o", max_concurrent=5) -> list[Chunk]`
  Dispatches by chunk.modality.
- `enrich_image_chunks(...)` — backward-compatibility alias for ingest.py callers.

---

## Why this matters

Without enrichment, an image chunk just has text="[figure]" which embeds poorly.
After enrichment it has a rich structured description covering the figure's content,
structure, and relationships — making it retrievable by natural language queries.

---

## What this produced

- `src/doc_parser/ingestion/image_captioner.py`
