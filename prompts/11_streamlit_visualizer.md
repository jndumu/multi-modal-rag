# Prompt 11 — Streamlit Document Visualizer (app.py)

## Stage
Build an interactive UI for uploading a PDF and seeing GLM-OCR's detected
elements overlaid as colored bounding boxes.

---

## Prompt

Write `app.py` — a Streamlit app that lets me upload a PDF, parse it with
DocumentParser, and see every detected element as a colored bounding box
overlaid on the rendered page image.

### Layout

Wide page layout.  Title: "GLM-OCR Document Visualizer".

**Sidebar:**
- File uploader (accepts .pdf).
- "Parse Document" button.
- "Show element text" checkbox.
- "Show page Markdown" checkbox.

**Main area:**
- Page slider (1 to N).
- Left column (3/5 width): annotated page image + legend.
- Right column (2/5 width): element breakdown counts + expandable element list.
- Full-width expander at bottom: full document Markdown.

---

### Bounding box rendering

The GLM-OCR MaaS API normalises all bbox_2d coordinates to a 0–1000 scale.
To convert to pixel coordinates:
```
pixel_x = bbox_x * rendered_width  / 1000
pixel_y = bbox_y * rendered_height / 1000
```

Render pages with PyMuPDF at RENDER_DPI=150, then draw on PIL using ImageDraw.

Each element box: semi-transparent filled rectangle (alpha=35) + solid outline
(alpha=220) + a small colored badge label in the top-left corner.

---

### Color map

Define a LABEL_COLORS dict mapping all 17 label types to (R, G, B) tuples:
- document_title  → red       (220, 50, 50)
- paragraph_title → blue      (30, 100, 220)
- abstract        → teal      (20, 160, 160)
- paragraph/text  → green     (40, 160, 40)
- table           → orange    (230, 120, 0)
- formula         → purple    (150, 50, 220)
- inline_formula  → light purple (180, 80, 220)
- figure_caption/caption → cyan (0, 180, 200)
- code_block      → yellow    (200, 180, 0)
- algorithm       → magenta   (220, 0, 180)
- footnotes       → pink      (200, 80, 120)
- reference       → brown     (140, 80, 40)
- header/footer   → light gray (160, 160, 160)
- page_number     → gray      (120, 120, 120)
- image           → dark gray (100, 100, 100)
- seal            → near black (60, 60, 60)
Unknown labels fall back to yellow (180, 180, 0).

---

### Session state

Use `st.session_state` to cache the ParseResult across widget reruns.
Only re-parse when the user uploads a DIFFERENT file (compare by filename).
Comparing by filename avoids re-parsing on every slider/checkbox interaction.

---

### Legend

Below the page image, render a color-coded legend for all label types
present on the current page. Use inline HTML spans with background color.
Display in 4 columns.

---

## What this produced

- `app.py` — Streamlit visualizer (247 lines)
