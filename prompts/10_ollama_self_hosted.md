# Prompt 10 — Ollama Self-Hosted Pipeline

## Stage
Add a fully local alternative to the cloud MaaS backend using Ollama.

---

## Prompt

I want to support a second parser backend mode: PARSER_BACKEND=ollama.
In this mode the glmocr SDK connects to a local Ollama instance instead of
the Z.AI cloud API.

### ollama/config.yaml

Write a glmocr config with `maas.enabled: false` and point `ollama.endpoint`
to `http://localhost:11434`.  The model IDs should match the GLM-OCR models
pulled via `ollama pull`.

The config.yaml for Ollama mode is separate from the root config.yaml (which
is MaaS mode).  When PARSER_BACKEND=ollama the Settings validator auto-sets
config_yaml_path to "ollama/config.yaml".

---

### ollama/api_parse.py — Direct HTTP client

Write a script that sends a PDF page to the Ollama HTTP API at
`http://localhost:11434/api/generate` directly (without the glmocr SDK).

This is a debugging/inspection tool:
- Convert each PDF page to a base64 PNG using PyMuPDF.
- POST to Ollama with model name, prompt, and the base64 image.
- Parse the streaming JSON response and collect the generated elements.
- Print a summary of detected elements per page.

---

### ollama/test_parse.py — Smoke test

A simple script that:
- Instantiates DocumentParser() with PARSER_BACKEND=ollama.
- Parses a provided PDF file.
- Prints per-page element counts and the first 3 elements per page.
- Saves output to ollama/output/ (gitignored).

---

### ollama/visualize.py — Bounding box visualization

A script that renders parse results visually:
- For each page, render the PDF at 150 DPI with PyMuPDF.
- Draw bounding boxes from ParsedElement.bbox (0–1000 normalised coords).
- Color-code by label (same colors as app.py LABEL_COLORS).
- Save annotated images to ollama/output/visualizations/.
- Useful for comparing Ollama vs MaaS detection quality side-by-side.

---

### ollama/README.md

Document:
1. How to install Ollama.
2. Which models to pull (`ollama pull glm-ocr`).
3. How to set PARSER_BACKEND=ollama in .env.
4. Known differences vs cloud mode (pypdfium2 vs PyMuPDF page counts).
5. How to run test_parse.py and visualize.py.

---

## What this produced

- `ollama/config.yaml`
- `ollama/api_parse.py`
- `ollama/test_parse.py`
- `ollama/visualize.py`
- `ollama/README.md`
