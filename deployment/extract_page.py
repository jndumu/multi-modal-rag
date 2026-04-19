"""
deployment/extract_page.py — Extract a single page from a PDF.

Useful for testing: large PDFs can time out on the cloud API.
Extract page 1 first, verify the pipeline works, then ingest the full doc.

Usage:
    uv run python deployment/extract_page.py <input.pdf> <page_number> [output.pdf]

    page_number is 0-based (0 = first page)

Examples:
    uv run python deployment/extract_page.py docling_report.pdf 0
    uv run python deployment/extract_page.py docling_report.pdf 0 data/raw/page1.pdf
    uv run python deployment/extract_page.py paper.pdf 2 data/raw/paper_page3.pdf
"""
import sys
from pathlib import Path


def extract_page(input_path: str, page_number: int, output_path: str | None = None) -> Path:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF not installed. Run: uv pip install pymupdf")
        sys.exit(1)

    src = Path(input_path)
    if not src.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    doc = fitz.open(str(src))
    n_pages = len(doc)

    if page_number < 0 or page_number >= n_pages:
        print(f"ERROR: Page {page_number} out of range. Document has {n_pages} page(s) (0-based).")
        sys.exit(1)

    if output_path is None:
        out = src.parent / f"{src.stem}_page{page_number + 1}.pdf"
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)
    new_doc.save(str(out))
    new_doc.close()
    doc.close()

    print(f"Extracted page {page_number + 1} of {n_pages} from {src.name}")
    print(f"Saved to: {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_pdf = sys.argv[1]
    page_num = int(sys.argv[2])
    output_pdf = sys.argv[3] if len(sys.argv) > 3 else None

    extract_page(input_pdf, page_num, output_pdf)
