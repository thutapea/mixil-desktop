"""
One-time script to extract text from all PDFs in datasheets/.
Output saved to processed/<filename>.txt — already-processed files are skipped.

Strategy (cheapest first):
  1. Try pymupdf direct text extraction (free, instant).
  2. If a page has no selectable text (scanned/image-only), fall back to
     OpenAI gpt-4o-mini vision OCR (~$0.001/page).

Most instrument datasheets (Agilent, Keithley, NI) are digitally generated
and will be extracted for free. Only genuinely scanned pages cost anything.

Usage:
    export OPENAI_API_KEY=sk-...   # only needed if any scanned pages exist
    python scripts/ocr_datasheets.py

    # dry-run: see how many pages would need OCR before spending anything
    python scripts/ocr_datasheets.py --dry-run
"""

import base64
import os
import sys
from pathlib import Path

import fitz  # pymupdf

ROOT = Path(__file__).parent.parent
DATASHEETS_DIR = ROOT / "datasheets"
PROCESSED_DIR = ROOT / "processed"
OCR_MODEL = "gpt-4o-mini"

# A page is considered "text-free" (needs OCR) if it has fewer than this many chars.
MIN_TEXT_CHARS = 50


def extract_page_text(page: fitz.Page) -> str:
    """Extract selectable text from a PDF page."""
    return page.get_text().strip()


def page_to_b64(page: fitz.Page) -> str:
    """Render a PDF page to a base64-encoded PNG at 2x zoom."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    return base64.b64encode(pix.tobytes("png")).decode()


def ocr_page_openai(client, page: fitz.Page) -> str:
    """Call gpt-4o-mini vision to extract text from a single image-only page."""
    resp = client.chat.completions.create(
        model=OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this datasheet page. "
                            "Preserve tables, spec lists, and section headers. "
                            "Output extracted text only — no commentary."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{page_to_b64(page)}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return resp.choices[0].message.content


def process_pdf(path: Path, client=None, dry_run: bool = False) -> str:
    """
    Extract text from all pages. Uses direct extraction where possible,
    falls back to OCR for image-only pages.
    Returns the full extracted text as a string.
    """
    doc = fitz.open(path)
    pages_text = []
    ocr_count = 0
    text_count = 0

    for i, page in enumerate(doc):
        label = f"page {i + 1}/{len(doc)}"
        direct = extract_page_text(page)

        if len(direct) >= MIN_TEXT_CHARS:
            # Good selectable text — use it directly
            pages_text.append(f"=== Page {i + 1} ===\n{direct}")
            text_count += 1
            print(f"  {label}  [text]    ", end="\r", flush=True)
        else:
            # Image-only page — needs OCR
            ocr_count += 1
            if dry_run:
                pages_text.append(f"=== Page {i + 1} ===\n[would OCR this page]")
                print(f"  {label}  [OCR needed]", end="\r", flush=True)
            elif client is None:
                pages_text.append(f"=== Page {i + 1} ===\n[skipped: no API key for OCR]")
                print(f"  {label}  [OCR skipped — no API key]", end="\r", flush=True)
            else:
                print(f"  {label}  [OCR...]  ", end="\r", flush=True)
                try:
                    ocr_text = ocr_page_openai(client, page)
                    pages_text.append(f"=== Page {i + 1} ===\n{ocr_text}")
                except Exception as e:
                    pages_text.append(f"=== Page {i + 1} ===\n[OCR failed: {e}]")

    print(f"  Done: {text_count} text pages, {ocr_count} OCR pages.    ")
    return "\n\n".join(pages_text)


def main():
    dry_run = "--dry-run" in sys.argv

    # Only initialize OpenAI client if we have a key
    key = os.environ.get("OPENAI_API_KEY")
    client = None
    if key:
        from openai import OpenAI
        client = OpenAI(api_key=key)
    elif not dry_run:
        print("Note: OPENAI_API_KEY not set — image-only pages will be skipped.")
        print("      Set the key and re-run (already-processed files are skipped).\n")

    PROCESSED_DIR.mkdir(exist_ok=True)

    pdfs = sorted(DATASHEETS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATASHEETS_DIR}")
        sys.exit(0)

    print(f"Found {len(pdfs)} PDFs in {DATASHEETS_DIR.name}/")
    if dry_run:
        print("Dry-run mode: no API calls will be made.\n")
    print()

    total_ocr_pages = 0

    for pdf in pdfs:
        out = PROCESSED_DIR / (pdf.stem + ".txt")
        if out.exists() and not dry_run:
            print(f"  [skip]  {pdf.name}  (already processed)")
            continue

        print(f"  {pdf.name}")
        try:
            text = process_pdf(pdf, client=client, dry_run=dry_run)
            if not dry_run:
                out.write_text(text, encoding="utf-8")
                print(f"  -> saved {out.name}")
            else:
                # Count OCR-needed lines for cost estimate
                ocr_pages = text.count("[would OCR this page]")
                total_ocr_pages += ocr_pages
        except Exception as e:
            print(f"  -> failed: {e}")
        print()

    if dry_run:
        est_cost = total_ocr_pages * 0.001
        print(f"Dry-run summary: {total_ocr_pages} pages would need OCR.")
        print(f"Estimated cost: ${est_cost:.3f}  (at ~$0.001/page for gpt-4o-mini)")
    else:
        print("Done. Run:  python src/version4.py")


if __name__ == "__main__":
    main()
