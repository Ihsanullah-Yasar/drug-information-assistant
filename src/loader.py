"""Load a PDF into page-level raw text records for the local RAG pipeline.

Step 2 intentionally performs *extraction only*. Cleaning, chunking, and
embedding are separate stages of the assignment.
"""

from pathlib import Path

import pymupdf


def load_pdf(pdf_path: str | Path) -> list[dict[str, str | int]]:
    """Return one raw-text record for every page in a PDF.

    Example record::
        {
            "filename": "ibuprofen_200mg_example.pdf",
            "page_number": 1,
            "text": "Text extracted from page 1...",
        }

    Notes:
      * Page numbers start at 1 (human-readable citations).
      * Empty-text pages remain in the output; do not silently drop pages.
      * This does not clean, split, summarize, or alter the source PDF.
      * An image-only page may have empty text. OCR is outside this step.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {path}")

    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                raise ValueError(f"File is not a PDF: {path}")
            if document.needs_pass:
                raise ValueError(f"Password-protected PDF is not supported: {path}")
            if document.page_count < 1:
                raise ValueError(f"PDF has no pages: {path}")

            pages: list[dict[str, str | int]] = []
            for page_number, page in enumerate(document, start=1):
                # sort=True attempts top-to-bottom reading order.
                # It is not a guarantee for complex tables/multi-column pages.
                text = page.get_text("text", sort=True)
                pages.append(
                    {
                        "filename": path.name,
                        "page_number": page_number,
                        "text": text,
                    }
                )
            return pages

    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
        raise RuntimeError(f"PDF could not be opened or is damaged: {path}") from error
