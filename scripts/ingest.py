"""Extract PDFs into per-document JSON arrays of raw page records."""

import json
from pathlib import Path
import sys

from src.loader import load_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ingest_pdfs(input_dir: Path, output_dir: Path) -> None:
    """Write raw page records for PDFs in deterministic filename order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pdf_paths = sorted(
        (
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name,
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdf_paths:
        try:
            records = load_pdf(pdf_path)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"Cannot read PDF '{pdf_path}': {error}") from error

        output_path = output_dir / f"{pdf_path.stem}.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")

        empty_pages = [
            record["page_number"] for record in records
            if not record["text"].strip()
        ]
        print(f"{pdf_path.name}: {len(records)} pages")
        if empty_pages:
            print(f"  Empty-text pages: {', '.join(map(str, empty_pages))}")
        print(f"  Output: {output_path}")


def main() -> int:
    """Run raw PDF ingestion and report actionable errors to stderr."""
    try:
        ingest_pdfs(PROJECT_ROOT / "data/documents", PROJECT_ROOT / "data/raw")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
