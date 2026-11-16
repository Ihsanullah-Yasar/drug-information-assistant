"""Clean page-level JSON files created by `python -m scripts.ingest`."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

from src.cleaner import clean_document

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, payload: list[dict]) -> None:
    """Write a completed JSON file atomically so interrupted runs are safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clean_documents(input_dir: Path, output_dir: Path) -> dict:
    """Clean all raw JSON documents and return an auditable summary."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Raw JSON directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No raw JSON files found in: {input_dir}")

    # Avoid silently counting stale output files from an earlier collection.
    existing = {p.name for p in output_dir.glob("*.json")} if output_dir.exists() else set()
    incoming = {p.name for p in files}
    if extra := sorted(existing - incoming):
        raise ValueError(f"Unexpected stale cleaned JSON file(s): {', '.join(extra)}")

    summary = {"documents": 0, "pages": 0, "nonempty_pages": 0, "empty_pages": []}
    for raw_path in files:
        try:
            with raw_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            cleaned = clean_document(records, expected_filename=f"{raw_path.stem}.pdf")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid raw document '{raw_path.name}': {error}") from error

        # Do not write an output until this document's entire set of pages passed validation.
        output_path = output_dir / raw_path.name
        _atomic_json(output_path, cleaned)
        empty = [r["page_number"] for r in cleaned if not r["text"]]
        summary["documents"] += 1
        summary["pages"] += len(cleaned)
        summary["nonempty_pages"] += len(cleaned) - len(empty)
        summary["empty_pages"].extend(
            {"filename": raw_path.stem + ".pdf", "page_number": n} for n in empty
        )
        print(f"[{summary['documents']}/{len(files)}] {raw_path.name} | {len(cleaned)} pages | {len(empty)} empty", flush=True)

    print(f"Cleaning audit: {summary['documents']} documents; {summary['pages']} pages; "
          f"{summary['nonempty_pages']} pages with text; {len(summary['empty_pages'])} empty-text pages")
    if summary["empty_pages"]:
        print("Empty-text pages preserved for review (they will not be embedded as empty chunks):")
        for entry in summary["empty_pages"]:
            print(f"  {entry['filename']} — page {entry['page_number']}")
    return summary


def main() -> int:
    try:
        clean_documents(PROJECT_ROOT / "data/raw", PROJECT_ROOT / "data/cleaned")
    except (OSError, ValueError, TypeError) as error:
        print(f"Cleaning failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
