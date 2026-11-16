"""Convert data/cleaned/*.json into page-cited data/chunks/*.json.

Usage (Git Bash): python -m scripts.chunk [--chunk-size 1000] [--overlap 180]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from src.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_document

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def chunk_documents(input_dir: Path, output_dir: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> dict:
    """Produce one JSON array per PDF and an aggregate audit (no external calls)."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Cleaned input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No cleaned JSON files found in {input_dir}")
    old = {p.name for p in output_dir.glob("*.json")} if output_dir.exists() else set()
    new = {p.name for p in files}
    if old - new:
        raise ValueError(f"Unexpected stale chunk files: {', '.join(sorted(old - new))}")
    summary = {"documents": 0, "pages": 0, "empty_pages": 0, "chunks": 0, "unique_ids": 0}
    ids: set[str] = set()
    for i, path in enumerate(files, 1):
        try:
            pages = json.loads(path.read_text(encoding="utf-8"))
            chunks = chunk_document(pages, expected_filename=f"{path.stem}.pdf", chunk_size=chunk_size, overlap=overlap)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid cleaned document {path.name}: {error}") from error
        page_empty = sum(not record["text"].strip() for record in pages)
        for chunk in chunks:
            if chunk["chunk_id"] in ids:
                raise ValueError(f"Duplicate chunk ID: {chunk['chunk_id']}")
            ids.add(chunk["chunk_id"])
        _atomic_json(output_dir / path.name, chunks)
        summary["documents"] += 1
        summary["pages"] += len(pages)
        summary["empty_pages"] += page_empty
        summary["chunks"] += len(chunks)
        print(f"[{i}/{len(files)}] {path.name} | {len(pages)} pages | {len(chunks)} chunks | {page_empty} empty pages", flush=True)
    summary["unique_ids"] = len(ids)
    print(f"Chunking audit: {summary['documents']} documents; {summary['pages']} pages; "
          f"{summary['empty_pages']} empty pages; {summary['chunks']} chunks; "
          f"{summary['unique_ids']} distinct chunk IDs")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    try:
        chunk_documents(PROJECT_ROOT / "data/cleaned", PROJECT_ROOT / "data/chunks", chunk_size=args.chunk_size, overlap=args.overlap)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Chunking failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
