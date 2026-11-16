"""Step 6 — index the existing BGE .npz files in local ChromaDB.

Git Bash:
    python -m scripts.index_vectors --limit-files 2
    python -m scripts.index_vectors --limit-files 2 --audit-only
    python -m scripts.index_vectors
    python -m scripts.index_vectors --audit-only

The index always uses the precomputed BGE vectors; it NEVER generates new ones.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

import numpy as np

from scripts.embed import read_chunks, validate_archive
from src.embedder import EMBEDDING_DIM
from src.vectordb import (
    COLLECTION_NAME, build_metadata, connect_collection,
    records_match, retrieve_document_records, upsert_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = ["filename", "product_name", "packager", "label_updated_on", "source_url"]


def read_sources(path: Path) -> dict[str, dict[str, str]]:
    """Load the verified DailyMed manifest; reject duplicate identities."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"Expected CSV columns {CSV_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise ValueError("Source manifest is empty")
    sources: dict[str, dict[str, str]] = {}
    set_ids: set[str] = set()
    for row in rows:
        if set(row) != set(CSV_COLUMNS) or any(not row[field].strip() for field in CSV_COLUMNS):
            raise ValueError("Invalid source manifest row")
        filename = row["filename"]
        if (Path(filename).name != filename or not filename.lower().endswith(".pdf")
                or filename.casefold() in {k.casefold() for k in sources}):
            raise ValueError(f"Duplicate/unsafe PDF filename: {filename}")
        parsed = urlparse(row["source_url"])
        if parsed.scheme != "https" or parsed.hostname not in {"dailymed.nlm.nih.gov", "www.dailymed.nlm.nih.gov"}:
            raise ValueError(f"Non-DailyMed source: {row['source_url']}")
        setid = parse_qs(parsed.query).get("setid")
        if not setid or len(setid) != 1 or setid[0].casefold() in set_ids:
            raise ValueError(f"Missing/duplicate DailyMed Set ID: {filename}")
        set_ids.add(setid[0].casefold())
        sources[filename] = row
    return sources


def read_pair(chunk_path: Path, archive_path: Path, source: dict) -> tuple[list[dict], np.ndarray, list[dict]]:
    """Cryptographically match a per-document chunk JSON with its NPZ archive."""
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing embeddings {archive_path.name}. Complete Step 5 first or use --limit-files."
        )
    chunks, source_sha = read_chunks(chunk_path)
    validate_archive(archive_path, chunks, source_sha)
    with np.load(archive_path, allow_pickle=False) as archive:
        vectors = np.array(archive["embeddings"], dtype=np.float32, copy=True)
    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        raise ValueError(f"Vector count/dimension mismatch for {chunk_path.name}")
    metadatas = [
        build_metadata(row, source=source, source_sha256=source_sha, vector=vector)
        for row, vector in zip(chunks, vectors, strict=True)
    ]
    return chunks, vectors, metadatas


def index_vectors(
    chunk_dir: Path,
    embedding_dir: Path,
    db_dir: Path,
    sources_csv: Path,
    *,
    limit_files: int | None = None,
    batch_size: int = 64,
    audit_only: bool = False,
    client=None,
) -> dict:
    """Index or audit a deterministic subset, preserving prior successful work.

    Audit-only never creates a collection or mutates records. A full audit
    ensures that the collection contains exactly the expected chunk IDs.
    """
    if limit_files is not None and (type(limit_files) is not int or limit_files < 1):
        raise ValueError("limit_files must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Missing chunk directory: {chunk_dir}")
    paths = sorted(chunk_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No chunk JSON files in {chunk_dir}")
    if limit_files is not None:
        paths = paths[:limit_files]
    sources = read_sources(sources_csv)
    collection = connect_collection(db_dir, create=not audit_only, client=client)
    seen_ids: set[str] = set()
    summary = {
        "documents": 0, "vectors": 0, "unique_ids": 0,
        "indexed": 0, "reused": 0, "dimension": EMBEDDING_DIM,
        "collection": COLLECTION_NAME,
    }
    for number, path in enumerate(paths, start=1):
        source = sources.get(f"{path.stem}.pdf")
        if source is None:
            raise ValueError(f"No DailyMed source CSV row for {path.name}")
        chunks, vectors, metadatas = read_pair(path, embedding_dir / f"{path.stem}.npz", source)
        ids = {chunk["chunk_id"] for chunk in chunks}
        if ids.intersection(seen_ids) or len(ids) != len(chunks):
            raise ValueError(f"Duplicate chunk IDs across documents: {path.name}")
        seen_ids.update(ids)
        if audit_only:
            existing = retrieve_document_records(collection, source["filename"])
            if not records_match(existing, chunks, vectors, metadatas):
                raise ValueError(f"Missing, stale, or incorrect Chroma records for {path.name}")
            status = "verified"
        else:
            status = upsert_document(
                collection, chunks=chunks, vectors=vectors,
                metadatas=metadatas, batch_size=batch_size,
            )
            summary["indexed" if status == "indexed" else "reused"] += 1
        summary["documents"] += 1
        summary["vectors"] += len(chunks)
        print(f"[{number}/{len(paths)}] {path.name} | {status} | {len(chunks)} vectors", flush=True)

    summary["unique_ids"] = len(seen_ids)
    if summary["vectors"] != summary["unique_ids"]:
        raise RuntimeError("Vector count differs from distinct chunk ID count")
    total_count = collection.count()
    summary["collection_count"] = total_count
    # Only the full set of source files may claim the whole collection is exact.
    if limit_files is None and total_count != summary["vectors"]:
        raise ValueError(
            f"Chroma collection contains {total_count} records, but the current corpus "
            f"has {summary['vectors']} chunks. Check for unrelated/stale documents."
        )
    print(
        f"Chroma audit: {summary['documents']} documents checked; "
        f"{summary['vectors']} vectors; {summary['unique_ids']} distinct IDs; "
        f"dimension {EMBEDDING_DIM}; collection records {total_count}; "
        f"{summary['indexed']} indexed; {summary['reused']} reused"
    )
    if limit_files is not None:
        print("Partial audit: the entire 100-document collection has NOT yet been verified.")
    else:
        print("Full index audit passed: all expected sources and vectors match ChromaDB.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-files", type=int, help="Index only first N alphabetically sorted documents")
    parser.add_argument("--batch-size", type=int, default=64, help="Vectors per ChromaDB write batch")
    parser.add_argument("--audit-only", action="store_true", help="Validate existing index without writing")
    args = parser.parse_args()
    try:
        index_vectors(
            PROJECT_ROOT / "data/chunks", PROJECT_ROOT / "data/embeddings",
            PROJECT_ROOT / "data/chroma", PROJECT_ROOT / "data/sources.csv",
            limit_files=args.limit_files, batch_size=args.batch_size,
            audit_only=args.audit_only,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Chroma indexing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
