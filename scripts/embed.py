"""Generate resumable per-document BGE embeddings from data/chunks/*.json.

Git Bash:
    python -m scripts.embed --limit-files 2
    python -m scripts.embed
    python -m scripts.embed --audit-only

Saved .npz files are NumPy archives containing `embeddings`, `chunk_ids`,
`source_sha256`, `model_name`, `dimension`, and `normalized`. Step 6 loads
these archives and aligns each vector with the corresponding chunk text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable

import numpy as np

from src.embedder import BGEEmbedder, EMBEDDING_DIM, MODEL_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {"embeddings", "chunk_ids", "source_sha256", "model_name", "dimension", "normalized"}


def read_chunks(path: Path) -> tuple[list[dict], str]:
    """Check all chunk IDs against their text and precise PDF source metadata."""
    raw = path.read_bytes()
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{path.name}: expected a nonempty JSON array")
    expected_pdf = f"{path.stem}.pdf"
    seen: set[str] = set()
    for i, row in enumerate(parsed, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}: chunk {i} is not an object")
        chunk_id = row.get("chunk_id")
        text = row.get("text")
        page = row.get("page_number")
        start = row.get("start_char")
        end = row.get("end_char")
        index = row.get("chunk_index")
        if row.get("filename") != expected_pdf or not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path.name}: chunk {i} has invalid source name or text")
        if any(type(x) is not int for x in (page, start, end, index)) or page < 1 or index < 1 or start < 0 or end <= start or end - start != len(text):
            raise ValueError(f"{path.name}: chunk {i} has invalid page/index/offsets")
        identity = f"{expected_pdf}\x00{page}\x00{start}\x00{end}\x00{text}"
        expected_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if chunk_id != expected_id or chunk_id in seen:
            raise ValueError(f"{path.name}: chunk {i} has altered or repeated chunk_id")
        seen.add(chunk_id)
    return parsed, hashlib.sha256(raw).hexdigest()


def validate_archive(path: Path, chunks: list[dict], source_sha256: str) -> int:
    """Return vector count if saved embeddings exactly match this source file."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_KEYS:
            raise ValueError("Unexpected .npz fields")
        vectors = archive["embeddings"]
        ids = archive["chunk_ids"].tolist()
        if (
            str(archive["model_name"].item()) != MODEL_NAME
            or str(archive["source_sha256"].item()) != source_sha256
            or int(archive["dimension"].item()) != EMBEDDING_DIM
            or bool(archive["normalized"].item()) is not True
        ):
            raise ValueError("Model settings or chunk source have changed")
        if ids != [chunk["chunk_id"] for chunk in chunks]:
            raise ValueError("Stored chunk IDs do not match current chunks")
        if vectors.dtype != np.float32 or vectors.shape != (len(chunks), EMBEDDING_DIM):
            raise ValueError("Incorrect vector dtype or dimensions")
        if not np.isfinite(vectors).all():
            raise ValueError("Saved vectors contain NaN or infinity")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=0.02, rtol=0):
            raise ValueError("Saved vectors are not L2-normalized")
    return len(chunks)


def write_archive(path: Path, chunks: list[dict], vectors: np.ndarray, source_sha256: str) -> None:
    """Write a validated .npz atomically; never leave partial output files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                embeddings=np.asarray(vectors, dtype=np.float32),
                chunk_ids=np.asarray([c["chunk_id"] for c in chunks], dtype="<U64"),
                source_sha256=np.asarray(source_sha256),
                model_name=np.asarray(MODEL_NAME),
                dimension=np.asarray(EMBEDDING_DIM),
                normalized=np.asarray(True),
            )
            handle.flush()
            os.fsync(handle.fileno())
        validate_archive(temporary, chunks, source_sha256)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate_embeddings(
    input_dir: Path,
    output_dir: Path,
    *,
    limit_files: int | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    audit_only: bool = False,
    embedder_factory: Callable[[], BGEEmbedder] | None = None,
) -> dict:
    """Save and audit one NumPy archive per chunk JSON document.

    Existing valid archives are reused without loading the neural model.
    """
    if limit_files is not None and (type(limit_files) is not int or limit_files < 1):
        raise ValueError("--limit-files must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory missing: {input_dir}")
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No chunk JSON files in {input_dir}")
    if limit_files is not None:
        files = files[:limit_files]
    output_dir = Path(output_dir)
    used: set[str] = set()
    model = None
    summary = {"documents": 0, "embeddings": 0, "unique_ids": 0, "reused": 0, "saved": 0, "dimension": EMBEDDING_DIM}

    for file_number, input_path in enumerate(files, 1):
        chunks, digest = read_chunks(input_path)
        ids = [c["chunk_id"] for c in chunks]
        duplicates = used.intersection(ids)
        if duplicates:
            raise ValueError(f"Duplicate chunk ID across documents: {input_path.name}")
        used.update(ids)
        destination = output_dir / f"{input_path.stem}.npz"
        if destination.is_file():
            try:
                count = validate_archive(destination, chunks, digest)
            except (OSError, ValueError, KeyError, EOFError):
                if audit_only:
                    raise ValueError(f"Outdated or corrupt saved vectors: {destination.name}") from None
            else:
                summary["reused"] += 1
                summary["documents"] += 1
                summary["embeddings"] += count
                print(f"[{file_number}/{len(files)}] {input_path.name} | reused | {count} embeddings", flush=True)
                continue
        if audit_only:
            raise FileNotFoundError(f"Missing saved vectors: {destination.name}")
        if model is None:
            model = embedder_factory() if embedder_factory is not None else BGEEmbedder(device=device)
            if int(model.dimension) != EMBEDDING_DIM:
                raise ValueError("Embedder has incorrect vector dimensions")
        all_vectors: list[np.ndarray] = []
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            try:
                vectors = model.encode_passages([c["text"] for c in batch], batch_size=batch_size)
            except (ValueError, RuntimeError) as exc:
                first = batch[0]
                raise ValueError(
                    f"{input_path.name}, page {first['page_number']}, batch starting at "
                    f"chunk {offset + 1}: {exc}"
                ) from exc
            all_vectors.append(np.asarray(vectors, dtype=np.float32))
        vectors = np.concatenate(all_vectors, axis=0)
        if vectors.shape != (len(chunks), EMBEDDING_DIM):
            raise ValueError(f"Unexpected embedding dimensions for {input_path.name}: {vectors.shape}")
        write_archive(destination, chunks, vectors, digest)
        summary["saved"] += 1
        summary["documents"] += 1
        summary["embeddings"] += len(chunks)
        print(f"[{file_number}/{len(files)}] {input_path.name} | saved | {len(chunks)} embeddings", flush=True)

    summary["unique_ids"] = len(used)
    if summary["embeddings"] != summary["unique_ids"]:
        raise RuntimeError("Mismatch between embeddings and unique source IDs")
    print(f"Embedding audit: {summary['documents']} documents; {summary['embeddings']} embeddings; "
          f"{summary['unique_ids']} distinct IDs; dimension {EMBEDDING_DIM}; "
          f"{summary['saved']} saved; {summary['reused']} reused")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-files", type=int, help="Only process first N alphabetical chunk JSON files")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--audit-only", action="store_true", help="Validate saved vectors without loading the model")
    args = parser.parse_args()
    try:
        generate_embeddings(
            PROJECT_ROOT / "data/chunks", PROJECT_ROOT / "data/embeddings",
            limit_files=args.limit_files, batch_size=args.batch_size,
            device=args.device, audit_only=args.audit_only,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Embedding step failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
