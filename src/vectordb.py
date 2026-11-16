"""Persistent ChromaDB storage for precomputed BGE-small embeddings.

Step 6: keeps the neural embedding stage separate from vector indexing.
No LangChain / LlamaIndex / Haystack. Query embedding is implemented in Step 7.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.embedder import EMBEDDING_DIM, MODEL_NAME

COLLECTION_NAME = "drug_labels_bge_small_v15"
DISTANCE_METRIC = "cosine"
COLLECTION_METADATA = {
    "model_name": MODEL_NAME,
    "dimension": EMBEDDING_DIM,
    "distance_metric": DISTANCE_METRIC,
    "pipeline": "drug_ir_python_v1",
}


def connect_collection(db_dir: Path, *, create: bool = True, client: Any = None) -> Any:
    """Open a local persistent collection; refuse mismatched existing indexes.

    `client` is injectable for offline tests. Real usage uses PersistentClient.
    `create=False` does not create a collection during an audit-only check.
    """
    if client is None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Run: python -m pip install 'chromadb>=1.5,<2'"
            ) from exc
        db_dir = Path(db_dir)
        if create:
            db_dir.mkdir(parents=True, exist_ok=True)
        elif not db_dir.is_dir():
            raise FileNotFoundError(f"ChromaDB directory does not exist: {db_dir}")
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(db_dir), settings=Settings(anonymized_telemetry=False)
        )

    if create:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_METADATA,
            configuration={"hnsw": {"space": DISTANCE_METRIC}},
            embedding_function=None,
        )
    else:
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to open collection {COLLECTION_NAME!r}; run the indexing command first: {exc}"
            ) from exc

    metadata = collection.metadata or {}
    for key, expected in COLLECTION_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Incompatible Chroma collection metadata {key!r}: "
                f"expected {expected!r}, received {metadata.get(key)!r}. "
                "Use a new collection name for different model/index settings."
            )

    # Chroma >=1.0 uses `configuration` for immutable distance configuration.
    configuration = getattr(collection, "configuration", None)
    if isinstance(configuration, dict):
        hnsw = configuration.get("hnsw")
        if isinstance(hnsw, dict):
            space = hnsw.get("space")
            if space is not None and space != DISTANCE_METRIC:
                raise ValueError(f"Chroma collection must use cosine, found {space!r}")
    return collection


def vector_sha256(vector: np.ndarray) -> str:
    """Stable fingerprint of one float32 passage embedding."""
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (EMBEDDING_DIM,) or not np.isfinite(array).all():
        raise ValueError("Invalid vector for fingerprint")
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_metadata(chunk: dict, *, source: dict, source_sha256: str, vector: np.ndarray) -> dict:
    """Record enough provenance to display a PDF page citation later."""
    if chunk["filename"] != source["filename"]:
        raise ValueError("CSV filename does not match the chunk source")
    return {
        "filename": chunk["filename"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
        "start_char": chunk["start_char"],
        "end_char": chunk["end_char"],
        "source_url": source["source_url"],
        "packager": source["packager"],
        "label_updated_on": source["label_updated_on"],
        "source_sha256": source_sha256,
        "vector_sha256": vector_sha256(vector),
    }


def records_match(result: dict, chunks: list[dict], vectors: np.ndarray, metadatas: list[dict]) -> bool:
    """Compare ALL stored fields by chunk ID, independent of Chroma result order."""
    ids = result.get("ids")
    docs = result.get("documents")
    metadata = result.get("metadatas")
    stored_vectors = result.get("embeddings")
    if ids is None or docs is None or metadata is None or stored_vectors is None:
        return False
    if len(ids) != len(chunks) or len(set(ids)) != len(chunks):
        return False
    if len(docs) != len(ids) or len(metadata) != len(ids) or len(stored_vectors) != len(ids):
        return False
    by_id = {value: position for position, value in enumerate(ids)}
    for index, chunk in enumerate(chunks):
        position = by_id.get(chunk["chunk_id"])
        if position is None:
            return False
        if docs[position] != chunk["text"] or metadata[position] != metadatas[index]:
            return False
        found = np.asarray(stored_vectors[position], dtype=np.float32)
        if found.shape != (EMBEDDING_DIM,) or not np.isfinite(found).all():
            return False
        if not np.allclose(found, vectors[index], rtol=1e-5, atol=1e-6):
            return False
    return True


def retrieve_document_records(collection: Any, filename: str) -> dict:
    """Fetch all records associated with one source PDF for reconciliation."""
    return collection.get(
        where={"filename": filename},
        include=["documents", "metadatas", "embeddings"],
    )


def upsert_document(
    collection: Any,
    *,
    chunks: list[dict],
    vectors: np.ndarray,
    metadatas: list[dict],
    batch_size: int = 64,
) -> str:
    """Write the target records, then remove outdated IDs for the same PDF.

    If interrupted, rerunning repairs the partial document. Other PDFs are
    never removed. Chroma itself persists the data locally.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not chunks or vectors.shape != (len(chunks), EMBEDDING_DIM):
        raise ValueError("Chunk/vector mismatch")
    if len(metadatas) != len(chunks):
        raise ValueError("Chunk/metadata mismatch")

    filename = chunks[0]["filename"]
    if any(row["filename"] != filename for row in chunks):
        raise ValueError("Cannot write multiple PDFs as one document")
    existing = retrieve_document_records(collection, filename)
    if records_match(existing, chunks, vectors, metadatas):
        return "reused"

    ids = [row["chunk_id"] for row in chunks]
    target_ids = set(ids)
    stale = [old_id for old_id in existing.get("ids", []) if old_id not in target_ids]
    for offset in range(0, len(chunks), batch_size):
        part = slice(offset, offset + batch_size)
        collection.upsert(
            ids=ids[part],
            documents=[row["text"] for row in chunks[part]],
            metadatas=metadatas[part],
            embeddings=vectors[part].tolist(),
        )
    # Do NOT delete until the new records have been written.
    for offset in range(0, len(stale), batch_size):
        collection.delete(ids=stale[offset:offset + batch_size])

    confirmed = retrieve_document_records(collection, filename)
    if not records_match(confirmed, chunks, vectors, metadatas):
        raise RuntimeError(f"ChromaDB verification failed after indexing {filename}")
    return "indexed"
