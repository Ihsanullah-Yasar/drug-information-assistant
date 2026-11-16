"""Step 7: source-grounded Top-K semantic retrieval from the local ChromaDB.

The index and BGE model come from Steps 5 and 6. This module does not
write to ChromaDB, regenerate passage embeddings, or generate answers.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import numpy as np

from src.embedder import EMBEDDING_DIM

DEFAULT_TOP_K = 5
MAX_TOP_K = 50


def _validate_filename(filename: str | None) -> str | None:
    if filename is None:
        return None
    if (not isinstance(filename, str) or not filename.strip()
            or filename.strip() != filename or "/" in filename or "\\" in filename
            or filename in {".", ".."} or not filename.lower().endswith(".pdf")):
        raise ValueError("--filename must be the exact PDF filename, without a path")
    return filename


def _validate_metadata(md: Any, *, text: str, filename_filter: str | None) -> None:
    """Fail closed if stored provenance is incomplete or cannot be cited."""
    if not isinstance(md, dict):
        raise ValueError("Chroma returned missing or invalid source metadata")
    filename = md.get("filename")
    if (not isinstance(filename, str) or not filename.lower().endswith(".pdf")
            or not filename or "/" in filename or "\\" in filename):
        raise ValueError("Chroma result has an invalid source PDF filename")
    if filename_filter is not None and filename != filename_filter:
        raise ValueError("Chroma returned a document outside the requested PDF")
    if any(type(md.get(field)) is not int for field in
           ("page_number", "chunk_index", "start_char", "end_char")):
        raise ValueError("Chroma result has invalid page/chunk offsets")
    if (md["page_number"] < 1 or md["chunk_index"] < 1
            or md["start_char"] < 0 or md["end_char"] <= md["start_char"]
            or md["end_char"] - md["start_char"] != len(text)):
        raise ValueError("Chroma result contains inconsistent source text offsets")
    url = md.get("source_url")
    if not isinstance(url, str):
        raise ValueError("Chroma result has no DailyMed source URL")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "dailymed.nlm.nih.gov", "www.dailymed.nlm.nih.gov"
    }:
        raise ValueError("Chroma result has an untrusted source URL")
    for optional in ("packager", "label_updated_on"):
        if optional in md and not isinstance(md[optional], str):
            raise ValueError(f"Chroma result has invalid {optional}")


def retrieve_chunks(
    question: str,
    *,
    collection: Any,
    embedder: Any,
    top_k: int = DEFAULT_TOP_K,
    filename: str | None = None,
    expected_count: int | None = None,
) -> list[dict]:
    """Retrieve ranked source text; similarity is NOT an answerability score.

    Uses BGEEmbedder.encode_query() (with the BGE query instruction), then
    collection.query(query_embeddings=...). Does not request Chroma text
    embedding and does not modify the collection.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must contain nonempty text")
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
        raise ValueError("expected_count must be a positive integer")
    filename = _validate_filename(filename)
    total = collection.count()
    if type(total) is not int or total < 1:
        raise ValueError("The Chroma collection is empty. Complete Step 6 indexing first.")
    if expected_count is not None and total != expected_count:
        raise ValueError(f"Expected {expected_count} indexed vectors, found {total}; audit Step 6")

    vector = np.asarray(embedder.encode_query(question.strip()), dtype=np.float32)
    if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"Query embedding must be {EMBEDDING_DIM} finite numbers")
    if not np.isclose(np.linalg.norm(vector), 1.0, rtol=0, atol=0.02):
        raise ValueError("Query vector must be L2-normalized")

    kwargs: dict[str, Any] = {
        "query_embeddings": [vector.tolist()],
        "n_results": min(top_k, total),
        "include": ["documents", "metadatas", "distances"],
    }
    if filename is not None:
        kwargs["where"] = {"filename": filename}
    raw = collection.query(**kwargs)
    if not isinstance(raw, dict):
        raise ValueError("Chroma returned an invalid query response")

    fields: dict[str, list] = {}
    for field in ("ids", "documents", "metadatas", "distances"):
        values = raw.get(field)
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            raise ValueError(f"Chroma response must contain one query in {field}")
        if not isinstance(values[0], (list, tuple)):
            raise ValueError(f"Invalid Chroma {field} results")
        fields[field] = values[0]
    count = len(fields["ids"])
    if count > min(top_k, total) or any(len(v) != count for v in fields.values()):
        raise ValueError("Chroma returned misaligned query result columns")

    matches: list[dict] = []
    seen: set[str] = set()
    previous_distance = float("-inf")
    for index in range(count):
        chunk_id = fields["ids"][index]
        doc = fields["documents"][index]
        md = fields["metadatas"][index]
        distance = fields["distances"][index]
        if (not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen
                or not isinstance(doc, str) or not doc.strip()):
            raise ValueError("Chroma returned duplicate/invalid chunk ID or text")
        seen.add(chunk_id)
        if type(distance) not in (int, float, np.float32, np.float64) or not np.isfinite(distance):
            raise ValueError("Chroma returned a nonfinite cosine distance")
        distance = float(distance)
        if distance < -1e-5 or distance > 2.00001:
            raise ValueError("Chroma returned an impossible cosine distance")
        if distance + 1e-6 < previous_distance:
            raise ValueError("Chroma results are not sorted by ascending distance")
        previous_distance = distance
        _validate_metadata(md, text=doc, filename_filter=filename)
        matches.append({
            "rank": index + 1,
            "chunk_id": chunk_id,
            "text": doc,
            "filename": md["filename"],
            "page_number": md["page_number"],
            "chunk_index": md["chunk_index"],
            "start_char": md["start_char"],
            "end_char": md["end_char"],
            "source_url": md["source_url"],
            "packager": md.get("packager", ""),
            "label_updated_on": md.get("label_updated_on", ""),
            "cosine_distance": distance,
            "cosine_similarity": max(-1.0, min(1.0, 1.0 - distance)),
        })
    return matches
