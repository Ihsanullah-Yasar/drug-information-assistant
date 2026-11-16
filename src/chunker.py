"""Manual, page-preserving character chunker (no RAG frameworks).

The source of truth is data/cleaned/*.json. Chunks never cross page or
PDF boundaries. Offsets index the exact cleaned page string (Python indices).
"""

from __future__ import annotations

import hashlib
from typing import Any

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 180


def _validate_options(chunk_size: int, overlap: int) -> None:
    if type(chunk_size) is not int or chunk_size < 2:
        raise ValueError("chunk_size must be an integer >= 2")
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be an integer >= 0 and < chunk_size")


def _preferred_end(text: str, start: int, chunk_size: int) -> int:
    """Prefer paragraph/sentence/word breaks near the end of the window.

    No text is altered, including doses, punctuation, line breaks, and negation.
    If no suitable boundary exists, split at the strict character limit.
    """
    end = min(start + chunk_size, len(text))
    if end == len(text):
        return end
    low = start + int(chunk_size * 0.8)
    if low >= end:
        return end
    # Priority: paragraph ends, line ends, sentence ends, word gaps.
    segment = text[low:end]
    for marker in ("\n\n", "\n", ". ", "? ", "! ", " "):
        pos = segment.rfind(marker)
        if pos >= 0:
            candidate = low + pos + len(marker)
            if candidate > start:
                return candidate
    return end


def chunk_page(
    text: str,
    filename: str,
    page_number: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """Return stable, traceable overlapping chunks from one cleaned PDF page."""
    _validate_options(chunk_size, overlap)
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(filename, str) or not filename.endswith(".pdf") or filename != filename.strip():
        raise ValueError("filename must be a valid PDF filename")
    if type(page_number) is not int or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    if not text.strip():
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        end = _preferred_end(text, start, chunk_size)
        part = text[start:end]
        if part.strip():
            index = len(chunks) + 1
            identity = f"{filename}\x00{page_number}\x00{start}\x00{end}\x00{part}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            chunks.append({
                "chunk_id": chunk_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": index,
                "start_char": start,
                "end_char": end,
                "text": part,
            })
        if end == len(text):
            break
        # Offset-based overlap means zero gaps in coverage.
        next_start = max(start + 1, end - overlap)
        if next_start >= end:  # defensive, should be unreachable
            raise RuntimeError("chunker made no forward progress")
        start = next_start
    return chunks


def chunk_document(
    records: list[dict[str, Any]],
    *,
    expected_filename: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """Chunk one validated cleaned document, never merging across pages."""
    _validate_options(chunk_size, overlap)
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list of page records")
    filename = None
    chunks: list[dict[str, Any]] = []
    for expected_page, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Page {expected_page}: expected an object")
        name = record.get("filename")
        page = record.get("page_number")
        text = record.get("text")
        if not isinstance(name, str) or not name.endswith(".pdf"):
            raise ValueError(f"Page {expected_page}: invalid filename")
        if filename is None:
            filename = name
        if name != filename or (expected_filename is not None and name != expected_filename):
            raise ValueError(f"Page {expected_page}: filename mismatch")
        if type(page) is not int or page != expected_page:
            raise ValueError(f"Page {expected_page}: page numbers must be consecutive from 1")
        if not isinstance(text, str):
            raise ValueError(f"Page {expected_page}: text must be a string")
        chunks.extend(chunk_page(text, name, page, chunk_size=chunk_size, overlap=overlap))
    return chunks
