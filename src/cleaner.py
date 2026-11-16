"""Conservative, page-preserving text cleaning for the drug-label corpus.

This module intentionally does not summarize, deduplicate medical instructions,
remove section headings, or merge text from different PDF pages.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Keep line feeds and tabs, but remove NUL and other non-printing ASCII controls.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


def _is_boundary_page_label(line: str, page_number: int, total_pages: int) -> bool:
    """Recognize an explicit 'Page N' or 'Page N of M' margin label only."""
    match = re.fullmatch(r"page\s+(\d+)(?:\s*(?:of|/)\s*(\d+))?", line, re.IGNORECASE)
    if match is None or int(match.group(1)) != page_number:
        return False
    stated_total = match.group(2)
    return stated_total is None or int(stated_total) == total_pages


def clean_text(text: str, *, page_number: int | None = None, total_pages: int | None = None) -> str:
    """Clean extraction artifacts while preserving medical wording and page identity.

    We use NFC, not NFKC: compatibility folding can change meaningful scientific
    notation, such as superscripts. We never drop numeric-only lines, because
    a number alone can be part of a dosage or a table.
    """
    if not isinstance(text, str):
        raise TypeError("Page text must be a string")
    if (page_number is None) != (total_pages is None):
        raise ValueError("page_number and total_pages must be supplied together")
    if page_number is not None and (type(page_number) is not int or type(total_pages) is not int
                                    or not 1 <= page_number <= total_pages):
        raise ValueError("Invalid page_number or total_pages")

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = _CONTROL_CHARS.sub("", text)

    lines = []
    for original_line in text.split("\n"):
        line = _HORIZONTAL_WHITESPACE.sub(" ", original_line).strip()
        if line:
            lines.append(line)

    # Explicit page labels are safe to discard only at a page boundary.
    # A bare number (e.g. '5') is NEVER discarded; it might be a dosage.
    if page_number is not None and lines:
        if _is_boundary_page_label(lines[0], page_number, total_pages):
            lines.pop(0)
        if lines and _is_boundary_page_label(lines[-1], page_number, total_pages):
            lines.pop()

    return "\n".join(lines)


def clean_document(records: list[dict[str, Any]], *, expected_filename: str | None = None) -> list[dict[str, Any]]:
    """Clean page-record JSON, verifying all pages are present and in order."""
    if not isinstance(records, list) or not records:
        raise ValueError("Document must have a nonempty list of page records")

    total_pages = len(records)
    cleaned = []
    first_filename = None
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Page {index}: expected a JSON object")
        filename = record.get("filename")
        page_number = record.get("page_number")
        text = record.get("text")
        if not isinstance(filename, str) or not filename.endswith(".pdf"):
            raise ValueError(f"Page {index}: invalid PDF filename")
        if first_filename is None:
            first_filename = filename
        if filename != first_filename or (expected_filename is not None and filename != expected_filename):
            raise ValueError(f"Page {index}: inconsistent source filename")
        if type(page_number) is not int or page_number != index:
            raise ValueError(f"Page {index}: page_number must be {index}")
        if not isinstance(text, str):
            raise ValueError(f"Page {index}: text must be a string")

        cleaned.append({
            "filename": filename,
            "page_number": index,
            "text": clean_text(text, page_number=index, total_pages=total_pages),
        })
    return cleaned
