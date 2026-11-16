"""Re-chunk ONLY pages exceeding BGE's 512-token limit; retain all other chunks.

Commands (Git Bash, project root):
  python -m scripts.repair_token_overflow
  python -m scripts.repair_token_overflow --apply

No text truncation. Dry-run by default. Keeps one backup per modified chunk JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import tempfile
from collections import defaultdict

from src.chunker import chunk_page
from src.embedder import MODEL_NAME, MODEL_MAX_TOKENS
from scripts.embed import read_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Go below the course's normal 800-1200-character target only for pathological
# pages whose subword token counts do not fit the actual embedding model.
CANDIDATE_SIZES = (800, 700, 600, 500, 400, 300, 200, 120, 80)


def token_counts(texts: list[str], tokenizer, *, batch_size: int = 32) -> list[int]:
    """Use the *same* tokenizer family as the SentenceTransformer, no truncation."""
    counts: list[int] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        if not batch:
            continue
        response = tokenizer(
            batch, add_special_tokens=True, padding=False,
            truncation=False, return_attention_mask=False,
        )
        sequences = response["input_ids"]
        if len(sequences) != len(batch):
            raise ValueError("Tokenizer returned unexpected sequence count")
        counts.extend(len(ids) for ids in sequences)
    return counts


def _validate_original_against_cleaned(chunks: list[dict], pages: list[dict], name: str) -> dict[int, str]:
    """Require exact text offsets and full page coverage before changing any chunk."""
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"{name}: cleaned document must contain pages")
    expected_pdf = f"{Path(name).stem}.pdf"
    by_page: dict[int, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_page[chunk["page_number"]].append(chunk)
    texts = {}
    for number, page in enumerate(pages, 1):
        if (not isinstance(page, dict) or page.get("filename") != expected_pdf
                or page.get("page_number") != number or not isinstance(page.get("text"), str)):
            raise ValueError(f"{name}: invalid cleaned page {number}")
        text = page["text"]
        texts[number] = text
        records = sorted(by_page.pop(number, []), key=lambda c: c["start_char"])
        if not text.strip():
            if records:
                raise ValueError(f"{name}: empty page {number} unexpectedly has chunks")
            continue
        if not records or records[0]["start_char"] != 0:
            raise ValueError(f"{name}: chunk coverage begins incorrectly on page {number}")
        reached = 0
        for item in records:
            start, end = item["start_char"], item["end_char"]
            if start > reached or end > len(text) or item["text"] != text[start:end]:
                raise ValueError(f"{name}: source text or chunk coverage mismatch on page {number}")
            reached = max(reached, end)
        if reached != len(text):
            raise ValueError(f"{name}: page {number} is not fully covered")
    if by_page:
        raise ValueError(f"{name}: chunks refer to nonexistent page(s): {sorted(by_page)}")
    return texts


def repair_chunks(chunks: list[dict], pages: list[dict], name: str, tokenizer,
                  *, max_tokens: int = MODEL_MAX_TOKENS) -> tuple[list[dict], dict]:
    """Return corrected chunks and statistics; do not modify inputs or files."""
    if type(max_tokens) is not int or max_tokens < 4:
        raise ValueError("max_tokens must be an integer >= 4")
    source_text = _validate_original_against_cleaned(chunks, pages, name)
    counts = token_counts([c["text"] for c in chunks], tokenizer)
    oversize = {c["page_number"] for c, count in zip(chunks, counts) if count > max_tokens}
    if not oversize:
        return chunks, {"affected_pages": [], "before": len(chunks), "after": len(chunks), "max_tokens_before": max(counts, default=0), "max_tokens_after": max(counts, default=0)}

    original_by_page: dict[int, list[dict]] = defaultdict(list)
    for chunk in chunks:
        original_by_page[chunk["page_number"]].append(chunk)
    replacements: dict[int, list[dict]] = {}
    size_used: dict[int, int] = {}
    for page_number in sorted(oversize):
        text = source_text[page_number]
        for size in CANDIDATE_SIZES:
            # Use the original 180-character overlap where possible.
            overlap = min(180, max(1, size // 4))
            pieces = chunk_page(text, f"{Path(name).stem}.pdf", page_number,
                                chunk_size=size, overlap=overlap)
            if pieces and max(token_counts([c["text"] for c in pieces], tokenizer)) <= max_tokens:
                replacements[page_number] = pieces
                size_used[page_number] = size
                break
        if page_number not in replacements:
            raise ValueError(f"{name}: even 80-character chunks exceed {max_tokens} tokens on page {page_number}; manual review required")

    repaired = []
    for page_number in range(1, len(pages) + 1):
        repaired.extend(replacements.get(page_number, original_by_page.get(page_number, [])))

    if len({c["chunk_id"] for c in repaired}) != len(repaired):
        raise ValueError(f"{name}: duplicated IDs after repair")
    new_max = max(token_counts([c["text"] for c in repaired], tokenizer), default=0)
    if new_max > max_tokens:
        raise ValueError(f"{name}: repaired chunks still exceed {max_tokens} tokens")
    _validate_original_against_cleaned(repaired, pages, name)
    return repaired, {"affected_pages": sorted(oversize), "before": len(chunks), "after": len(repaired),
                      "max_tokens_before": max(counts), "max_tokens_after": new_max, "size_used": size_used}


def _atomic_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def scan_and_repair(root: Path, tokenizer, *, apply: bool = False, max_tokens: int = MODEL_MAX_TOKENS) -> dict:
    """Scan all chunks; on --apply change only documents containing oversized chunks."""
    chunks_dir, cleaned_dir = root / "data/chunks", root / "data/cleaned"
    files = sorted(chunks_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No chunk JSON files in {chunks_dir}")
    planned = []
    chunk_total = 0
    for i, path in enumerate(files, 1):
        original, _ = read_chunks(path)
        counts = token_counts([c["text"] for c in original], tokenizer)
        if max(counts) <= max_tokens:
            chunk_total += len(original)
            continue
        clean_path = cleaned_dir / path.name
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing cleaned source: {clean_path}")
        pages = json.loads(clean_path.read_text(encoding="utf-8"))
        repaired, result = repair_chunks(original, pages, path.name, tokenizer, max_tokens=max_tokens)
        planned.append((path, repaired, result))
        chunk_total += len(repaired)
        print(f"[{i}/{len(files)}] {path.name}: pages {result['affected_pages']} need repair | "
              f"{result['before']} -> {result['after']} chunks | max tokens {result['max_tokens_before']} -> {result['max_tokens_after']}", flush=True)

    # Two-phase behavior: finish validating all documents BEFORE editing any file.
    if apply:
        backup_dir = chunks_dir / "_token_repair_backups"
        for path, repaired, result in planned:
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / path.name
            if not backup.exists():
                # Original bytes saved once, without changing the original.
                temporary = backup.with_suffix(".json.tmp")
                temporary.write_bytes(path.read_bytes())
                os.replace(temporary, backup)
            _atomic_json(path, repaired)
            # Validate the resulting per-chunk SHA-256 identity and offsets.
            read_chunks(path)
            print(f"  repaired: {path.name} (original backed up)", flush=True)
    else:
        print("Dry run only: no file changed. Use --apply to repair the listed documents.", flush=True)

    summary = {"documents": len(files), "modified": len(planned), "predicted_chunks": chunk_total, "applied": apply}
    print(f"Token safety audit: {summary['documents']} documents; {summary['modified']} need repair; "
          f"predicted total {summary['predicted_chunks']} chunks; applied={apply}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Change only the affected documents; keep backups")
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        scan_and_repair(PROJECT_ROOT, tokenizer, apply=args.apply)
    except (OSError, RuntimeError, ValueError, KeyError, ImportError, TypeError) as exc:
        print(f"Token repair failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
