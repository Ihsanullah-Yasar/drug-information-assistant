"""Step 7 CLI: search existing ChromaDB with a natural-language question.

Example (Git Bash, project root):
    python -m scripts.retrieve --question "What are ibuprofen side effects?"
    python -m scripts.retrieve --question "What is ibuprofen used for?" --top-k 5

This script only reads Chroma. It does not generate medical advice or answers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from src.embedder import BGEEmbedder
from src.drug_router import resolve_drug_filename
from src.retriever import retrieve_chunks
from src.vectordb import connect_collection

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_query(
    question: str,
    *,
    db_dir: Path,
    top_k: int = 5,
    filename: str | None = None,
    expected_count: int | None = None,
    device: str = "cpu",
    json_out: Path | None = None,
    preview_chars: int = 300,
    embedder=None,
    collection=None,
) -> list[dict]:
    """Search, display, optionally save full evidence JSON (without answering)."""
    if type(preview_chars) is not int or not 80 <= preview_chars <= 3000:
        raise ValueError("preview_chars must be between 80 and 3000")
    # Validate inexpensive inputs before loading the neural model.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty")
    if type(top_k) is not int or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
        raise ValueError("expected_count must be positive")
    if collection is None:
        collection = connect_collection(db_dir, create=False)
    if collection.count() == 0:
        raise ValueError("ChromaDB collection is empty")
    if expected_count is not None and collection.count() != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, found {collection.count()}")
    if embedder is None:
        embedder = BGEEmbedder(device=device)
    matches = retrieve_chunks(
        question, collection=collection, embedder=embedder, top_k=top_k,
        filename=filename, expected_count=expected_count,
    )
    print(f"\nQuestion: {question.strip()}")
    print(f"Search: Top-{top_k} | collection size: {collection.count()} | returned: {len(matches)}")
    if filename is not None:
        print(f"PDF filter: {filename}")
    if not matches:
        print("No indexed chunks match the filter. No evidence was retrieved.")
    for match in matches:
        text_preview = " ".join(match["text"].split())
        if len(text_preview) > preview_chars:
            text_preview = text_preview[:preview_chars].rstrip() + " ..."
        print(f"\n[{match['rank']}] cosine distance: {match['cosine_distance']:.4f} "
              f"| similarity: {match['cosine_similarity']:.4f}")
        print(f"    PDF: {match['filename']} | page {match['page_number']} "
              f"| chunk {match['chunk_index']}")
        print(f"    Text: {text_preview}")
        print(f"    Source: {match['source_url']}")
    if json_out is not None:
        destination = Path(json_out)
        if not destination.suffix.lower() == ".json":
            raise ValueError("--json-out must end with .json")
        if destination.is_dir():
            raise ValueError("--json-out must point to a file, not a directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"question": question.strip(), "top_k": top_k,
                   "filename_filter": filename, "matches": matches}
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix=f".{destination.stem}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        print(f"\nFull retrieval evidence saved: {destination}")
    print("\nThese are search results, not a medical answer or a reliability score.")
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Natural-language query enclosed in quotes")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest chunks (default 5)")
    parser.add_argument("--filename", help="Explicit exact source PDF; overrides automatic drug routing")
    parser.add_argument("--all-drugs", action="store_true",
                        help="Opt out of automatic single-drug scope for corpus-wide analysis")
    parser.add_argument("--require-count", type=int, help="Reject incomplete collection count; 7194 in verified corpus")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--preview-chars", type=int, default=300)
    parser.add_argument("--json-out", type=Path, help="Optional location for complete retrieved text and citations")
    args = parser.parse_args()
    try:
        selected_filename = resolve_drug_filename(
            args.question,
            sources_csv=PROJECT_ROOT / "data/sources.csv",
            targets_txt=PROJECT_ROOT / "data/target_drugs.txt",
            filename=args.filename,
            all_drugs=args.all_drugs,
        )
        if args.filename is None and selected_filename is not None:
            print(f"Drug-aware routing: automatically selected {selected_filename}")
        elif args.all_drugs:
            print("Drug-aware routing: disabled by --all-drugs (global similarity search)")
        run_query(
            args.question, db_dir=PROJECT_ROOT / "data/chroma",
            top_k=args.top_k, filename=selected_filename, expected_count=args.require_count,
            device=args.device, json_out=args.json_out, preview_chars=args.preview_chars,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Retrieval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
