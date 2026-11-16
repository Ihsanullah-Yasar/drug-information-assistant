"""Step 8 CLI: build a manual, source-grounded chat prompt from Step 7 JSON.

Run from the project root, e.g.:
python -m scripts.build_prompt --retrieval-json data/retrieval_runs/ibuprofen_drug_aware.json \
  --json-out data/prompts/ibuprofen_prompt.json --text-out data/prompts/ibuprofen_prompt.txt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from src.prompt_builder import build_prompt_package


def _atomic_write(path: Path, contents: str, *, input_path: Path) -> None:
    path = Path(path)
    if path.resolve() == input_path.resolve():
        raise ValueError("Output cannot overwrite the original retrieval evidence JSON")
    if path.exists() and path.is_dir():
        raise ValueError("Output must be a file, not a directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temp = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def render_prompt_text(package: dict) -> str:
    if package["status"] != "ready":
        return f"STATUS: {package['status']}\n{package['note']}\n{package['fallback_answer']}\n"
    messages = package["messages"]
    return ("=== SYSTEM INSTRUCTIONS ===\n"
            + messages[0]["content"]
            + "\n=== USER QUESTION AND FULL EVIDENCE ===\n"
            + messages[1]["content"] + "\n")


def build_from_file(
    retrieval_json: Path,
    *,
    json_out: Path | None = None,
    text_out: Path | None = None,
    max_sources: int = 5,
    max_context_chars: int = 10000,
    allow_multiple_documents: bool = False,
) -> dict:
    retrieval_json = Path(retrieval_json)
    if not retrieval_json.is_file():
        raise FileNotFoundError(f"Retrieval JSON not found: {retrieval_json}")
    if json_out is not None and Path(json_out).suffix.lower() != ".json":
        raise ValueError("--json-out must end in .json")
    if text_out is not None and Path(text_out).suffix.lower() != ".txt":
        raise ValueError("--text-out must end in .txt")
    if json_out is not None and text_out is not None and Path(json_out).resolve() == Path(text_out).resolve():
        raise ValueError("--json-out and --text-out must be distinct files")
    for path in (json_out, text_out):
        if path is not None and Path(path).resolve() == retrieval_json.resolve():
            raise ValueError("Do not overwrite retrieval input JSON")
    data = json.loads(retrieval_json.read_text(encoding="utf-8"))
    package = build_prompt_package(
        data, max_sources=max_sources, max_context_chars=max_context_chars,
        allow_multiple_documents=allow_multiple_documents,
    )
    if json_out is not None:
        _atomic_write(Path(json_out), json.dumps(package, indent=2, ensure_ascii=False) + "\n",
                      input_path=retrieval_json)
    if text_out is not None:
        _atomic_write(Path(text_out), render_prompt_text(package), input_path=retrieval_json)
    print(f"Prompt status: {package['status']}")
    print(f"Question: {package['question']}")
    print(f"Sources included: {package['source_count']} / {package['retrieved_match_count']}")
    if package["status"] == "ready":
        for src in package["sources"]:
            print(f"  [{src['source_id']}] {src['filename']} | page {src['page_number']} | chunk {src['chunk_index']}")
    else:
        print(f"Reason: {package['note']}")
        print(f"Fallback: {package['fallback_answer']}")
    if json_out is not None:
        print(f"Saved JSON prompt: {json_out}")
    if text_out is not None:
        print(f"Saved readable prompt: {text_out}")
    print("This step builds a prompt only. It does NOT call Ollama or generate an answer.")
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--text-out", type=Path)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=10000)
    parser.add_argument("--allow-multiple-documents", action="store_true")
    args = parser.parse_args()
    try:
        build_from_file(args.retrieval_json, json_out=args.json_out, text_out=args.text_out,
                        max_sources=args.max_sources, max_context_chars=args.max_context_chars,
                        allow_multiple_documents=args.allow_multiple_documents)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Prompt construction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
