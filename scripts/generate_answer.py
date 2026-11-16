"""Step 9 CLI: generate one grounded response from an existing Step 8 prompt JSON.

Uses ONLY the local Ollama API and NEVER reads or modifies ChromaDB vectors.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from src.ollama_generator import (
    DEFAULT_CONTEXT, DEFAULT_MODEL, DEFAULT_PREDICT, generate_grounded_answer,
)


def _write_atomic(path: Path, content: str, *, forbidden: Path) -> None:
    path = Path(path)
    if path.resolve() == forbidden.resolve():
        raise ValueError("Refusing to overwrite source prompt JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run_from_file(prompt_json: Path, *, model: str = DEFAULT_MODEL,
                  num_ctx: int = DEFAULT_CONTEXT, max_output_tokens: int = DEFAULT_PREDICT,
                  timeout: int = 600, allow_multiple_documents: bool = False,
                  json_out: Path | None = None, text_out: Path | None = None,
                  sender=None) -> dict:
    source = Path(prompt_json)
    if not source.is_file() or source.suffix.lower() != ".json":
        raise FileNotFoundError(f"Prompt JSON not found: {source}")
    for output, suffix in ((json_out, ".json"), (text_out, ".txt")):
        if output is not None and Path(output).suffix.lower() != suffix:
            raise ValueError(f"Output must end with {suffix}")
    paths = [Path(x).resolve() for x in (json_out, text_out) if x is not None]
    if len(paths) != len(set(paths)) or source.resolve() in paths:
        raise ValueError("Outputs must be distinct and must not overwrite the prompt")

    package = json.loads(source.read_text(encoding="utf-8"))
    result = generate_grounded_answer(
        package, model=model, num_ctx=num_ctx, max_output_tokens=max_output_tokens,
        timeout=timeout, allow_multiple_documents=allow_multiple_documents,
        sender=sender,
    )
    if json_out is not None:
        _write_atomic(Path(json_out), json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      forbidden=source)
    if text_out is not None:
        lines = [result["answer"], "", "Sources:"]
        for item in result["cited_sources"]:
            lines.append(f"[{item['source_id']}] {item['filename']}, page {item['page_number']}, {item['source_url']}")
        lines += ["", result["note"]]
        _write_atomic(Path(text_out), "\n".join(lines) + "\n", forbidden=source)

    print(f"Answer status: {result['status']}")
    print(f"Reason: {result['reason']}")
    print(f"Question: {result['question']}")
    print(f"Model: {result['model']}")
    print("\nAnswer:\n" + result["answer"])
    for item in result["cited_sources"]:
        print(f"  [{item['source_id']}] {item['filename']} | page {item['page_number']}")
        print(f"      {item['source_url']}")
    usage = result["usage"]
    if usage["prompt_eval_count"] is not None:
        print(f"Prompt tokens (reported by Ollama): {usage['prompt_eval_count']}")
    if json_out is not None:
        print(f"JSON output: {json_out}")
    if text_out is not None:
        print(f"Readable output: {text_out}")
    print("Citation IDs were validated; answer claims still require manual/source checking.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_PREDICT)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--allow-multiple-documents", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--text-out", type=Path)
    args = parser.parse_args()
    try:
        run_from_file(
            args.prompt_json, model=args.model, num_ctx=args.num_ctx,
            max_output_tokens=args.max_output_tokens, timeout=args.timeout,
            allow_multiple_documents=args.allow_multiple_documents,
            json_out=args.json_out, text_out=args.text_out,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Step 9 generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
