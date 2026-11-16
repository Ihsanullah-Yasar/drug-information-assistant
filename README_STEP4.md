# Step 4 — Manual Text Chunking

## Purpose
Split all 100 cleaned drug-label documents into small, overlapping, page-cited text pieces before embedding. No LangChain, LlamaIndex, or Haystack.

## Install
From this ZIP, copy these files into the existing repository (do not replace old code or data):

- `src/chunker.py`
- `scripts/chunk.py`
- `tests/test_chunker.py`

Your existing `src/__init__.py` and `scripts/__init__.py` remain unchanged.

## Git Bash commands (from project root, with .venv activated)

```bash
python -m py_compile src/chunker.py scripts/chunk.py
python -m unittest discover -s tests -p 'test_*.py' -v
python -m scripts.chunk
```

Review the final audit. Expect **100 documents**, **2,292 page records**, **8 empty-text pages**, and a positive number of chunks. The number of chunks is determined by actual text length; do not assume it equals 2,292.

Check count, unique IDs, and sample record:

```bash
python -c "import json; from pathlib import Path; fs=list(Path('data/chunks').glob('*.json')); ds=[c for f in fs for c in json.loads(f.read_text(encoding='utf-8'))]; print('Files:',len(fs),'Chunks:',len(ds),'Unique IDs:',len({c['chunk_id'] for c in ds})); print('Sample:',json.dumps(ds[0],indent=2,ensure_ascii=False)[:1200])"
```

## What the code does

1. Read `data/cleaned/*.json`, one array of page records per PDF.
2. Validate source filename, page sequence, and text type.
3. Split **within each page**; a chunk never contains text from two different pages or PDFs.
4. Use **1000 characters max** and **180 characters overlap** by default. Prefer a paragraph, newline, sentence or word boundary in the last fifth of each window; otherwise split at the character limit.
5. Preserve the original cleaned source substring exactly, including whitespace within the span. `start_char` (inclusive) and `end_char` (exclusive) are zero-based Python character offsets.
6. Produce stable SHA-256 `chunk_id` values from text + source identity + offsets.
7. Write to `data/chunks/<document>.json` atomically. Leave `data/cleaned/` and original PDFs untouched.
8. Skip empty pages (zero chunks), but include them in the audit.

Each chunk record:

```json
{
  "chunk_id": "sha256 hex digest",
  "filename": "acetaminophen_325mg_health_pharma.pdf",
  "page_number": 1,
  "chunk_index": 1,
  "start_char": 0,
  "end_char": 1000,
  "text": "Actual extracted and cleaned label text..."
}
```

## Tradeoffs and cautions

- Character length is **not token length**. Embedding-model token limits must be checked in Step 5; long chunks may be truncated by the model unless handled explicitly.
- Overlap repeats content and costs index/storage space, but reduces lost context at chunk boundaries.
- Offsets let us check that `text == cleaned_page_text[start_char:end_char]`.
- Page-boundary chunking improves citation accuracy but may separate sentences split across PDF pages. We'll assess its retrieval impact later.
- The extractor identified eight pages without extractable text; we do not silently invent OCR content.
- Re-running the command with a different chunk size overwrites previous chunk JSON files. Rebuild embeddings and indexes when chunk settings change.

## Completion criteria

- Syntax check succeeds.
- Prior **46** tests plus the **9** new chunker tests pass (**55** total, if prior test files are unchanged).
- Audit shows 100 documents, 2,292 pages, 8 empty pages.
- Number of distinct IDs equals number of chunks.
- Output JSON files exist in `data/chunks/`.

Only then advance to embedding generation.
