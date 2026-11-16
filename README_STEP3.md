# Step 3: Text cleaning — Drug Information Assistant

## What this does

Converts `data/raw/*.json` to `data/cleaned/*.json` while retaining the same page-record JSON structure:

```json
[{"filename": "medicine.pdf", "page_number": 1, "text": "...cleaned text..."}]
```

The script never changes `data/raw`, `data/documents`, `data/sources.csv`, or the collection state.

## Learning goals

- **Preprocessing:** Prepare text for retrieval by eliminating extraction artifacts.
- **Unicode normalization (NFC):** Combine canonically equivalent character sequences without compatibility-folding scientific symbols.
- **Whitespace normalization:** Remove repeated horizontal whitespace while keeping line boundaries.
- **Metadata preservation:** Keep the PDF name and 1-based page number so future answers can cite sources.
- **Loss minimization:** Preserve drug names, strengths, warnings, directions, and other medically relevant lines.
- **Empty-page handling:** Retain empty pages in JSON to make data quality auditable. Do not create embeddings for empty text.

The code removes only *explicit* `Page N` / `Page N of M` labels at page boundaries. It does not remove bare numbers (which could be doses), or unreviewed repeated headers/footers. It does not perform OCR; image-only pages remain empty.

## Install the files

Place these files in your existing repository (replace the existing empty `src/cleaner.py`):

- `src/cleaner.py`
- `scripts/clean.py`
- `tests/test_cleaner.py`

Do not delete or replace `scripts/ingest.py`, `src/loader.py`, or anything under `data/`.

## Commands (Git Bash, with your venv active)

```bash
python -m py_compile src/cleaner.py scripts/clean.py
python -m unittest discover -s tests -p 'test_*.py' -v
python -m scripts.clean
python -c "from pathlib import Path; import json; fs=list(Path('data/cleaned').glob('*.json')); pages=sum(len(json.loads(f.read_text(encoding='utf-8'))) for f in fs); print('Cleaned documents:',len(fs),'| Page records:',pages)"
```

Expected counts for your present corpus: **100 cleaned documents** and **2,292 page records**. Your ingestion log reported **8 empty-text pages**; the cleaner should retain them (zero chunks should eventually be created from them). The expected nonempty page count is **2,284**.

## Verify an example

```bash
python -c "import json; from pathlib import Path; p=Path('data/cleaned/acetaminophen_325mg_health_pharma.json'); pages=json.loads(p.read_text(encoding='utf-8')); print(pages[0]['filename'], pages[0]['page_number']); print(pages[0]['text'][:800])"
```

## What is not done yet

This step does *not* do chunking, tokenization, stemming, stop-word removal, retrieval, embedding generation, Ollama, or ChromaDB. The following assignment step will manually split cleaned text into chunks while preserving source metadata.
