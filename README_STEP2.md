# Step 2 - PDF Document Loading (Drug Information Assistant)

## Scope

Keep the existing `scripts/ingest.py` unchanged. Add these **two** files to
corresponding locations within your project:

- `src/loader.py`
- `tests/test_loader.py`

The loader creates page-level records with exactly the keys your current
ingestion script expects: `filename`, `page_number`, `text`.

## Concepts

- **Corpus:** the 100 labeled PDFs you collected from DailyMed.
- **Document loading:** converting each PDF page into Python text data.
- **Page-level record:** a small dictionary that preserves the origin of text.
- **Provenance:** filename and page number let the later RAG system cite evidence.
- **No cleaning yet:** line breaks, unusual whitespace, and repeated headers stay
  as extracted. Those are handled at Step 3.
- **Scanned/image-only pages:** PyMuPDF may return empty text. That is explicitly
  counted/reported by your existing `scripts/ingest.py`. OCR is not done here.

## Git Bash commands (project root, .venv activated)

1. Verify syntax:

   ```bash
   python -m py_compile src/loader.py scripts/ingest.py
   ```

2. Run all tests (including prior 26 collection tests):

   ```bash
   python -m unittest discover -s tests -p 'test_*.py' -v
   ```

   Expect your **26 existing tests plus 8 new tests**, if your original test
   suite remains unchanged.

3. Ingest all 100 PDFs (reads PDF files; writes one JSON per PDF):

   ```bash
   python -m scripts.ingest
   ```

4. Audit the extracted pages with this Git Bash-compatible, read-only command:

   ```bash
   python -c "import json,pathlib; r=pathlib.Path('data/raw'); fs=list(r.glob('*.json')); ds=[json.loads(f.read_text(encoding='utf-8')) for f in fs]; print('JSON documents:',len(fs),'| Page records:',sum(map(len,ds)),'| Empty-text pages:',sum(not str(p['text']).strip() for doc in ds for p in doc))"
   ```

   If `data/raw/` does not contain unrelated files, the expected count is
   `100` JSON documents and `2292` page records for the current dataset. The
   total is based on your successful Step 1 collection audit. Compare it to
   your actual Step 2 result. Nonzero empty-text pages warrant inspection.

## Expected output structure

`data/raw/acetaminophen_325mg_health_pharma.json`:

```json
[
  {
    "filename": "acetaminophen_325mg_health_pharma.pdf",
    "page_number": 1,
    "text": "ACETAMINOPHEN..."
  }
]
```

Each PDF page becomes a separate object in the same JSON array, with page
numbers starting at 1. The text shown is an illustration, not an exact copy.

## Important

- Do **not** delete or edit files in `data/documents/`.
- Do **not** change `data/sources.csv` or `data/collection_state.json`.
- Do **not** chunk the documents yet; this step is **loading only**.
- The original PDFs remain the source of truth.
- If an ingestion error occurs, share its complete error and we will fix it
  before moving to cleaning.
