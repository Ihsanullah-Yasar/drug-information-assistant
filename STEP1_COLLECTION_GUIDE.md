# Drug Information Assistant — Step 1: Document Collection

This patch **only** replaces `scripts/collect_labels.py` and adds offline tests. It does not replace or modify `data/sources.csv`, your target list, or any of the existing three PDFs.

## 1. Before you start

From the root of your existing VS Code project (`drug-information-assistant`), activate the existing Python environment or use its interpreter directly.

```powershell
.\.venv\Scripts\python.exe -m pip install pymupdf
```

Back up your current `scripts/collect_labels.py` and `data/sources.csv`. Copy the supplied new collector into `scripts/collect_labels.py`, and the test file into `tests/test_collect_labels.py`.

## 2. Offline checks

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\collect_labels.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The offline suite mocks DailyMed responses and simulates local PDF files. It does **not** prove that DailyMed is currently online or that every target has an available label.

## 3. Reconcile the first three drugs

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_labels --limit 3
```

This step attempts to verify the reviewed acetaminophen, ibuprofen, and naproxen labels against DailyMed; preserves the three PDF files if valid; and adds the missing naproxen row to `data/sources.csv`. It also writes resumable receipts to `data/collection_state.json`.

When run successfully, expect `3/3` verified targets, three distinct Set IDs and three matching CSV rows. If DailyMed fails to respond, the command reports the failure and can be rerun.

## 4. Full batch after Step 3 passes

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_labels
```

This processes all **100** target names. `--search-pages 5` is the default maximum search depth for each name; if some are unresolved, inspect the specific errors and try increasing it, e.g. `--search-pages 10`. Increasing the value increases DailyMed requests. The default command always resumes previously verified work.

## 5. Local audit (no network)

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_labels --audit-only
```

A complete audit must show 100 verified targets, 100 distinct Set IDs, 100 readable and text-extractable PDFs, 100 matching CSV rows and at least 100 PDF pages.

## Notes

- Existing PDF filenames and the existing two source rows are retained.
- A DailyMed Set ID identifies a label, not an active ingredient. An SPL XML check is required to verify the target match.
- Candidate selection prefers oral tablets but accepts other single-ingredient human dosage forms if the ingredient or active moiety matches.
- Files are committed with temporary-file/atomic writes and per-document SHA-256 receipts, supporting safe reruns.
- DailyMed availability and labels change. Some targets may need a higher search depth or manual investigation, and no program can guarantee that all 100 have a suitable label.
- We will implement PDF ingestion / `src/loader.py` in **Step 2**. Do not run `scripts/ingest.py` while `src/loader.py` is empty.
