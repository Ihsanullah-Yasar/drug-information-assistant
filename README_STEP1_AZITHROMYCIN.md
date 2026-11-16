# Step 1 — Azithromycin ingredient verification correction

## Diagnosis
The `azithromycin` DailyMed search works, but candidates use the **structured ingredient** `AZITHROMYCIN DIHYDRATE` and **active moiety** `AZITHROMYCIN ANHYDROUS`. Previously the collector required an exact match to `azithromycin` and skipped these products. It is important not to assume all salts and hydrates are interchangeable by string similarity: this patch uses an explicitly declared SPL active-moiety relationship and removes only the qualifier `ANHYDROUS` from the active-moiety *name*.

## Files
- Replace **only** `scripts/collect_labels.py` with the copy in this ZIP.
- Add `tests/test_ingredient_matching.py`.
- `tests/test_responsive_search.py` is included as a convenience; it is unchanged.
- Do **not** replace `data/collection_state.json`, `data/sources.csv`, `data/documents/` or `data/target_drugs.txt`.

## Commands (Git Bash on Windows)
```bash
cp scripts/collect_labels.py scripts/collect_labels.before_moiety_fix.py
python -m unittest discover -s tests -p 'test_*.py' -v
python -m scripts.collect_labels --limit 19 --max-candidates 12 --search-seconds 90
```

If the 19-drug check completes, you can resume the full batch:
```bash
python -m scripts.collect_labels
```

## What to look for
- `azithromycin | saved | ... pages`
- `verified targets 19/19` in the first audit
- Later: `verified targets 100/100` in the full audit

### Important boundaries
This patch is supported by DailyMed examples showing **AZITHROMYCIN DIHYDRATE** with active moiety **AZITHROMYCIN ANHYDROUS**. It does not guarantee every searched XML has identical structure. If it still fails, share the new candidate skips. Your first 18 labels remain safe and reusable. There are six new offline tests. Existing 10 tests are expected to continue to pass.
