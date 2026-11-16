# Step 1 — Finish the three unresolved drug labels

Your final log reports 97/100 verified targets, 97 unique Set IDs, 97 PDFs,
97 CSV records, and 2,211 pages. The remaining drugs are valacyclovir,
oseltamivir, and enalapril.

## Root cause
The SPL fields have these exact pairs:

| Target | Labeled active ingredient | SPL active moiety |
|---|---|---|
| valacyclovir | VALACYCLOVIR HYDROCHLORIDE | ACYCLOVIR |
| oseltamivir | OSELTAMIVIR PHOSPHATE | OSELTAMIVIR ACID |
| enalapril | ENALAPRIL MALEATE | ENALAPRILAT ANHYDROUS |

These pairs are explicitly allowlisted. The label must still be verified as
human-use, single-active-ingredient, with valid SPL Set ID, strength, dosage
form, route, source date, and a readable PDF. This is an identity-resolution
mechanism for label collection, NOT advice about drug substitution.

## Files to update

- Back up current `scripts/collect_labels.py`.
- Replace **only** `scripts/collect_labels.py` from this ZIP.
- Add `tests/test_final_three_salts.py`.
- Keep your existing tests and all original `data/` files.
- Do **not** replace or delete `collection_state.json`, `sources.csv`, or PDFs.

## Git Bash commands on Windows

```bash
cp scripts/collect_labels.py scripts/collect_labels.before_final_three.py
cp data/collection_state.json data/collection_state.backup.json
cp data/sources.csv data/sources.backup.csv
python -m py_compile scripts/collect_labels.py
python -m unittest discover -s tests -p 'test_*.py' -v
python -m scripts.collect_labels --max-candidates 12 --search-seconds 90
```

The final batch reuses the 97 completed PDFs and attempts the last three.
If an API request fails, rerun the same collection command. Do not alter
successful state or source rows to force the audit to pass.

## Expected successful result

```
Audit: verified targets 100/100; distinct Set IDs 100;
readable PDFs 100; matching CSV rows 100; total PDF pages ...
Collection audit passed: every checked target has a distinct verified label, readable PDF and source row.
```

Live DailyMed requests still need to be run on your own computer. This patch
has been verified using offline tests, not the real 100-drug download.
