# Step 7.1 — Drug-aware retrieval repair

## Why this is necessary

Your actual global query `What are the adverse reactions of ibuprofen?` retrieved top-5 passages from clindamycin, pantoprazole, diclofenac and meloxicam instead of ibuprofen. A high cosine similarity does NOT establish drug identity or medical correctness.

This patch uses the existing source-PDF filtering already validated in Step 7. It does NOT alter text chunks, BGE passage vectors, the ChromaDB index, or prior evidence JSON exports. It only changes the command-line routing behavior when exactly one known drug is named in the query.

## Files

- `src/drug_router.py` — new deterministic router (reads existing `data/target_drugs.txt` and `data/sources.csv`).
- `scripts/retrieve.py` — REPLACE existing Step 7 CLI with this version.
- `tests/test_drug_router.py` — 12 new offline tests.

## Algorithm

1. Read canonical drug names from `data/target_drugs.txt` and source PDF filenames from `data/sources.csv`.
2. Find exact, whole-token matches in the query. No fuzzy drug substitutions, no guessing synonyms, no API calls.
3. If exactly ONE known drug is mentioned and exactly ONE source PDF filename starts with the drug-name tokens, apply ChromaDB `where={"filename": ...}`.
4. If MULTIPLE known drugs are mentioned, reject the unscoped question instead of mixing drug facts. Explicit `--filename` selects one source label; `--all-drugs` intentionally disables routing for corpus-wide analysis.
5. If NO known drug is named, leave the search global (useful for generic IR research queries). This is *not* proof the response is safe to generate without further grounding checks.
6. If a matched drug has ZERO or MULTIPLE mapped PDFs, fail closed with an actionable error instead of guessing the product/formulation.

### Commands (Git Bash, active .venv)

Run from `D:/github/drug-information-assistant` after extracting the ZIP preserving folder structure:

```bash
python -m py_compile src/drug_router.py scripts/retrieve.py
python -m unittest discover -s tests -p "test_*.py" -v
python -m scripts.retrieve --question "What are the adverse reactions of ibuprofen?" --top-k 5 --require-count 7194 --json-out data/retrieval_runs/ibuprofen_drug_aware.json
```

Expected: `Drug-aware routing: automatically selected ibuprofen_200mg_haleon_us_holdings_llc.pdf` followed by up to 5 ibuprofen-only results. The actual ranking/distances will be computed on your PC.

Compare the intentionally unrestricted search with:

```bash
python -m scripts.retrieve --question "What are the adverse reactions of ibuprofen?" --all-drugs --top-k 5 --require-count 7194
```

This reproduces the original global ranking. It must NOT be used to infer ibuprofen-only facts from other drugs.

For a direct explicit PDF filter:

```bash
python -m scripts.retrieve --question "What are the adverse reactions of ibuprofen?" --filename ibuprofen_200mg_haleon_us_holdings_llc.pdf --top-k 5 --require-count 7194
```

## Important limitations

- Drug names must match `target_drugs.txt` exactly as whole tokens; brand synonyms and spelling mistakes are not automatically resolved.
- One source label for a drug is not a universal clinical drug monograph or a substitute for licensed medical guidance. Product/formulation differences matter.
- A filtered result still needs passage-level verification; cosine similarity is not an answerability score.
- Questions containing multiple distinct drug names should eventually be split into separate, source-specific retrievals for comparisons or interactions. Do not combine them in an unguarded answer prompt.
- The router only affects the command-line entry point. When building the subsequent Step 8 pipeline programmatically, explicitly call `resolve_drug_filename()` before calling `retrieve_chunks()`.
