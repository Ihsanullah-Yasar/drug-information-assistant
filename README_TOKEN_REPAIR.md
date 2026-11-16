# Step 5 — BGE token overflow repair (keep existing embeddings)

## Why the batch stopped

The user already generated valid per-document `.npz` archives for the first 97 source documents. Embedding the valacyclovir PDF failed because the 1000-character sliding-window chunker produced a chunk whose actual BGE tokenizer length was 626, exceeding 512 model tokens. Characters and model tokens are different units. We must not truncate drug label text.

## Files

- `scripts/repair_token_overflow.py` — add to your project's `scripts/` folder.
- `tests/test_token_repair.py` — add to the project's `tests/` folder.
- No changes to `src/chunker.py`, `src/embedder.py`, `scripts/embed.py`, or existing PDFs.

## Commands (Git Bash from project root)

1. Verify the script and tests:

   ```bash
   python -m py_compile scripts/repair_token_overflow.py
   python -m unittest discover -s tests -p 'test_*.py' -v
   ```

   With the earlier 67 tests installed but without Step 6's tests, expect `Ran 72 tests`.

2. **Dry run first** (uses the locally cached BGE tokenizer; does NOT load the neural weights or change files):

   ```bash
   python -m scripts.repair_token_overflow
   ```

   This scans all 100 existing chunk JSON files and shows exactly which files and pages exceed 512 tokens. The predicted total chunk count might differ from 7,193 because only problematic pages will be split into smaller chunks.

3. Apply the proposed local fixes:

   ```bash
   python -m scripts.repair_token_overflow --apply
   ```

   The script checks *all* files first. It rewrites only chunk JSON files containing overlong text, preserves all other chunks' IDs and offsets, and writes original backups under `data/chunks/_token_repair_backups/`. It re-chunks individual problematic PDF pages using progressively smaller sizes until every resulting chunk has <=512 tokens.

4. Continue from the existing saved vectors:

   ```bash
   python -m scripts.embed --batch-size 16
   ```

   The embedder compares saved `.npz` files with the corresponding chunk JSON hash and IDs. If unchanged, it prints `reused`; only missing or changed archives are generated. Its output should end with a successful summary. **Do not delete `data/embeddings/` or re-chunk the entire corpus.**

5. Audit the full embedding collection:

   ```bash
   python -m scripts.embed --audit-only
   ```

   Check: 100 documents, same count of embeddings and unique IDs, dimension 384. The exact number of embeddings may differ from 7,193, which was the total before repairing overlong chunks.

## Important details

- The script uses `AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)` and the same `truncation=False`, `add_special_tokens=True` settings as your `BGEEmbedder`.
- A valid cleaned page is authoritative, and every old and new chunk is checked against exact offsets in that text.
- Already safe pages are copied unchanged, so their IDs remain stable.
- Backups are created once and never intentionally overwritten on reruns.
- This script performs token-aware repair, **not medical advice or medical-data rewriting**.
- If another error appears, share its exact output rather than changing the entire dataset.
