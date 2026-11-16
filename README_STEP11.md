# Step 11 — Information Retrieval Evaluation (Hit Rate@5 and MRR@5)

Your first ten steps already built a local Drug Information Assistant from 100 DailyMed labels, using 7,194 indexed chunks. **This step adds retrieval evaluation.** It does NOT replace the Gradio UI, invoke Ollama, rebuild the index, or regenerate stored document embeddings.

The course assignment's formal deliverables are the implementation, dataset/link, README, and 5–10-page report; **Hit Rate@5 and MRR are additional, useful evaluation evidence**, not a numeric requirement explicitly stated in the assignment handout.

## Included files

- `src/ir_evaluator.py` — validation, retrieval across two modes, gold relevance checks, Hit Rate/MRR formulas, CSV/JSON export.
- `scripts/evaluate_ir.py` — create candidate pool and calculate metrics after gold is supplied.
- `scripts/finalize_ir_gold.py` — turn manually reviewed 0/1 candidate judgments into a gold file.
- `tests/test_ir_evaluator.py` and `tests/test_ir_gold_review.py` — 26 deterministic offline tests.
- `data/evaluation/questions.csv` — **nine starter questions**: three ibuprofen, three acetaminophen, three fluoxetine. Extend across more drugs before making general performance claims.
- `data/evaluation/gold.csv` — EMPTY gold annotation template. This is intentional: no fabricated ground truth.

All scripts use only the already installed project dependencies. There are no new pip packages.

## The learning goal

Retrieval quality cannot be measured by asking the LLM whether its answer seems correct. It requires an **independently judged relevant passage set** for each question. A chunk is relevant if its actual text supplies evidence that answers a question. Simply being in the right drug PDF is **not enough**.

For each question `q`, let `G(q)` be manually verified relevant **chunk IDs**, and `R(q)` be the Top-5 retrieved chunk IDs ranked by similarity.

- **Hit@5(q)** = 1 when any `R(q)` chunk matches a `G(q)` chunk; otherwise 0.
- **Hit Rate@5** = average of Hit@5(q) across all questions.
- **RR@5(q)** = 1 / rank of the *first* relevant chunk in the first five results; otherwise 0.
- **MRR@5** = average of RR@5(q).

Example: if relevant evidence first appears at rank 2, Hit@5 = 1 and RR@5 = 0.5. These are not medical accuracy scores or confidence probabilities.

## Step A — Extract this ZIP into your existing project

Merge `src/`, `scripts/`, `tests/`, and `data/evaluation/` under:

`D:\\github\\drug-information-assistant`

Do not copy the internal `fulltest` folder (it is not included in the ZIP).

From your active Windows Git Bash venv:

```bash
python -m unittest discover -s tests -p 'test_ir_*.py' -v
```

Expected: **26 tests, OK**.

Optional full regression check:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Step B — Build a candidate pool for manual relevance judgments

```bash
python -m scripts.evaluate_ir \
  --prepare \
  --top-k 10 \
  --require-count 7194 \
  --pool-out data/evaluation/candidate_pool.csv
```

This loads **one** BGE query-embedding model and the existing read-only Chroma collection. For each question, it queries:

1. **global** — whole 7,194-vector collection without a filename filter;
2. **drug_aware** — automatically resolves the medicine from the question and filters the source PDF, as the Gradio UI does.

Crucially, the `expected_filename` field is an **audit of the routing result**, not a source chosen for the global search or a gold relevance label. If routing differs, evaluation fails rather than hiding the discrepancy.

The pool contains up to **180 rows** (9 questions × 2 modes × Top-10). Because many chunks appear in both modes, some are duplicates.

Open `data/evaluation/candidate_pool.csv` in a spreadsheet editor or VS Code and inspect each full passage, preferably with the original DailyMed PDF open. In the column `review_relevant_1_or_0`, enter:

- `1` if this specific passage actually contains evidence relevant to this specific question;
- `0` if it does not.

**All rows must be judged.** Repeated (question ID, chunk ID) pairs across modes must have identical relevance judgments. Never mark a passage relevant merely because its filename matches the question.

For stronger research rigor, inspect relevant portions of the original PDF independently and identify other relevant chunks that the initial candidate pool missed. Pool-based labeling alone can be biased toward the systems under evaluation.

## Step C — Build the gold relevance file

After manually filling **every** annotation cell in `candidate_pool.csv`:

```bash
python -m scripts.finalize_ir_gold \
  --reviewed-pool data/evaluation/candidate_pool.csv \
  --gold-out data/evaluation/gold.csv
```

This exports only manually marked positive chunks, deduplicated. It refuses blank/invalid annotations, contradictory duplicate judgments, and queries without any verified positive chunk. If a question has no positive candidate, **do not invent one**: inspect the source more closely, expand the candidate pool, or revise the question.

`gold.csv` has the format:

```csv
question_id,chunk_id
```

Add any independently verified positives not found in the pool manually, using their exact 64-character `chunk_id`. The evaluator verifies that all gold IDs exist in ChromaDB and belong to the PDF designated for that question.

## Step D — Calculate honest metrics

```bash
python -m scripts.evaluate_ir \
  --top-k 5 \
  --gold data/evaluation/gold.csv \
  --require-count 7194 \
  --json-out data/evaluation/metrics.json \
  --csv-out data/evaluation/metrics_by_question.csv
```

The script prints Hit Rate@5 and MRR@5 for both **global** and **drug_aware** retrieval, based on **the same question set and the same manually verified gold IDs**. The latter mode uses a source restriction, so report that restriction when comparing their metrics; it is an advantage from drug routing, not solely an improvement in the embedding model.

If `gold.csv` has not been completed, the script will refuse to report any score. **There are no precomputed performance numbers in this package.**

The evaluator saves per-question binary hits, first relevant rank, reciprocal rank, and source-audit fields. All scores are reproducible from your own labels and unchanged Chroma index.

## What the screenshot teaches us beyond retrieval metrics

Your fluoxetine Gradio run proves functional integration. It does not prove that "anxiety disorders" is a precisely supported indication or that the listed effects are the "most common" for that formulation; the source excerpts shown in the screenshot are not enough to verify these claims.

For final assessment, **manually fact-check** at least 10 answers, recording:

- Is the selected drug/label correct?
- Does every factual claim follow from the cited chunk and original label?
- Are the stated conditions, dosage form, and indication accurate?
- Does the system abstain rather than invent content for questions with no evidence?

That is **answer faithfulness evaluation**, separate from Hit Rate@5 and MRR@5. The two metrics in this package are retrieval-only.

## Important methodology notes

- Answerable-only question set: nine starter questions each require at least one gold-positive chunk. This evaluation therefore does **not** measure abstention or performance on unanswerable questions. Test those separately.
- Your first nine questions reference only three medicines, not all 100. For a serious final report, expand to ~20–30 diverse questions across 10 or more medicines, sample them without choosing only easy examples, and mention the sample's limitations.
- Different source label versions can have different safety information. Relate ground truth to **your downloaded PDFs**, not general model knowledge.
- False positives, cross-drug substitutions, and unsupported medical claims must be discussed independently from the retrieval scores.
- No model answers are used as gold labels.
- If a question has missing labels or the local vector count differs from 7,194, the evaluator stops rather than reporting misleading averages.

## What to share for the next stage

Start by sharing the output of:

```bash
python -m unittest discover -s tests -p 'test_ir_*.py' -v
python -m scripts.evaluate_ir --prepare --top-k 10 --require-count 7194
```

After manual labeling, share the two measured metrics and we'll incorporate them into the final README and 5–10-page report.
