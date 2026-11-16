# Step 9.1 — Fix unnecessary abstention + citation-only abstention

This patch addresses the exact output observed on Windows with Qwen2.5 7B:

```
I don't have enough information in the provided documents. [S1], [S2], [S3]
```

This is an abstention with extraneous citation IDs, not an answer containing medical claims. The old validator classified it as `rejected_mixed_abstention`.

## Changes

1. **`src/prompt_builder.py`** — Revised instructions emphasize: **answer only the supported portion** if at least one retrieved passage explicitly answers the question; qualify the scope of incomplete passages; do not pretend a limited list of warnings is exhaustive. Abstain only when there is no relevant support. The abstention must be exactly one sentence without citations. The original no-hallucinations, no-cross-drug, no-personalized-advice, source-as-untrusted-data requirements are retained.
2. **`src/ollama_generator.py`** — A response consisting of the exact abstention sentence plus *only valid in-scope citation tags* is safely normalized to `status=abstained`, with citations removed. Unknown IDs, uncited substantive answers, fabricated IDs, and any actual mixture of a claim and abstention are still rejected. This is not an attempt to turn an abstention into an answer.
3. **`tests/test_step9_answering_fix.py`** — 9 added offline regression tests, including your exact observed raw response and safeguards against unsupported claims.

## Install on Windows

Extract the ZIP into your existing project root: `D:\github\drug-information-assistant`. Merge the `src/` and `tests/` folders; **replace both existing `src/prompt_builder.py` and `src/ollama_generator.py`**. Do not replace `scripts/build_prompt.py`, `scripts/generate_answer.py`, embeddings, ChromaDB, or drug PDFs.

### 1. Test all modules (Git Bash with `.venv` activated)

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

With the existing 148 tests unchanged, expect **157 tests**.

### 2. IMPORTANT: Rebuild the Step 8 prompt

Your previous `ibuprofen_warnings_prompt.json` embeds the *old* system instructions. Step 9 strictly validates the system message against the current prompt builder, so you must rebuild from the saved Step 7 evidence:

```bash
python -m scripts.build_prompt \
  --retrieval-json data/retrieval_runs/ibuprofen_warnings.json \
  --max-sources 3 \
  --json-out data/prompts/ibuprofen_warnings_prompt_v2.json \
  --text-out data/prompts/ibuprofen_warnings_prompt_v2.txt
```

This uses the original three *complete* retrieved passages; nothing is cut or reindexed.

### 3. Generate one answer

```bash
python -m scripts.generate_answer \
  --prompt-json data/prompts/ibuprofen_warnings_prompt_v2.json \
  --model qwen2.5:7b-instruct \
  --num-ctx 8192 \
  --max-output-tokens 512 \
  --json-out data/answers/ibuprofen_warnings_answer_v2.json \
  --text-out data/answers/ibuprofen_warnings_answer_v2.txt
```

If the conservative byte-based context checker fails, rebuild with `--max-sources 2` and test again. Keep full chunks. If Ollama memory allocation fails, reduce context size **only if your prompt fits the preflight budget**.

### 4. Inspect status and answer

```bash
cat data/answers/ibuprofen_warnings_answer_v2.txt
python -c "import json; p=json.load(open('data/answers/ibuprofen_warnings_answer_v2.json',encoding='utf-8')); print('Status:',p['status'],'| Citations:',p['citations'],'| Prompt tokens:',p['usage']['prompt_eval_count'])"
```

**Possible statuses:** `answered` means well-formed citations, not confirmed factual support. `abstained` means the model still didn't produce a substantive answer; if citations were appended to its abstention, they were removed safely. `rejected_*` means the answer failed validation. None of these outcomes alone proves medical correctness.

## What is NOT changed

- No vectors, embeddings, chunking, retrieval ranking, DailyMed PDFs, or Ollama installation changes.
- No removal of citations or abstention safety checks for substantive medical claims.
- No claim that a citation points to a medically supported claim (manual assessment still required).

If the updated prompt still abstains, inspect `data/prompts/ibuprofen_warnings_prompt_v2.txt` and the raw response to determine whether model evidence interpretation, not the format, is the issue. An easier diagnostic question is: **"What does the ibuprofen label say about stomach bleeding?"**, with fresh drug-aware retrieval and a newly built prompt.
