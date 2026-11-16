# Step 8 — Manual Prompt Construction & Evidence Grounding

**Project:** Drug Information Assistant (local RAG; no LangChain / LlamaIndex / Haystack)

**Prerequisites:** Complete Steps 1–7. In particular, you already have **100 DailyMed label PDFs**, **7,194 BGE embeddings** in a local ChromaDB collection, and drug-aware Top-5 retrieval in `scripts/retrieve.py`. The counts are from your verified dataset after the targeted token-overflow repair.

## 1. Learning goal

Retrieval finds relevant passages. **Prompt construction** packages the *question* and *retrieved evidence* into explicit instructions and source data for an LLM. It does not itself check factual truth, make medical decisions, or generate an answer.

We create two chat messages:

- **System message:** Source grounding, citations, exact abstention phrase, drug/formulation boundaries, no made-up medical information, and document text treated as untrusted data.
- **User message:** The actual question, followed by 1–5 complete labeled passages, each with its PDF filename, page, character offsets, source URL, and chunk ID.

The output `messages` list is formatted for a later local Ollama `/api/chat` request. **Step 8 does not call Ollama.**

## 2. Provided files

```
src/prompt_builder.py        # New; manual source-grounded prompt builder
scripts/build_prompt.py      # New; converts saved retrieval JSON to prompt JSON/text
tests/test_prompt_builder.py # New; 15 offline tests
README_STEP8.md              # This guide
```

Extract this ZIP into your project root, preserving the folders. **Do not replace or modify** existing embedding, chunk, index, or retrieval files.

## 3. How source citations work

A label such as `[S2]` is assigned to the second retrieved passage and maps to its `filename`, `page_number`, and `source_url`. The prompt includes the *complete* chunk text rather than just its CLI preview. The later LLM is instructed to cite only supplied IDs.

Example (illustrative text, **not an actual medical answer**):

```
QUESTION: What warnings are listed for ibuprofen?
REFERENCE PASSAGES:
  {"source_id":"S1", "filename":"ibuprofen_200mg_...pdf", "page_number":2,
   "source_url":"https://dailymed.nlm.nih.gov/...", "text":"..."}
  ...
```

Source text is JSON-quoted in the prompt. Any misleading instructions that appear *inside the source text* remain reference data, not instructions to the assistant. Prompt injection defenses are not perfect; later model outputs still need citation validation and evaluation.

## 4. Commands (Windows VS Code / Git Bash, active .venv)

**Test Python syntax:**

```bash
python -m py_compile src/prompt_builder.py scripts/build_prompt.py
```

**Run the complete test suite:**

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: **125 tests; OK**, assuming your previous 110 tests are unchanged. The authoring environment showed **125 tests, OK (skipped=2)** because it lacks real ChromaDB; both integration tests have already passed on your Windows setup, where they should run normally.

**Build a full ibuprofen prompt from the saved Step 7 results:**

```bash
python -m scripts.build_prompt \
  --retrieval-json data/retrieval_runs/ibuprofen_drug_aware.json \
  --json-out data/prompts/ibuprofen_prompt.json \
  --text-out data/prompts/ibuprofen_prompt.txt
```

The console should show:

```
Prompt status: ready
Question: What are the adverse reactions of ibuprofen?
Sources included: 5 / 5
  [S1] ibuprofen_200mg_haleon_us_holdings_llc.pdf | page ... | chunk ...
  ...
This step builds a prompt only. It does NOT call Ollama or generate an answer.
```

The actual page numbers and source count come from your retrieved evidence JSON. We expect five sources from your previously shared Step 7 output.

**Inspect the output:**

- `data/prompts/ibuprofen_prompt.txt` — human-readable complete system and user prompts.
- `data/prompts/ibuprofen_prompt.json` — ready for the next step's Ollama integration; includes the `messages`, `sources`, `status`, and `fallback_answer` fields.

**Optional: create a fresh acetaminophen JSON first and build its prompt:**

```bash
python -m scripts.retrieve \
  --question "What warnings are listed for acetaminophen?" \
  --top-k 5 --require-count 7194 \
  --json-out data/retrieval_runs/acetaminophen_drug_aware.json

python -m scripts.build_prompt \
  --retrieval-json data/retrieval_runs/acetaminophen_drug_aware.json \
  --json-out data/prompts/acetaminophen_prompt.json \
  --text-out data/prompts/acetaminophen_prompt.txt
```

**Inspect a JSON summary without launching any neural model:**

```bash
python -c "import json; p=json.load(open('data/prompts/ibuprofen_prompt.json',encoding='utf-8')); print('Status:',p['status'],'| Sources:',p['source_count'],'| Chat messages:',len(p['messages'])); print('Citation pages:',[(s['source_id'],s['page_number']) for s in p['sources']])"
```

## 5. Evidence guardrails

- **No citations or zero passages:** `status="no_evidence"`, `messages=[]`, exact fallback: `I don't have enough information in the provided documents.` The later generator should not be called.
- **Wrong drug:** If the Step 7 JSON selected a PDF, a passage from another PDF is rejected.
- **Multiple drug sources:** By default, a globally retrieved set from several different drug PDFs returns `status="needs_source_scope"` rather than a combined prompt. Only for explicitly broad/comparative tasks may you opt in with `--allow-multiple-documents`; never conflate drugs or formulations.
- **Context budget:** The builder includes up to five complete passages, max ~10,000 serialized evidence characters by default. If the budget is exceeded, *entire later passages are omitted*; the program never cuts off a warning or dosage instruction midway. Use `--max-context-chars 18000` to increase the limit if necessary. A context character budget is **not** a token limit; the next stage must verify Ollama model context capacity.
- **Cosine similarity is NOT a confidence score**, medical correctness guarantee, or proof the reference actually answers the question.
- **Medical safety:** The system prompt does not authorize personalized medication directions. A local LLM may still hallucinate, so Step 9 must handle unsupported claims and validate citations before exposing answers.
- **Offline:** No network, ChromaDB, embeddings, or Ollama are used when building a prompt from an existing JSON file.

## 6. Learning checkpoint

Explain (a) what the system and user messages do, (b) the distinction between retrieval and answering, (c) why citations use PDF page metadata, (d) why abstention matters, and (e) why another medicine's highly similar chunk must not be silently used as evidence for ibuprofen.

**Next:** Step 9 — Local LLM generation with Ollama, model-context checks, response verification, and safe citation rendering. We will implement it only after reviewing the Step 8 test and prompt-building output.
