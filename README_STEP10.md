# Step 10 — Gradio User Interface for the Local Drug Information Assistant

## What this stage adds

The earlier steps are already working on Windows:

- **100 DailyMed drug-label PDFs** (2,292 pages).
- **7,194** local BGE-small-en-v1.5 vectors stored in ChromaDB.
- **Drug-aware Top-5 retrieval** (one exact source PDF by default).
- **Manual, fully cited prompt construction**.
- **Ollama `qwen2.5:7b-instruct`**, tested by the user, including a complete cited ibuprofen warning answer.
- **157 previous unit tests passing** on the user's computer.

This patch creates a local browser UI that runs the pipeline automatically:

```text
User question
   |
   v
Single-drug routing (or manually selected exact PDF label)
   |
   v
Local ChromaDB + BGE-small query embedding -> Top-5 chunks
   |
   v
Manual prompt builder -> up to 3 COMPLETE passages by default
   |
   v
Local Ollama Qwen2.5 7B -> citation-ID validation
   |
   v
Plain-text answer + source excerpts, page citations, and status
```

**Important:** No documents are collected or downloaded, no corpus passage vectors are recalculated, no ChromaDB index is written, and no cloud model is contacted. Model weights may be loaded from the existing local Hugging Face cache on first use.

## Install — Windows + Git Bash

Project root:

```text
D:\github\drug-information-assistant
```

Unzip `Step10_Gradio_Interface_Ready.zip` **into the project root**, merging the folders. It adds these files:

```text
src/rag_pipeline.py
scripts/app.py
tests/test_rag_pipeline.py
README_STEP10.md
requirements-step10.txt
```

Your existing Steps 1–9 source files, embeddings, corpus PDFs, ChromaDB and previous answers are not replaced.

**1. Activate your existing virtual environment if necessary:**

```bash
source .venv/Scripts/activate
```

**2. Install the only new interface package:**

```bash
python -m pip install -r requirements-step10.txt
```

This specifies Gradio 6.x. The official Gradio package currently states Python >=3.10 support. If `pip` reports a Windows/Python 3.14 incompatibility, share the actual error instead of changing your working interpreter arbitrarily.

**3. Run your complete test suite:**

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected **177 tests** if your previous 157 tests remain unchanged and the 20 new tests are detected. On the authoring machine, all 20 new tests passed and 143 tests from available modules passed (2 optional Chroma integrations skipped). The authoring copy does not include your existing `scripts/ingest.py`, so the full 177-test suite can only be confirmed on your Windows project.

**4. Make sure Ollama is running and the 7B model is available:**

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

You already installed `qwen2.5:7b-instruct` and verified it responds. If the earlier RAM allocation issue returns, free available RAM before attempting larger context settings.

**5. Start the web UI from the project root:**

```bash
python -m scripts.app
```

It should open a tab automatically at:

```text
http://127.0.0.1:7860
```

If it does not open automatically, paste the address in your browser. Use **Ctrl+C** to stop the local server.

## First end-to-end question

In the web page:

1. In **Your question**, enter **What safety warnings are listed for ibuprofen?**
2. Leave **Source label (optional)** as **Auto-detect from question**.
3. Leave **Maximum complete evidence passages** at **3**.
4. Click **Find answer**.

The pipeline should select:

```text
ibuprofen_200mg_haleon_us_holdings_llc.pdf
```

It searches that PDF's existing Chroma vectors (Top-5), includes up to three complete retrieved passages, and calls local Ollama with `num_ctx=8192`, `num_predict=512`.

The page should show:

- The **answer** with citations such as `[S1]`, `[S2]` if model output passes the existing citation validator.
- **Validation status**, including `answered`, `abstained` or a rejection, retrieval count, included passages and reported prompt tokens.
- A **source table** with citation IDs, exact PDF filenames, page and chunk numbers, a plain-text evidence excerpt, original DailyMed URL and whether the passage was cited.

**`answered` means citation IDs passed structural checks, NOT that every medical statement was verified.** Inspect the quoted evidence and original DailyMed page. This is an educational project, not a prescribing/diagnostic tool.

## Important design decisions

- **Local only:** `server_name="127.0.0.1"`, `share=False`. Do not change to `0.0.0.0` or enable sharing for a medical demo without security/privacy review.
- **No cross-drug substitution:** If the question names two drugs, auto-routing refuses the query. Select one exact PDF to scope a specific label if appropriate, and do not treat one medicine's warnings as another medicine's.
- **One source per answer by default:** If no recognized drug is mentioned and no PDF is selected, the app asks for a specific drug rather than searching and merging unrelated medicines.
- **No silent truncation:** The context check can reduce the **number** of included passages from 3 to 2 or 1, but it never cuts a passage in half.
- **No hidden caching to disk:** Model and collection objects are held in the UI process for reuse; no questions, prompts or answers are saved automatically.
- **The query embedding is computed with the same BGE model as Step 7;** the existing 7,194 document embeddings are not recomputed.
- **Gradio callbacks use one concurrent inference at a time** to avoid overloading limited system RAM.
- **Not a live clinical database:** This is a static collection of specific label versions; the program does not check for updated warnings or changes to DailyMed.

## Troubleshooting

| Message | Meaning / action |
| --- | --- |
| `ModuleNotFoundError: No module named 'gradio'` | Run `python -m pip install -r requirements-step10.txt` inside `.venv`. |
| `Expected 7194` | Your Chroma collection count does not match the verified corpus. Run `python -m scripts.index_vectors --audit-only`. |
| `needs_drug_scope` | Enter a recognized drug name or select one PDF from the dropdown. |
| `Multiple drugs mentioned` | Select a single PDF to scope the request; cross-drug comparisons require a separate retrieval design. |
| `context_budget_exceeded` | A complete passage cannot fit the current 8192 budget; no passage was truncated. |
| `Cannot reach local Ollama` | Open Ollama in Windows, verify `curl http://127.0.0.1:11434/api/tags`. |
| `unable to allocate CUDA_Host buffer` | Free RAM / close applications, then try again; the 7B model previously worked after freeing RAM. |
| `abstained` or `rejected_*` | The local model refused/failed source or output validation; do not invent an answer. |
| Port 7860 already in use | Stop an earlier Gradio server with Ctrl+C and try again. |

## Assignment continuity

At this point you've implemented collection, loading, preprocessing, chunking, embedding, indexing, retrieval, prompting, Ollama generation, and a local UI. **The next milestone is Step 11: evaluation**, including a labeled question set, **Hit Rate@5** and **MRR**, followed by final README and a 5–10 page report. This requires writing gold relevance labels for each question; it is not enough to look only at model citations.
