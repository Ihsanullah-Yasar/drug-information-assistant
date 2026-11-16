# Step 7 — Top-K Semantic Retrieval (Drug Information Assistant)

## Verified input from the previous stage

Your **full Step 6 ChromaDB audit passed**:

- 100 verified DailyMed source PDFs; 2,292 PDF pages.
- 7,194 non-empty chunk embeddings after the one-page token-overflow repair.
- 7,194 unique chunk IDs, dimension 384.
- Persistent collection `drug_labels_bge_small_v15` in `data/chroma/`.
- Chroma is configured for cosine distance.

**Do not recreate or delete `data/chroma/`, `data/embeddings/`, or `data/chunks/`.** This stage is read-only.

## Where Step 7 fits

`PDFs → Extract → Clean → Chunk → BGE passage embeddings → ChromaDB → **BGE query embedding + Top-K search** → Prompt → Ollama → Gradio`

Given a question, the retriever:

1. Opens your existing Chroma collection **without creating a new one**.
2. Checks the index is not empty; optionally demands exactly 7,194 records.
3. Uses **the same `BAAI/bge-small-en-v1.5` model** to encode the question.
4. Uses `BGEEmbedder.encode_query()` from Step 5. This adds the model's recommended short-query prefix `Represent this sentence for searching relevant passages: `. Passage embeddings were created *without* that prefix; keep this distinction.
5. Performs `collection.query(query_embeddings=[vector], n_results=5, include=["documents", "metadatas", "distances"])`. With `--filename`, adds `where={"filename": exact_pdf_name}`.
6. Validates result structure, source metadata, and cosine-distance ordering.
7. Prints each text preview with PDF filename, page number, chunk number, DailyMed URL, cosine distance, and similarity.
8. Optionally saves the **full text** of the results to JSON for the prompt-construction stage.

**It does not generate an answer, run Ollama, change the index, or give medical advice.** A top-ranked chunk may still be irrelevant or incomplete.

## Files to install (only three new files)

Extract the ZIP into `D:/github/drug-information-assistant` preserving the folders:

```text
src/retriever.py             # new: query embedding, ranking and validation
scripts/retrieve.py          # new: Git Bash CLI + JSON evidence export
tests/test_retriever.py      # new: 12 offline tests + 1 optional real Chroma test
README_STEP7.md              # explanation and commands
```

There is no need to overwrite `src/embedder.py`, `src/vectordb.py`, or any existing script/data.

## A. Syntax and automated tests (Git Bash, activated .venv)

```bash
python -m py_compile src/retriever.py scripts/retrieve.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Target on your Windows machine: **98 tests passed** (previous 85 + 13 Step 7). The real Chroma smoke test runs in a child process to avoid Windows HNSW file locks. In the authoring environment, 96 tests passed with 2 Chroma-dependent tests skipped because Chroma is not installed there. All 12 new non-Chroma Step 7 tests passed.

If your local test count differs because you have added or removed tests, check the summary rather than relying only on the number.

## B. Start with a drug-specific query

You have `acetaminophen_325mg_health_pharma.pdf` in your dataset. Run:

```bash
python -m scripts.retrieve --question "What warnings are listed for acetaminophen?" --filename acetaminophen_325mg_health_pharma.pdf --top-k 5 --require-count 7194 --json-out data/retrieval_runs/acetaminophen_warnings.json
```

This query is **restricted to the selected PDF**. The model must be loaded to embed the query; the model weights should already be in your local Hugging Face cache from Step 5. The result should contain up to five ranked chunks, each referring only to `acetaminophen_325mg_health_pharma.pdf`. Actual ranking, distances, and text depend on the model's output, so they are not known in advance.

The `--json-out` file contains all retrieved passage text (the terminal only previews the first 300 characters), plus the source PDF/page and DailyMed link. It does not change Chroma.

## C. Search across all 100 drug labels

```bash
python -m scripts.retrieve --question "What are the adverse reactions of ibuprofen?" --top-k 5 --require-count 7194 --json-out data/retrieval_runs/ibuprofen_adverse_reactions.json
```

With **no `--filename`**, all 7,194 vectors are candidates. Read the top five sources critically: some nearest passages may concern other medicines or the wrong formulation. This demonstrates why vector similarity is not itself proof that a question is answerable.

### Optional: restrict to the exact ibuprofen label

```bash
python -m scripts.retrieve --question "What adverse reactions does the ibuprofen label list?" --filename ibuprofen_200mg_haleon_us_holdings_llc.pdf --top-k 5 --require-count 7194
```

The filter uses the *exact PDF filename* from your `data/sources.csv`, not a generic drug-name substring.

## Interpreting the scores

With the cosine collection, Chroma returns `cosine_distance = 1 - cosine_similarity`. **Smaller distance means closer vectors**. We print similarity for interpretation only; it is not a calibrated probability or proof that a medical claim is supported.

Example *invented for explanation* (not actual results):

```text
[1] cosine distance: 0.2400 | similarity: 0.7600
    PDF: ibuprofen_200mg_haleon_us_holdings_llc.pdf | page 6 | chunk 2
    Text: ...
    Source: https://dailymed.nlm.nih.gov/...
```

Chroma's Top-K search may be approximate, and overlapped chunks on the same page may both appear in the results. This is expected at this step.

## Why metadata matters

The retrieved chunk includes: `chunk_id`, `text`, `filename`, `page_number`, `chunk_index`, `start_char`, `end_char`, `source_url`, `packager`, `label_updated_on`, `cosine_distance`, `cosine_similarity`, and `rank`. Cite the *actual PDF page and original DailyMed label* in later answers, and avoid mixing unrelated formulations or label versions.

## Troubleshooting

- `The Chroma collection is empty`: Step 6 index directory is wrong or indexing has not completed.
- `Expected 7194 indexed vectors, found ...`: rerun `python -m scripts.index_vectors --audit-only` and investigate; don't reset the database.
- `Unable to open collection`: open the project from the correct root so `data/chroma/` is found.
- `--filename must be exact`: look in `data/sources.csv` and copy the exact PDF filename.
- `Query has too many tokens`: shorten the natural-language question; the model's 512-token limit still applies.
- `ModuleNotFoundError: sentence_transformers`: activate the same `.venv` used in Step 5.
- `No indexed chunks match the filter`: check the exact filename; the retriever returns an empty evidence list rather than fabricating a match.
- `FutureWarning` from Sentence Transformers or `DeprecationWarning` from Chroma: these are usually nonfatal warnings; share the traceback if the command fails.

## Next stage

After verifying real retrieval on your machine, Step 8 will construct a **manual evidence-grounded prompt** using the top five chunks, then call your local Ollama model. The prompt will instruct the assistant to state when the retrieved documents do not provide enough information, and to cite the PDFs/pages. We will evaluate Hit Rate@5 and MRR in the later evaluation stage.

## Upstream API references

- BGE model card: https://huggingface.co/BAAI/bge-small-en-v1.5
- Chroma collection query: https://cookbook.chromadb.dev/core/collections/
