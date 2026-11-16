# Step 6 — Local ChromaDB Vector Database

## What you already have

- 100 verified DailyMed PDFs and 100 records in `data/sources.csv`.
- 2,292 pages, 2,284 containing extractable text.
- 7,193 uniquely identified non-empty chunks in `data/chunks/`.
- Step 5 is producing 384-dimensional BGE passage vectors in per-document `.npz` archives.

**Do not delete or change** `data/documents/`, `data/raw/`, `data/cleaned/`, `data/chunks/`, `data/sources.csv`, or any valid `data/embeddings/` archives. We reuse them exactly.

## Where this step fits

Documents -> cleaning -> chunks -> BGE embeddings -> **ChromaDB index (this step)** -> Top-5 retriever (Step 7) -> prompt -> Ollama -> Gradio.

Chroma is a persistent vector database. It stores each chunk's 384-dimensional vector, the actual chunk text, its stable ID, and source metadata. HNSW (Hierarchical Navigable Small World) supports fast *approximate* nearest-neighbor search instead of comparing all vectors individually for every question. We use cosine distance, NOT Chroma's default L2 distance.

The application does not run Ollama or load the BGE model in this step. Embeddings are imported from the files Step 5 already generated. No LangChain, LlamaIndex, or Haystack.

## Files to copy into your existing repository

```
src/vectordb.py           # new, previously empty
scripts/index_vectors.py  # new indexing entry point
tests/test_vectordb.py    # 13 Step 6 tests, including a real Chroma smoke test
```

Unzip into `D:/github/drug-information-assistant`, preserving `src/`, `scripts/`, `tests/`. No other Python source files need replacing.

## Installation (Git Bash terminal, from the project root)

```bash
python -m pip show chromadb
```

If missing:

```bash
python -m pip install "chromadb==1.5.9"
```

This version provides the tested API contract (`PersistentClient`, `configuration={"hnsw":{"space":"cosine"}}`, `collection.upsert`, `collection.get`, `collection.query`). The ChromaDB wheel published on PyPI is `cp39-abi3-win_amd64` and supports modern CPython, including your Python 3.14 environment. Package installation requires internet access.

### A. Syntax and automated tests

```bash
python -m py_compile src/vectordb.py scripts/index_vectors.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected **80 tests total** if the existing 67 are unchanged. The new suite includes 12 offline fake-database tests and one real Chroma test. The real test runs after ChromaDB is installed; otherwise it is skipped. Note: the 12 fake tests passed in the authoring environment; the live Chroma integration test could not run there because ChromaDB was not available.

### B. Index only 2 completed documents

You can run this while the long Step 5 embedding process continues in another terminal, because the **first two** `.npz` files are already complete:

```bash
python -m scripts.index_vectors --limit-files 2 --batch-size 32
python -m scripts.index_vectors --limit-files 2 --audit-only
```

The first command creates a local collection named `drug_labels_bge_small_v15` inside `data/chroma/`. It writes acetaminophen and acyclovir only, then verifies them. The second command verifies the two records against the source files without changing Chroma. Expected count is **42 vectors** (7 acetaminophen, 35 acyclovir). A partial audit does **not** claim the entire 100-document collection is complete.

You should see output like:

```
[1/2] acetaminophen_325mg_health_pharma.json | indexed | 7 vectors
[2/2] acyclovir_400mg_nucare_pharmaceuticals_inc.json | indexed | 35 vectors
Chroma audit: 2 documents checked; 42 vectors; 42 distinct IDs; dimension 384; collection records 42; ...
Partial audit: the entire 100-document collection has NOT yet been verified.
```

### C. Full index only AFTER Step 5 finishes

Wait until you have successfully run:

```bash
python -m scripts.embed --audit-only
```

Expected: 100 source documents, 7,193 embeddings, and 7,193 unique IDs (if the chunk files are unchanged).

Then:

```bash
python -m scripts.index_vectors --batch-size 64
python -m scripts.index_vectors --audit-only
```

Expected final result:

```
Chroma audit: 100 documents checked; 7193 vectors; 7193 distinct IDs; dimension 384; collection records 7193; ...
Full index audit passed: all expected sources and vectors match ChromaDB.
```

No model is downloaded or used during the indexing stage.

## What each Chroma record stores

- **id:** exact SHA-256 `chunk_id` generated in Step 4
- **embedding:** precomputed 384-dimensional normalized BGE vector from Step 5
- **document:** exact chunk text
- **metadata:** `filename`, `page_number`, `chunk_index`, `start_char`, `end_char`, `source_url`, `packager`, `label_updated_on`, `source_sha256`, `vector_sha256`

Metadata fields retain the evidence needed to display exact filename, page and original DailyMed URL in future answers.

## Safe resumption

For every document, the script checks what Chroma already contains. When the stored text, metadata, IDs and numeric vectors are all equal to the current source files, it **reuses** them. When a document changes, it upserts its new records, then removes stale IDs for that same PDF. If the process stops mid-document, rerun the same command to reconcile it. It never resets the full database or deletes a different PDF's records.

The full audit requires that the collection have exactly as many records as the corpus (7,193 with your current chunks), and each expected document must match the original archive. A partial `--limit-files` audit checks only its selected files and reports clearly that it is partial.

**Do not launch two simultaneous Step 6 index writers for the same `data/chroma/` directory.** Chroma's local PersistentClient is not designed for competing independent processes writing to the same database path.

## How distance works

Cosine similarity is

```
cos(q, d) = (q · d) / (||q|| ||d||)
```

Chroma returns **cosine distance** `1 - cosine_similarity` for a cosine collection: a *smaller* distance means more similar chunks. Step 7 will encode a question using the **same BGE model** and query `collection.query(query_embeddings=[...], n_results=5)` to retrieve the top 5 relevant chunks. The existence of a nearest neighbor alone does not prove that a question is answerable; we will add abstention logic and a retrieval evaluation dataset in later steps.

## Troubleshooting

- `ModuleNotFoundError: chromadb`: install Chroma inside your activated `.venv`.
- `Missing embeddings ... Complete Step 5`: the current PDF hasn't finished embedding; test with `--limit-files 2` and retry after Step 5's full audit.
- `Incompatible Chroma collection metadata`: old/incompatible model or metric. Don't combine two embedding models in one collection; investigate before creating a separate collection.
- `Missing, stale, or incorrect Chroma records`: an audit detected mismatch; run the normal indexing command to repair, then re-audit.
- `Chroma collection contains ... records` after a full run: an unrelated record remained in this named collection; stop rather than deleting records without review.
- Network warnings from the BGE model in Step 5 are unrelated to Chroma's local database.

## Next

Once Step 6's real 2-document indexing test passes, share your terminal output. Then complete Step 5's 100-document embedding audit and finish the full Chroma index. **Step 7 is the retriever**: embed a query, retrieve top 5 chunks, inspect cosine distances, and measure Hit Rate@5 and MRR.
