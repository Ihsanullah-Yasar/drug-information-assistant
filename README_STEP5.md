# Step 5 — Semantic Embedding Generation

## Where this fits

Your verified corpus has 100 PDF documents -> 2,292 extracted/cleaned page records -> **7,193 chunks** (`data/chunks/*.json`). Step 5 produces **one numeric vector per nonempty chunk**. Step 6 will put those vectors, their chunk text, and their metadata into ChromaDB. No LangChain, LlamaIndex, or Haystack.

## Model and concepts

- **Model:** `BAAI/bge-small-en-v1.5` (the model recommended in the teacher's manual).
- **Embedding dimension:** 384. Each chunk becomes a 384-element float32 vector.
- **Input limit:** 512 model tokens including special tokens; **characters are not tokens**. Step 5 checks real token lengths and refuses to silently truncate medical information.
- **L2 normalization:** `normalize_embeddings=True` makes each vector have norm approximately 1. A query and a chunk vector can then be compared using the dot product, equivalent to cosine similarity for normalized vectors.
- **Query handling:** the model's recommended short-query instruction is added when using `encode_query()`. Passage text is encoded **without** this instruction.
- **The embedding model does not generate answers.** It maps text into a vector space; Ollama will generate grounded answers in a later step.

## Install the files

Copy exactly these files from the ZIP into the existing project, preserving the folder paths:

- `src/embedder.py` (your prior file was empty)
- `scripts/embed.py` (new command-line stage)
- `tests/test_embedder.py` (12 new offline tests)

Do not replace `src/chunker.py`, `scripts/chunk.py`, the 100 PDF files, `data/chunks/`, or the source CSV.

## Commands — Git Bash, from the project root

First, check whether the required Python packages are installed:

```bash
python -m pip show sentence-transformers numpy
```

If sentence-transformers is missing, install it into your active virtual environment:

```bash
python -m pip install -U sentence-transformers numpy
```

The first model load needs internet access to download the weights from Hugging Face. Later runs can use the cached model. The default device is CPU; no GPU is required.

### Phase A: syntax and offline tests

```bash
python -m py_compile src/embedder.py scripts/embed.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

If previous 55 tests are unchanged, expect **67** tests. The 12 new tests use a fake model and therefore do not download real weights.

### Phase B: real model on two documents only

```bash
python -m scripts.embed --limit-files 2 --batch-size 8
python -m scripts.embed --limit-files 2 --audit-only
```

The **first run** may download the real BGE model. It should write two `.npz` archives in `data/embeddings/`; the **audit-only** command checks those saved files without loading the model.

### Phase C: entire dataset

```bash
python -m scripts.embed --batch-size 16
python -m scripts.embed --audit-only
```

If all documents succeed, the final audit should show:

```
Embedding audit: 100 documents; 7193 embeddings; 7193 distinct IDs; dimension 384; ...
```

The actual output's `saved`/`reused` counts depend on how much work has already been completed.

## File format and safe resumption

Each `data/chunks/example.json` creates `data/embeddings/example.npz`, containing:

- `embeddings` — float32 NumPy matrix shaped `[number_of_chunks, 384]`
- `chunk_ids` — exactly aligned, same order as the corresponding chunk JSON
- `source_sha256` — checksum of the complete source chunk JSON
- `model_name` — ensures the embeddings came from the expected model
- `dimension`, `normalized` — validates settings

The writer uses temporary files and atomic replacement. A rerun validates and **reuses** existing correct archives. If a source chunk file changes, its archive is regenerated. Do not change source metadata or overwrite archive files manually.

Step 6 will load a `.npz` using `np.load(path, allow_pickle=False)` and pair each vector with the corresponding JSON record by `chunk_ids`.

## If you get errors

- **`ModuleNotFoundError: sentence_transformers`**: run the pip install command above.
- **Model download / SSL / connectivity error**: verify network access to `huggingface.co`, then rerun (the earlier successful `.npz` files are reused).
- **Out-of-memory error**: rerun with `--batch-size 4` (or `--device cpu`).
- **`Text ... has ... tokens, exceeding model limit 512`**: never silently truncate safety-related text. Run the chunker again with smaller chunks, e.g. `python -m scripts.chunk --chunk-size 800 --overlap 180`, then rerun this stage. Previously saved archives will be regenerated as necessary because their source checksum changed. This changes the total number of chunks; update the expected count in your audit.
- **`Outdated or corrupt saved vectors`** during `--audit-only`: run the normal embedding command again; it reconstructs the affected archive.

## Explaining this step to your teacher

1. We extracted **7,193** text chunks with source-page references.
2. We chose the specified **BGE-small v1.5** model to convert each chunk into a **384-dimensional semantic vector**.
3. We token-count inputs so no label content is silently truncated.
4. We encode in memory-controlled batches, normalize the vectors, and save them in matching `.npz` archives.
5. We validate and persist chunk-to-vector identity so the next stage can index vectors without losing PDF/page citations.

## Done when

- The extended tests pass.
- Real two-document trial succeeds.
- The full command creates **100** `.npz` archives and the audit reports **7,193 distinct chunk/vector IDs** (assuming Step 4 chunks unchanged).
- `--audit-only` succeeds without downloading or encoding anything.
