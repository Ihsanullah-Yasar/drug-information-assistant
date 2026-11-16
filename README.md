# Drug Information Assistant

A local Drug Information Assistant for an Information Retrieval course assignment. The planned application will retrieve relevant passages from collected drug labels and use them to ground answers to user questions.

## Current status: PDF loading implemented

PDF loading and raw-text storage are implemented. Ingestion reads PDFs from `data/documents/` in filename order and writes one readable UTF-8 JSON array per PDF to `data/raw/<pdf_stem>.json`. Every page is preserved, including pages with empty text, with its filename, 1-based page number, and unmodified extracted text. Re-running replaces each output file. Missing input directories, zero PDFs, and unreadable PDFs produce an error and a nonzero exit status.

Cleaning, chunking, embedding, indexing, retrieval, model integration, and the Gradio interface remain scaffolded. `requirements.txt` currently lists only PyMuPDF; other dependencies will be added as their stages are implemented. No models are needed for PDF loading.

## Run PDF loading

From PowerShell, using the project virtual environment:

```powershell
Set-Location 'D:\github\drug-information-assistant'
.\.venv\Scripts\python.exe -m scripts.ingest
```

The command reports each filename, page count, any empty-text page numbers, and the output path. The original PDFs and source CSV are not modified.

## Planned workflows

1. **Ingestion:** PDF → page records → cleaning → chunking → embeddings → ChromaDB.
2. **Question answering:** question → query embedding → top-k passages → grounded prompt → local Ollama model → answer in Gradio.

## Planned stack

- Python 3.11+
- PyMuPDF for PDF loading
- BAAI/bge-small-en-v1.5 for embeddings
- ChromaDB for vector storage and search
- Ollama for local model serving
- Qwen2.5-7B-Instruct for answer generation
- Gradio for the question-answering interface

## Project layout

```text
app.py                 Planned Gradio interface
src/
    __init__.py        Empty package initializer
    loader.py          PDF loading into page records
    cleaner.py         Page text cleaning
    chunker.py         Passage chunking
    embedder.py        Passage and query embeddings
    vectordb.py        ChromaDB storage and search
    retriever.py       Top-k passage retrieval
    prompts.py         Grounded prompt construction
    llm.py             Local Ollama integration
scripts/
    __init__.py        Empty package initializer
    ingest.py          PDF loading and raw JSON entry point
data/
    documents/         Original PDFs
    sources.csv        Source metadata and original URLs
    raw/               Extracted page records as JSON arrays
    chroma/            Reserved for the ChromaDB index
requirements.txt       Dependencies for implemented stages
README.md
.gitignore
LICENSE
```

Generated data in `data/raw/` and `data/chroma/`, the local virtual environment, and Python bytecode are ignored by Git. Original PDFs and `data/sources.csv` remain eligible for version control. Empty directories exist locally but are not tracked by Git.
