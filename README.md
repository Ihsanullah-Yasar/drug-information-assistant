# Drug Information Assistant

Local, source-grounded question answering over FDA DailyMed drug labels.

This project is a complete retrieval-augmented generation (RAG) pipeline for an Information Retrieval course assignment. It retrieves passages from a curated local corpus of drug-label PDFs, builds citable prompts, and answers with a local Ollama model (`qwen2.5:7b-instruct`). Answers are fail-closed: citation IDs are validated before they are shown, and the system abstains when evidence is insufficient.

> **Educational use only.** This assistant is not a clinical decision-support tool. It does not diagnose, prescribe, or replace professional medical judgment.

---

## Features

- **100 DailyMed labels** collected and indexed locally
- **Manual RAG pipeline** (no LangChain / LlamaIndex / Haystack)
- **Drug-aware retrieval** with deterministic source routing
- **Grounded generation** via local Ollama + Qwen2.5 7B Instruct
- **Streaming Gradio chat UI** with stop/cancel and citation panels
- **IR evaluation** (Hit Rate@K, MRR@K) over manually reviewed gold labels
- **Fail-closed validation** for missing, malformed, or out-of-scope citations

---

## Architecture

```text
PDF labels ──► ingest ──► clean ──► chunk ──► embed (BGE) ──► ChromaDB
                                                                  │
User question ──► drug router ──► retrieve Top-K ──► prompt builder
                                                                  │
                                              Ollama (Qwen2.5 7B) ◄┘
                                                                  │
                                              citation check ──► Gradio UI
```

| Stage | Role |
|------|------|
| Ingest | Extract page-level text from PDFs with PyMuPDF |
| Clean | Conservative, page-preserving text normalization |
| Chunk | Character chunking that never crosses page boundaries |
| Embed | `BAAI/bge-small-en-v1.5` embeddings |
| Index | Persistent ChromaDB vectors (~7,194 chunks) |
| Retrieve | Source-scoped Top-K semantic search |
| Generate | Local Ollama chat with streaming + citation checks |
| Evaluate | Labeled IR metrics for assignment reporting |

---

## Stack

| Component | Choice |
|----------|--------|
| Language | Python 3.11+ |
| PDF extraction | PyMuPDF |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector store | ChromaDB |
| LLM runtime | Ollama |
| Model | `qwen2.5:7b-instruct` |
| UI | Gradio 6 |

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/drug-information-assistant.git
cd drug-information-assistant

python -m venv .venv
# Windows (Git Bash)
source .venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-step10.txt
# Also install remaining pipeline deps used by Steps 2–11
# (chromadb, sentence-transformers, etc.) as required by your environment.
```

### 2. Install and start Ollama

1. Install Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Pull the model:

```bash
ollama pull qwen2.5:7b-instruct
```

3. Serve locally (recommended for controlled GPU settings):

```bash
export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_NUM_PARALLEL=1
ollama serve
```

On 4 GB GPUs (for example Quadro T2000), keep flash attention off and free system RAM before loading the 7B model.

### 3. Launch the assistant

With the corpus already indexed under `data/`:

```bash
source .venv/Scripts/activate
python -m scripts.app
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860).

The app warms Ollama on startup, then opens a ChatGPT-style local interface.

---

## Example questions

- What safety warnings are listed for ibuprofen?
- What is the acetaminophen liver warning?
- What is ampicillin used for?
- What are the side effects of fluoxetine?

Name a drug from the local catalog (or pick its PDF in **Settings**) so retrieval stays source-scoped.

---

## Rebuild the corpus (optional)

If you need to regenerate pipeline artifacts from PDFs:

```bash
python -m scripts.ingest
python -m scripts.clean
python -m scripts.chunk
python -m scripts.embed
python -m scripts.index_vectors
```

Useful utilities:

```bash
python -m scripts.retrieve --question "What warnings are listed for ibuprofen?"
python -m scripts.build_prompt --retrieval-json path/to/retrieval.json
python -m scripts.generate_answer --prompt-json path/to/prompt.json
python -m scripts.evaluate_ir
```

---

## Project layout

```text
drug-information-assistant/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-step10.txt
├── scripts/                 # CLI entry points
│   ├── app.py               # Gradio chat UI
│   ├── ingest.py
│   ├── clean.py
│   ├── chunk.py
│   ├── embed.py
│   ├── index_vectors.py
│   ├── retrieve.py
│   ├── build_prompt.py
│   ├── generate_answer.py
│   └── evaluate_ir.py
├── src/                     # Library modules
│   ├── loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectordb.py
│   ├── drug_router.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── ollama_generator.py
│   ├── ollama_stream.py
│   ├── rag_pipeline.py
│   ├── chat_ui.py
│   └── ir_evaluator.py
├── tests/                   # Offline unit tests
└── data/
    ├── documents/           # Source PDFs
    ├── sources.csv          # PDF registry + DailyMed URLs
    ├── target_drugs.txt     # Target drug list
    ├── cleaned/             # Cleaned page JSON
    ├── chunks/              # Chunk JSON
    ├── embeddings/          # Per-document embedding artifacts
    ├── chroma/              # Local ChromaDB (gitignored)
    └── evaluation/          # IR gold labels and metrics
```

---

## Testing

```bash
source .venv/Scripts/activate
python -m unittest discover -s tests -v
```

Tests are offline by default (fake Ollama / fake retrieval). No GPU is required to run the suite.

---

## Design principles

1. **Local-only inference** — no cloud LLM calls
2. **Source grounding** — one drug label per answer unless explicitly allowed
3. **No silent truncation** of medical passages in prompts
4. **Fail closed** on missing citations, unknown IDs, mixed abstentions, or incomplete generations
5. **Citation ≠ verification** — IDs locate passages; they do not prove medical entailment

---

## Hardware notes

| Resource | Recommendation |
|---------|----------------|
| RAM | 16 GB+ recommended for Qwen2.5 7B + embeddings |
| GPU | Optional; 4 GB VRAM works with partial offload |
| Disk | Space for PDFs, embeddings, and ChromaDB |

If Ollama returns out-of-memory errors, free RAM, keep a single model loaded, and retry after the app warmup completes.

---

## License

See [`LICENSE`](LICENSE).

---

## Acknowledgments

- [DailyMed](https://dailymed.nlm.nih.gov/) for publicly available drug labeling
- [Ollama](https://ollama.com/) for local model serving
- [Qwen](https://huggingface.co/Qwen) for Qwen2.5 Instruct
- [BAAI](https://huggingface.co/BAAI) for `bge-small-en-v1.5`
