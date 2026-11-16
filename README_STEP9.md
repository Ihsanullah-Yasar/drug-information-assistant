# Step 9 — Local LLM Answer Generation with Ollama

**Project:** Drug Information Assistant, a manual local RAG project for an Information Retrieval course.

**Starting point already verified on your Windows PC:** 100 DailyMed PDF labels, 7,194 BGE passage embeddings, the complete ChromaDB index, and a **Step 8 ibuprofen prompt** at `data/prompts/ibuprofen_prompt.json` with `status=ready`, 5 ibuprofen source passages, and 2 chat messages. Step 8 suite passed 125 tests.

## 1. What this step adds

- `src/ollama_generator.py` — local-only `/api/chat` request and output validation. Pure Python standard library, no LangChain, LlamaIndex or extra Ollama Python client.
- `scripts/generate_answer.py` — CLI that reads the **existing** Step 8 JSON, sends its two chat messages, and saves a citation-labeled answer.
- `tests/test_ollama_generator.py` — 23 **offline** tests using fake Ollama responses; no LLM download or running server required for tests.

This step **does not rerun retrieval**, **does not regenerate the 7,194 embeddings**, and **does not modify ChromaDB**.

## 2. Ollama installation (Windows)

1. Open https://ollama.com/download/windows and install Ollama if needed.
2. Open Git Bash in `D:/github/drug-information-assistant`, activate `.venv` as before.
3. Confirm the CLI and local Ollama service:

   ```bash
   ollama --version
   ollama list
   curl.exe http://127.0.0.1:11434/api/tags
   ```

   On Windows, the Ollama desktop application normally runs the local server. If it isn't running, start the application; if you are deliberately using the CLI server without the application, launch `ollama serve` in a separate terminal. Avoid running a duplicate server on the same port.

4. Install the requested **local language model** if `ollama list` doesn't show it:

   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama list
   ```

   The model is about 4.7 GB to download (quantized variant); make sure there is enough disk and RAM. Downloads require Internet once, but answer generation uses the **local Ollama endpoint** after the model is installed.

## 3. Extract files (do not overwrite existing code)

Copy the ZIP contents into the project root. Paths must be:

```
src/ollama_generator.py
scripts/generate_answer.py
tests/test_ollama_generator.py
README_STEP9.md
```

**Do not delete/replace** `data/embeddings/`, `data/chroma/`, `src/prompt_builder.py`, `scripts/build_prompt.py`, or the saved prompt JSON.

## 4. Commands to run in Git Bash

**Python syntax:**

```bash
python -m py_compile src/ollama_generator.py scripts/generate_answer.py
```

**Offline unit tests:**

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected **148 tests, OK** if your previous 125 tests are unchanged. The build environment ran 148 tests with 2 Chroma integration tests skipped because ChromaDB was not installed there; those tests previously passed on your own Windows machine. These tests **do not confirm real Ollama inference**.

**One real ibuprofen question from the already-built Step 8 prompt:**

```bash
python -m scripts.generate_answer \
  --prompt-json data/prompts/ibuprofen_prompt.json \
  --model qwen2.5:7b-instruct \
  --num-ctx 16384 \
  --max-output-tokens 512 \
  --json-out data/answers/ibuprofen_answer.json \
  --text-out data/answers/ibuprofen_answer.txt
```

The model may give a grounded answer with `[S1]`-style citations, or reply with the exact refusal sentence. The **actual answer depends on your local model**; no answer text can be predicted or guaranteed in advance.

**Inspect result without a second generation:**

```bash
python -c "import json; p=json.load(open('data/answers/ibuprofen_answer.json',encoding='utf-8')); print('Status:',p['status'],'| Citations:',p['citations'],'| Prompt tokens:',p['usage']['prompt_eval_count'])"
```

If `status=answered`, inspect the answer *against the original PDF text*, especially each claimed drug side effect, safety warning, and dosage. A model may attach the correct citation ID to a fact the source does not support.

## 5. Technical explanation (for your seminar/report)

The Step 8 prompt is an ordered array of **two chat messages**:

1. `system`: constrained medical-document answering, citations, source separation, and abstention instructions.
2. `user`: question and five **complete** retrieved ibuprofen passages, each JSON-labeled `[S1]` ... `[S5]`, with page numbers, filenames and DailyMed links.

We call the native Ollama API:

```
POST http://127.0.0.1:11434/api/chat
Content-Type: application/json
{
  "model": "qwen2.5:7b-instruct",
  "messages": ["<the two existing objects from Step 8>"],
  "stream": false,
  "options": {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 16384,
    "num_predict": 512
  }
}
```

`messages` in the actual request is an array of objects, *not* the illustrative placeholder string above. We do not use Ollama's default embedding functions, and we do not ask it to search the Internet.

### Context-limit handling

A text character count is **not a token count**. For the chosen Qwen2.5 model, the code conservatively checks `UTF-8 prompt bytes + 256 chat-template reserve + max_output_tokens <= num_ctx` before making the request. This deliberately overestimates many normal prompts; it avoids pretending our text has a known, exact Ollama token count. The returned `prompt_eval_count` is recorded for inspection. This preflight applies to the **Qwen2.5 byte-level tokenizer**, not an arbitrary tokenizer or custom-model implementation.

The default `num_ctx=16384` can use significant memory. If you see a context-check error, rebuild Step 8 with fewer complete passages (`--max-sources 3`) or use a larger context only if your machine has enough memory. Never silently truncate a medical passage or assume its citation is valid after truncation.

### Citation checks and abstention

- `answered`: at least one valid supplied citation ID and normal `done_reason=stop`.
- `abstained`: exact fallback from the model: `I don't have enough information in the provided documents.`
- `rejected_*`: the model returned no citations, fabricated a citation ID, mixed an answer and refusal, or stopped due to length. In these cases the user-visible answer is the abstention sentence.
- `skipped_*`: the saved Step 8 prompt was not `ready`; **the Ollama server is not called**.

**Important limitation:** the program checks the *syntax/identity of citation references*. It **cannot verify semantic entailment**, i.e., that a cited passage truly supports each medical claim. This requires additional evidence-based evaluation, human checking, and/or more sophisticated verification. The output is for educational use, not individualized medical advice or clinical decision-making.

The program saves `answer`, `status`, cited PDF metadata, local model name, prompt hash, and Ollama token usage. It does not save unsafe rejected raw model output.

## 6. Troubleshooting

**`ollama: command not found`** — Install Ollama for Windows, reopen the terminal, ensure it is in PATH, and check `ollama --version`.

**`Cannot reach local Ollama ...11434`** — Start the Windows Ollama app or the standalone server, then check `curl.exe http://127.0.0.1:11434/api/tags`.

**`Ollama HTTP 404`** — `ollama list`; the exact model tag has not been downloaded, or `--model` is spelled incorrectly. Run `ollama pull qwen2.5:7b-instruct`.

**`Conservative context check failed`** — Your prompt plus reserved output exceeds the chosen `num_ctx` according to the conservative preflight. Use fewer complete Step 8 passages, or increase `--num-ctx` with sufficient RAM.

**`rejected_uncited` or `rejected_unknown_citation`** — Your local model did not obey citation instructions. The program intentionally abstains. Do not disable validation or invent citations.

**Generation too long / memory pressure** — Increase `--timeout` if needed (up to 3600 seconds). If hardware cannot run 7B, a smaller local model could be tested via `--model`, but the assignment's model specification should take priority.

## 7. Next step

After your **actual** Ollama response passes inspection, implement **Step 10: Gradio UI** to connect Step 7 retrieval → Step 8 prompt building → Step 9 local generation. Later add evaluation with Hit Rate@5, MRR and citation-quality assessments, along with your assignment's README and report.
