"""Step 10: local, source-scoped RAG orchestration for the Gradio interface.

Reuses Steps 5-9 without re-embedding corpus passages, mutating ChromaDB,
calling cloud LLMs, or relying on LangChain/LlamaIndex/Haystack.

The generated answer is not medical advice. Citation-ID validation is structural,
NOT a proof that each medical claim follows from the cited source.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.drug_router import resolve_drug_filename
from src.embedder import BGEEmbedder
from src.ollama_generator import (
    DEFAULT_MODEL,
    TEMPLATE_OVERHEAD,
    generate_grounded_answer,
)
from src.prompt_builder import ABSTENTION, build_prompt_package
from src.retriever import retrieve_chunks
from src.vectordb import connect_collection

DEFAULT_TOP_K = 5
DEFAULT_SOURCE_LIMIT = 3
DEFAULT_EXPECTED_COUNT = 7194
DEFAULT_NUM_CTX = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 512
MAX_QUESTION_CHARS = 1000


def available_pdf_labels(sources_csv: Path) -> list[str]:
    """Load the exact local PDF labels for a dropdown; reject invalid filenames."""
    source_file = Path(sources_csv)
    with source_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError("data/sources.csv must contain a filename column")
        files = []
        for item in reader:
            value = item.get("filename")
            if (not isinstance(value, str) or not value.endswith(".pdf")
                    or value != Path(value).name or "\\" in value
                    or not value or value.strip() != value):
                raise ValueError("Invalid filename in data/sources.csv")
            files.append(value)
    if not files or len(files) != len(set(files)):
        raise ValueError("data/sources.csv must list unique drug-label PDF files")
    return sorted(files, key=str.casefold)


def _simple_result(question: str, status: str, reason: str, *, filename: str | None = None,
                   retrieved_count: int = 0, included_count: int = 0,
                   sources: list[dict] | None = None) -> dict:
    return {
        "question": question,
        "answer": ABSTENTION,
        "status": status,
        "reason": reason,
        "filename": filename,
        "retrieved_count": retrieved_count,
        "included_count": included_count,
        "citations": [],
        "sources": sources or [],
        "usage": {"prompt_eval_count": None, "eval_count": None},
        "note": "Citation references are not independent medical-claim verification.",
    }


def _evidence_rows(package: dict, matches: list[dict], cited_ids: list[str]) -> list[dict]:
    by_chunk_id = {match["chunk_id"]: match for match in matches}
    cited = set(cited_ids)
    rows = []
    for source in package.get("sources", []):
        match = by_chunk_id[source["chunk_id"]]
        # Plain strings only. Gradio displays these in a non-editable text table.
        excerpt = " ".join(match["text"].split())
        rows.append({
            "citation": f"[{source['source_id']}]",
            "filename": source["filename"],
            "page": source["page_number"],
            "chunk": source["chunk_index"],
            "excerpt": excerpt[:320] + (" ..." if len(excerpt) > 320 else ""),
            "source_url": source["source_url"],
            "used_in_answer": source["source_id"] in cited,
        })
    return rows


class LocalDrugRAG:
    """One local UI process can reuse the same model and collection each question.

    External dependencies can be injected for deterministic, offline unit tests.
    The default corpus count is the user's verified 7,194 Chroma vectors.
    """

    def __init__(self, project_root: Path, *, collection: Any = None,
                 embedder: Any = None, expected_count: int = DEFAULT_EXPECTED_COUNT,
                 model: str = DEFAULT_MODEL, num_ctx: int = DEFAULT_NUM_CTX,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                 device: str = "cpu", generator=None):
        self.project_root = Path(project_root)
        if type(expected_count) is not int or expected_count < 1:
            raise ValueError("expected_count must be a positive integer")
        if type(num_ctx) is not int or not 1024 <= num_ctx <= 131072:
            raise ValueError("num_ctx out of range")
        if type(max_output_tokens) is not int or not 16 <= max_output_tokens <= 8192:
            raise ValueError("max_output_tokens out of range")
        if max_output_tokens + TEMPLATE_OVERHEAD >= num_ctx:
            raise ValueError("num_ctx is too small for the output reserve")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        self.collection = collection
        self.embedder = embedder
        self.expected_count = expected_count
        self.model = model
        self.num_ctx = num_ctx
        self.max_output_tokens = max_output_tokens
        self.device = device
        self.generator = generator or generate_grounded_answer
        self.sources_csv = self.project_root / "data/sources.csv"
        self.targets_txt = self.project_root / "data/target_drugs.txt"
        self.db_dir = self.project_root / "data/chroma"

    def _get_collection(self):
        if self.collection is None:
            self.collection = connect_collection(self.db_dir, create=False)
        return self.collection

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = BGEEmbedder(device=self.device)
        return self.embedder

    def ask(self, question: str, *, filename: str | None = None,
            max_sources: int = DEFAULT_SOURCE_LIMIT) -> dict:
        """Answer one question grounded in one local drug-label PDF.

        Must not silently mix labels. Larger retrieved chunks remain intact; if
        needed, source count decreases until the conservative context check fits.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please enter a question")
        question = question.strip()
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question must be at most {MAX_QUESTION_CHARS} characters")
        if type(max_sources) is not int or not 1 <= max_sources <= 5:
            raise ValueError("max_sources must be between 1 and 5")
        if filename is not None:
            if filename not in available_pdf_labels(self.sources_csv):
                raise ValueError("Choose a PDF label from the local source catalog")
        selected = resolve_drug_filename(
            question, sources_csv=self.sources_csv, targets_txt=self.targets_txt,
            filename=filename, all_drugs=False,
        )
        if selected is None:
            return _simple_result(
                question, "needs_drug_scope",
                "Name a drug from your catalog in the question, or explicitly select a source PDF. "
                "The interface does not combine different medicines' evidence.",
            )

        collection = self._get_collection()
        actual_count = collection.count()
        if actual_count != self.expected_count:
            raise ValueError(
                f"The local index has {actual_count} vectors, expected {self.expected_count}. "
                "Run the Step 6 full index audit."
            )
        matches = retrieve_chunks(
            question, collection=collection, embedder=self._get_embedder(),
            top_k=DEFAULT_TOP_K, filename=selected,
            expected_count=self.expected_count,
        )
        payload = {"question": question, "top_k": DEFAULT_TOP_K,
                   "filename_filter": selected, "matches": matches}

        # Build full, source-bounded prompts; never cut medical text mid-passage.
        selected_package = None
        last_package = None
        for number in range(min(max_sources, len(matches)), 0, -1):
            package = build_prompt_package(payload, max_sources=number)
            last_package = package
            if package["status"] != "ready":
                break
            conservative = sum(
                len(item["content"].encode("utf-8")) for item in package["messages"]
            ) + TEMPLATE_OVERHEAD + self.max_output_tokens
            if conservative <= self.num_ctx:
                selected_package = package
                break

        if selected_package is None:
            if not matches:
                reason = "No relevant chunks were retrieved from the selected PDF."
                status = "no_evidence"
            elif last_package is not None and last_package["status"] != "ready":
                reason = last_package["note"]
                status = last_package["status"]
            else:
                reason = (
                    "Even one complete evidence passage exceeds the conservative context budget. "
                    "No text was truncated. Increase the context budget only if RAM allows."
                )
                status = "context_budget_exceeded"
            return _simple_result(
                question, status, reason, filename=selected, retrieved_count=len(matches)
            )

        result = self.generator(
            selected_package, model=self.model, num_ctx=self.num_ctx,
            max_output_tokens=self.max_output_tokens,
        )
        if not isinstance(result, dict):
            raise RuntimeError("The Step 9 generator returned an invalid response")
        citations = result.get("citations")
        if not isinstance(citations, list) or any(not isinstance(c, str) for c in citations):
            raise RuntimeError("Invalid generated citation manifest")
        source_ids = {s["source_id"] for s in selected_package["sources"]}
        if not set(citations).issubset(source_ids):
            raise RuntimeError("The generator included citation IDs outside the evidence set")
        status, answer = result.get("status"), result.get("answer")
        if not isinstance(status, str) or not isinstance(answer, str):
            raise RuntimeError("Missing generator answer or status")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("Missing generator usage")
        return {
            "question": question,
            "answer": answer,
            "status": status,
            "reason": str(result.get("reason", "")),
            "filename": selected,
            "retrieved_count": len(matches),
            "included_count": selected_package["source_count"],
            "citations": citations,
            "sources": _evidence_rows(selected_package, matches, citations),
            "usage": usage,
            "note": "Citation IDs checked only; factual medical support needs source review.",
        }
