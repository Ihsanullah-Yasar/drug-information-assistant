"""Step 11: reproducible labeled chunk-level information-retrieval evaluation.

Hit Rate@K and MRR@K measure ranked retrieval against manually reviewed
relevant CHUNK IDs. They do not evaluate LLM factuality or safety.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

SHA256 = re.compile(r"^[0-9a-f]{64}$")
QID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    expected_filename: str


def load_questions(path: Path) -> list[Question]:
    """Validate question file; expected_filename is an audit label, NOT a global filter."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"question_id", "question", "expected_filename"}.issubset(reader.fieldnames):
            raise ValueError("Questions CSV needs question_id, question, expected_filename columns")
        rows: list[Question] = []
        seen = set()
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            question = (row.get("question") or "").strip()
            filename = (row.get("expected_filename") or "").strip()
            if not QID.fullmatch(qid) or qid in seen:
                raise ValueError(f"Invalid or repeated question ID: {qid!r}")
            if not question or len(question) > 1000:
                raise ValueError(f"Missing/oversized question for {qid}")
            if (not filename.endswith(".pdf") or filename != Path(filename).name
                    or "/" in filename or "\\" in filename or filename.startswith(".")):
                raise ValueError(f"Invalid expected filename for {qid}: {filename!r}")
            rows.append(Question(qid, question, filename))
            seen.add(qid)
    if not rows:
        raise ValueError("Questions CSV contains no questions")
    return rows


def load_gold(path: Path, questions: list[Question]) -> dict[str, set[str]]:
    """Independent gold relevance judgments: ≥1 verified chunk ID per question."""
    expected = {q.question_id for q in questions}
    gold = {qid: set() for qid in expected}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"question_id", "chunk_id"}.issubset(reader.fieldnames):
            raise ValueError("Gold CSV must have question_id,chunk_id columns")
        for row in reader:
            qid, chunk = (row.get("question_id") or "").strip(), (row.get("chunk_id") or "").strip()
            if qid not in gold:
                raise ValueError(f"Gold references nonexistent question {qid!r}")
            if not SHA256.fullmatch(chunk):
                raise ValueError(f"Gold chunk_id for {qid} must be a 64-digit lowercase SHA256 ID")
            if chunk in gold[qid]:
                raise ValueError(f"Repeated gold chunk_id for {qid}")
            gold[qid].add(chunk)
    missing = sorted(qid for qid, ids in gold.items() if not ids)
    if missing:
        raise ValueError(
            "Missing MANUALLY VERIFIED relevant chunk IDs for: " + ", ".join(missing)
            + ". Do not calculate Hit Rate/MRR from guessed labels."
        )
    return gold



def verify_gold_provenance(collection: Any, questions: list[Question], gold: dict[str, set[str]]) -> None:
    """Verify that labeled chunk IDs really exist in the declared source PDF.

    This proves chunk identity and provenance, NOT semantic relevance. People
    must still read and independently judge each page/passsage.
    """
    expected = {q.question_id: q.expected_filename for q in questions}
    if set(gold) != set(expected):
        raise ValueError("Gold IDs and questions do not match")
    ids = sorted({chunk for values in gold.values() for chunk in values})
    if not ids:
        raise ValueError("No gold chunks")
    result = collection.get(ids=ids, include=["metadatas"])
    found_ids = result.get("ids") if isinstance(result, dict) else None
    found_md = result.get("metadatas") if isinstance(result, dict) else None
    if (not isinstance(found_ids, list) or not isinstance(found_md, list)
            or len(found_ids) != len(found_md)):
        raise ValueError("Cannot verify gold provenance from Chroma metadata")
    lookup = dict(zip(found_ids, found_md))
    if len(lookup) != len(ids) or set(lookup) != set(ids):
        missing = sorted(set(ids) - set(lookup))
        raise ValueError(f"Gold chunk IDs not present in Chroma: {missing[:5]}")
    for qid, values in gold.items():
        for chunk_id in values:
            md = lookup[chunk_id]
            if not isinstance(md, dict) or md.get("filename") != expected[qid]:
                raise ValueError(
                    f"Gold chunk {chunk_id} for {qid} is not from "
                    f"the expected PDF {expected[qid]}"
                )


def score_one(retrieved: list[str], gold: set[str], *, k: int = 5) -> dict[str, Any]:
    """One ranked result list, zero-based internal index and 1-based reported rank."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not gold or any(not isinstance(x, str) or not x for x in gold):
        raise ValueError("gold must be a nonempty set of chunk IDs")
    if not isinstance(retrieved, list) or any(not isinstance(x, str) or not x for x in retrieved):
        raise ValueError("retrieved must be a list of chunk ID strings")
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("retrieved results contain duplicate chunk IDs")
    first = next((i for i, cid in enumerate(retrieved[:k], start=1) if cid in gold), None)
    return {"hit": int(first is not None), "reciprocal_rank": (1.0 / first if first else 0.0),
            "first_relevant_rank": first, "retrieved_count": min(len(retrieved), k),
            "gold_count": len(gold)}


def score_rankings(questions: list[Question], gold: dict[str, set[str]],
                   runs: dict[str, dict[str, list[str]]], *, k: int = 5) -> dict[str, Any]:
    """Runs: mode -> question_id -> ranked chunk IDs. No missing/skipped query."""
    if not questions or not runs:
        raise ValueError("Need both questions and retrieval runs")
    qids = [q.question_id for q in questions]
    if len(qids) != len(set(qids)) or set(gold) != set(qids):
        raise ValueError("Question IDs and gold judgments are inconsistent")
    report: dict[str, Any] = {"k": k, "question_count": len(qids), "modes": {}}
    for mode, rankings in runs.items():
        if not mode or set(rankings) != set(qids):
            raise ValueError(f"Mode {mode!r} must contain every question exactly once")
        rows = []
        for question in questions:
            values = score_one(rankings[question.question_id], gold[question.question_id], k=k)
            rows.append({"question_id": question.question_id, "question": question.question,
                         "expected_filename": question.expected_filename, **values})
        report["modes"][mode] = {
            "hit_rate_at_k": sum(row["hit"] for row in rows) / len(rows),
            "mrr_at_k": sum(row["reciprocal_rank"] for row in rows) / len(rows),
            "evaluated_queries": len(rows),
            "rows": rows,
        }
    return report


def _validate_ranked_matches(matches: Any, *, top_k: int) -> list[dict]:
    if not isinstance(matches, list) or len(matches) > top_k:
        raise ValueError("Invalid retrieval result length")
    ids = []
    for i, item in enumerate(matches, start=1):
        if (not isinstance(item, dict) or item.get("rank") != i
                or not isinstance(item.get("chunk_id"), str)
                or not SHA256.fullmatch(item["chunk_id"])):
            raise ValueError("Invalid ranked Chroma result")
        ids.append(item["chunk_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ranked Chroma result")
    return matches


def query_both_modes(
    questions: list[Question], *,
    collection: Any, embedder: Any, project_root: Path,
    top_k: int = 5, expected_count: int = 7194,
    retriever: Callable | None = None, router: Callable | None = None,
) -> tuple[dict[str, dict[str, list[str]]], list[dict]]:
    """Reuse one local BGE model/Chroma collection for global and scoped queries.

    expected_filename only audits automatic routing; it never selects the filename.
    We never use expected gold chunk IDs in either retrieval call.
    """
    if type(top_k) is not int or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected_count must be positive")
    if collection.count() != expected_count:
        raise ValueError(f"Index count mismatch: expected {expected_count}, found {collection.count()}")
    if retriever is None:
        from src.retriever import retrieve_chunks
        retriever = retrieve_chunks
    if router is None:
        from src.drug_router import resolve_drug_filename
        router = resolve_drug_filename
    root = Path(project_root)
    runs: dict[str, dict[str, list[str]]] = {"global": {}, "drug_aware": {}}
    pool: list[dict] = []
    for question in questions:
        chosen = router(
            question.question, sources_csv=root / "data/sources.csv",
            targets_txt=root / "data/target_drugs.txt", filename=None, all_drugs=False,
        )
        if chosen != question.expected_filename:
            raise ValueError(
                f"Routing mismatch for {question.question_id}: {chosen!r} != "
                f"expected {question.expected_filename!r}. Fix your question/catalog before scoring."
            )
        for mode, filter_file in (("global", None), ("drug_aware", chosen)):
            matches = _validate_ranked_matches(retriever(
                question.question, collection=collection, embedder=embedder,
                top_k=top_k, filename=filter_file, expected_count=expected_count,
            ), top_k=top_k)
            runs[mode][question.question_id] = [row["chunk_id"] for row in matches]
            for match in matches:
                pool.append({
                    "question_id": question.question_id,
                    "question": question.question,
                    "mode": mode,
                    "rank": match["rank"],
                    "chunk_id": match["chunk_id"],
                    "filename": match.get("filename", ""),
                    "page_number": match.get("page_number", ""),
                    "chunk_index": match.get("chunk_index", ""),
                    "cosine_similarity": match.get("cosine_similarity", ""),
                    "full_passage": match.get("text", ""),
                    "review_relevant_1_or_0": "",  # intentionally NEVER auto-label
                })
    return runs, pool



def gold_from_reviewed_candidates(path: Path, questions: list[Question]) -> list[dict[str, str]]:
    """Convert FULLY manual-labeled candidate pool (0/1) into gold ID rows.

    Duplicate candidates (global+scoped) must have the SAME human judgment.
    Zero positives for any question is rejected instead of scoring fake zeros.
    """
    qids = {q.question_id for q in questions}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {"question_id", "chunk_id", "review_relevant_1_or_0"}
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError("Candidate pool is missing required annotation columns")
        judgments: dict[tuple[str, str], str] = {}
        num_rows = 0
        for row in reader:
            num_rows += 1
            qid = (row.get("question_id") or "").strip()
            chunk = (row.get("chunk_id") or "").strip()
            value = (row.get("review_relevant_1_or_0") or "").strip()
            if qid not in qids or not SHA256.fullmatch(chunk):
                raise ValueError(f"Invalid question/chunk ID in candidate row {num_rows}")
            if value not in ("0", "1"):
                raise ValueError(f"Candidate row {num_rows} not reviewed: enter exactly 0 or 1")
            key = qid, chunk
            if key in judgments and judgments[key] != value:
                raise ValueError(f"Conflicting relevance judgments for {qid}, {chunk}")
            judgments[key] = value
    if num_rows == 0:
        raise ValueError("No candidate rows to review")
    gold = [
        {"question_id": qid, "chunk_id": chunk}
        for (qid, chunk), value in sorted(judgments.items()) if value == "1"
    ]
    missing = sorted(qid for qid in qids if not any(row["question_id"] == qid for row in gold))
    if missing:
        raise ValueError(
            "No relevant chunks were labeled for " + ", ".join(missing)
            + ". Review the original PDF and add independently identified gold passages "
              "before running metrics; do not invent positives."
        )
    return gold


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    dest = Path(path)
    if dest.suffix.lower() != ".csv":
        raise ValueError("Output path must end in .csv")
    if not rows:
        raise ValueError("No rows to save")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=dest.parent,
                                         prefix="." + dest.stem + ".", suffix=".tmp", delete=False) as fp:
            tmp = Path(fp.name)
            writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, dest)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: dict) -> None:
    dest = Path(path)
    if dest.suffix.lower() != ".json":
        raise ValueError("JSON output path must end in .json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dest.parent,
                                         prefix="." + dest.stem + ".", suffix=".tmp", delete=False) as fp:
            tmp = Path(fp.name)
            json.dump(value, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, dest)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
