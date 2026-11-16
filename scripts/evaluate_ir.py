"""Step 11: evaluate manually labeled retrieval in both unfiltered and drug-aware modes.

Examples:
    python -m scripts.evaluate_ir --prepare --top-k 10
    python -m scripts.evaluate_ir --top-k 5

No Ollama calls and no index mutation. Reads pre-existing Chroma vectors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.ir_evaluator import (
    load_gold, load_questions, query_both_modes, score_rankings, verify_gold_provenance,
    write_csv_atomic, write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
POOL_FIELDS = [
    "question_id", "question", "mode", "rank", "chunk_id", "filename",
    "page_number", "chunk_index", "cosine_similarity", "full_passage",
    "review_relevant_1_or_0",
]
RESULT_FIELDS = [
    "mode", "question_id", "question", "expected_filename", "hit",
    "reciprocal_rank", "first_relevant_rank", "retrieved_count", "gold_count",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true",
                        help="Export unlabeled ranked passages for human review; do NOT calculate scores")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/evaluation/questions.csv")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/evaluation/gold.csv")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Ranked results per mode; 5 for official evaluation; 10-20 for candidate review")
    parser.add_argument("--require-count", type=int, default=7194)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--pool-out", type=Path, default=ROOT / "data/evaluation/candidate_pool.csv")
    parser.add_argument("--json-out", type=Path, default=ROOT / "data/evaluation/metrics.json")
    parser.add_argument("--csv-out", type=Path, default=ROOT / "data/evaluation/metrics_by_question.csv")
    args = parser.parse_args()
    try:
        if not 1 <= args.top_k <= 50:
            raise ValueError("--top-k must be between 1 and 50")
        if not args.prepare and args.top_k != 5:
            raise ValueError("Final Hit Rate@5 / MRR@5 report requires --top-k 5")
        questions = load_questions(args.questions)
        gold = None if args.prepare else load_gold(args.gold, questions)
        # Labels are validated BEFORE any expensive BGE or Chroma access.
        from src.embedder import BGEEmbedder
        from src.vectordb import connect_collection
        collection = connect_collection(ROOT / "data/chroma", create=False)
        if gold is not None:
            verify_gold_provenance(collection, questions, gold)
        embedder = BGEEmbedder(device=args.device)
        rankings, candidates = query_both_modes(
            questions, collection=collection, embedder=embedder,
            project_root=ROOT, top_k=args.top_k, expected_count=args.require_count,
        )
        if args.prepare:
            write_csv_atomic(args.pool_out, candidates, POOL_FIELDS)
            print(f"Candidate pool exported: {args.pool_out} ({len(candidates)} rows)")
            print("These are UNLABELED results. Human-review the full passages and PDF pages.")
            print("Put independently verified relevant SHA256 chunk IDs in data/evaluation/gold.csv.")
            print("Nothing was scored; do not claim retrieval accuracy from this pool.")
            return 0
        assert gold is not None
        report = score_rankings(questions, gold, rankings, k=5)
        report["notes"] = (
            "Chunk-level metrics; manually judged relevance only. Unfiltered and drug-aware "
            "modes evaluated on identical questions and gold IDs. The latter is explicitly "
            "restricted to the named medicine, and is not comparable to unrestricted retrieval "
            "without describing this filter. This does not test answer factuality."
        )
        write_json_atomic(args.json_out, report)
        rows = []
        for mode, metrics in report["modes"].items():
            print(f"{mode:>12}: Hit Rate@5 = {metrics['hit_rate_at_k']:.3f} "
                  f"| MRR@5 = {metrics['mrr_at_k']:.3f} "
                  f"| Queries = {metrics['evaluated_queries']}")
            rows.extend({"mode": mode, **row} for row in metrics["rows"])
        write_csv_atomic(args.csv_out, rows, RESULT_FIELDS)
        print(f"Saved detailed metrics: {args.json_out}; {args.csv_out}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"IR evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
