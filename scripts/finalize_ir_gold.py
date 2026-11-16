"""Convert manually reviewed Step 11 candidate pool (1 relevant, 0 irrelevant) to gold.csv.

Example:
    python -m scripts.finalize_ir_gold --reviewed-pool data/evaluation/candidate_pool.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from src.ir_evaluator import (
    gold_from_reviewed_candidates, load_questions, write_csv_atomic,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-pool", type=Path,
                        default=ROOT / "data/evaluation/candidate_pool.csv")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data/evaluation/questions.csv")
    parser.add_argument("--gold-out", type=Path,
                        default=ROOT / "data/evaluation/gold.csv")
    args = parser.parse_args()
    try:
        questions = load_questions(args.questions)
        rows = gold_from_reviewed_candidates(args.reviewed_pool, questions)
        write_csv_atomic(args.gold_out, rows, ["question_id", "chunk_id"])
        print(f"Wrote {len(rows)} manually judged relevant chunk IDs: {args.gold_out}")
        print("Important: pool-based judgments can bias evaluation if the pool omits other relevant chunks.")
        print("For rigorous results, independently check the original label and add omitted gold chunks.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Could not finalize gold labels: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
