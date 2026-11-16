from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from src.ir_evaluator import Question, gold_from_reviewed_candidates

A = 'a'*64
B = 'b'*64
C = 'c'*64


def make_pool(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'question_id', 'chunk_id', 'review_relevant_1_or_0'])
        writer.writeheader()
        writer.writerows([{'question_id': q, 'chunk_id': c, 'review_relevant_1_or_0': v}
                          for q, c, v in rows])


class GoldReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name)/'candidates.csv'
        self.questions = [Question('q1', 'test?', 'drug.pdf')]

    def test_positive_and_negative_labels(self):
        make_pool(self.file, [('q1', A, '1'), ('q1', B, '0')])
        self.assertEqual(gold_from_reviewed_candidates(self.file, self.questions),
                         [{'question_id': 'q1', 'chunk_id': A}])

    def test_consistent_duplicate_global_scoped_is_deduplicated(self):
        make_pool(self.file, [('q1', A, '1'), ('q1', A, '1')])
        self.assertEqual(len(gold_from_reviewed_candidates(self.file, self.questions)), 1)

    def test_conflicting_labels_are_rejected(self):
        make_pool(self.file, [('q1', A, '1'), ('q1', A, '0')])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            gold_from_reviewed_candidates(self.file, self.questions)

    def test_unreviewed_row_rejected(self):
        make_pool(self.file, [('q1', A, '1'), ('q1', B, '')])
        with self.assertRaisesRegex(ValueError, 'not reviewed'):
            gold_from_reviewed_candidates(self.file, self.questions)

    def test_requires_verified_positive(self):
        make_pool(self.file, [('q1', A, '0')])
        with self.assertRaisesRegex(ValueError, 'No relevant chunks'):
            gold_from_reviewed_candidates(self.file, self.questions)

    def test_unknown_question_rejected(self):
        make_pool(self.file, [('q2', A, '1')])
        with self.assertRaisesRegex(ValueError, 'Invalid question'):
            gold_from_reviewed_candidates(self.file, self.questions)


if __name__ == '__main__':
    unittest.main()
