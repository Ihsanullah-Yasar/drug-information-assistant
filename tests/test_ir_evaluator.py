from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from src.ir_evaluator import (
    Question, load_gold, load_questions, query_both_modes, score_one,
    score_rankings, verify_gold_provenance, write_csv_atomic, write_json_atomic,
)

A = 'a' * 64
B = 'b' * 64
C = 'c' * 64
D = 'd' * 64


class EvalMetricsTests(unittest.TestCase):
    def test_first_rank_gives_one(self):
        self.assertEqual(score_one([A, B], {A})['reciprocal_rank'], 1.0)

    def test_rank_three_gives_one_third(self):
        self.assertAlmostEqual(score_one([A, B, C], {C})['reciprocal_rank'], 1 / 3)

    def test_miss_is_zero(self):
        record = score_one([A, B], {C})
        self.assertEqual(record['hit'], 0)
        self.assertEqual(record['reciprocal_rank'], 0)
        self.assertIsNone(record['first_relevant_rank'])

    def test_k_cutoff(self):
        self.assertEqual(score_one([A, B, C], {C}, k=2)['hit'], 0)
        self.assertEqual(score_one([A, B, C], {C}, k=3)['hit'], 1)

    def test_multiple_relevant_first_match(self):
        self.assertAlmostEqual(score_one([A, B, C], {B, C})['reciprocal_rank'], 0.5)

    def test_duplicate_results_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            score_one([A, A], {A})

    def test_empty_gold_rejected(self):
        with self.assertRaises(ValueError):
            score_one([A], set())

    def test_aggregate_modes_and_no_leakage(self):
        qs = [Question('q1', 'question a', 'a.pdf'), Question('q2', 'question b', 'b.pdf')]
        gold = {'q1': {B}, 'q2': {D}}
        runs = {
            'global': {'q1': [A, B], 'q2': [A]},
            'drug_aware': {'q1': [B], 'q2': [D]},
        }
        report = score_rankings(qs, gold, runs)
        self.assertEqual(report['modes']['global']['hit_rate_at_k'], 0.5)
        self.assertEqual(report['modes']['global']['mrr_at_k'], 0.25)
        self.assertEqual(report['modes']['drug_aware']['hit_rate_at_k'], 1)
        self.assertEqual(report['modes']['drug_aware']['mrr_at_k'], 1)
        self.assertEqual(report['question_count'], 2)

    def test_missing_question_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            score_rankings([Question('q1', 'q', 'a.pdf')], {'q1': {A}}, {'global': {}})


class EvalFileTests(unittest.TestCase):
    def test_questions_and_gold_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            with (path / 'q.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['question_id', 'question', 'expected_filename'])
                w.writerow(['q_1', 'What is this?', 'drug.pdf'])
            with (path / 'g.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['question_id', 'chunk_id'])
                w.writerow(['q_1', A])
            qs = load_questions(path / 'q.csv')
            self.assertEqual(qs[0].question_id, 'q_1')
            self.assertEqual(load_gold(path / 'g.csv', qs), {'q_1': {A}})

    def test_gold_without_annotations_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'g.csv'
            dest.write_text('question_id,chunk_id\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'MANUALLY VERIFIED'):
                load_gold(dest, [Question('q1', 'hi', 'a.pdf')])

    def test_gold_bad_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'g.csv'
            dest.write_text('question_id,chunk_id\nq1,wrong\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_gold(dest, [Question('q1', 'hi', 'a.pdf')])

    def test_bad_pdf_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'q.csv'
            dest.write_text('question_id,question,expected_filename\nq1,hi,../a.pdf\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'filename'):
                load_questions(dest)

    def test_atomic_csv_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            csvp, jsonp = Path(d) / 'report.csv', Path(d) / 'report.json'
            write_csv_atomic(csvp, [{'a': 'hello, world', 'b': 'text'}], ['a', 'b'])
            write_json_atomic(jsonp, {'value': 123})
            self.assertIn('hello, world', csvp.read_text('utf-8-sig'))
            self.assertIn('123', jsonp.read_text('utf-8'))


class EvalRetrievalTests(unittest.TestCase):
    @staticmethod
    def fixtures():
        qs = [Question('q1', 'How does ibuprofen work?', 'ibuprofen.pdf')]
        collection = Mock()
        collection.count.return_value = 7194
        router = Mock(return_value='ibuprofen.pdf')
        def search(question, *, collection, embedder, top_k, filename, expected_count):
            ids = [A, B] if filename is None else [B, A]
            return [{'rank': i, 'chunk_id': x, 'filename': filename or 'wrong.pdf',
                     'page_number': i, 'chunk_index': 1, 'text': 'some passage'}
                    for i, x in enumerate(ids, start=1)]
        return qs, collection, router, search

    def test_both_modes_use_same_queries_without_gold_leakage(self):
        qs, col, router, search = self.fixtures()
        runs, pool = query_both_modes(qs, collection=col, embedder=object(),
            project_root=Path('/tmp/project'), top_k=5, expected_count=7194,
            retriever=search, router=router)
        self.assertEqual(runs['global']['q1'], [A, B])
        self.assertEqual(runs['drug_aware']['q1'], [B, A])
        self.assertEqual(len(pool), 4)
        self.assertTrue(all(row['review_relevant_1_or_0'] == '' for row in pool))
        router.assert_called_once()

    def test_router_mismatch_rejected(self):
        qs, col, _, search = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'Routing mismatch'):
            query_both_modes(qs, collection=col, embedder=object(),
                project_root=Path('/tmp'), retriever=search,
                router=Mock(return_value='not-ibuprofen.pdf'))

    def test_collection_count_rejected(self):
        qs, col, router, search = self.fixtures()
        col.count.return_value = 32
        with self.assertRaisesRegex(ValueError, 'Index count mismatch'):
            query_both_modes(qs, collection=col, embedder=object(),
                project_root=Path('/tmp'), retriever=search, router=router)

    def test_provenance_accepted(self):
        col = Mock()
        col.get.return_value = {'ids': [A], 'metadatas': [{'filename': 'a.pdf'}]}
        verify_gold_provenance(col, [Question('q1', 'q', 'a.pdf')], {'q1': {A}})

    def test_provenance_missing_rejected(self):
        col = Mock()
        col.get.return_value = {'ids': [], 'metadatas': []}
        with self.assertRaisesRegex(ValueError, 'not present'):
            verify_gold_provenance(col, [Question('q1', 'q', 'a.pdf')], {'q1': {A}})

    def test_provenance_cross_drug_rejected(self):
        col = Mock()
        col.get.return_value = {'ids': [A], 'metadatas': [{'filename': 'other.pdf'}]}
        with self.assertRaisesRegex(ValueError, 'not from'):
            verify_gold_provenance(col, [Question('q1', 'q', 'a.pdf')], {'q1': {A}})


if __name__ == '__main__':
    unittest.main()
