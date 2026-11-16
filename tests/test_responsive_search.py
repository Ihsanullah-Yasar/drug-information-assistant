"""Regression tests for bounded DailyMed search (no network needed)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'collect_labels.py'
spec = importlib.util.spec_from_file_location('collector_responsive', MODULE_PATH)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def candidate(n: int, title='AZITHROMYCIN TABLET') -> dict:
    return {'setid': str(UUID(int=n+1)), 'title': title}


class BoundedSearchTests(unittest.TestCase):
    def test_rejects_tablets_without_inspecting_hundreds(self):
        payload = json.dumps({'data': [candidate(n) for n in range(100)],
                              'metadata': {'total_pages': 100}}).encode()
        with patch.object(collector, 'fetch', return_value=payload) as fetch, \
             patch.object(collector, 'label_by_set_id', side_effect=ValueError('no ingredient match')) as check:
            with self.assertRaisesRegex(ValueError, 'after 8/12 candidate checks'):
                collector.select_label('azithromycin', max_pages=5, max_candidates=12)
        self.assertEqual(check.call_count, 8)
        self.assertEqual(fetch.call_count, 1)

    def test_non_tablet_form_uses_remaining_budget(self):
        payload = json.dumps({'data': [candidate(i) for i in range(30)] +
                              [candidate(200, 'AZITHROMYCIN CAPSULE')],
                              'metadata': {'total_pages': 1}}).encode()
        def verify(set_id, drug, candidate):
            if candidate['title'].endswith('CAPSULE'):
                return {'Oral tablet': False, 'Set ID': set_id}
            raise ValueError('no match')
        with patch.object(collector, 'fetch', return_value=payload), \
             patch.object(collector, 'label_by_set_id', side_effect=verify) as check:
            label = collector.select_label('azithromycin', max_candidates=12)
        self.assertEqual(check.call_count, 9)
        self.assertFalse(label['Oral tablet'])

    def test_failed_target_does_not_block_later_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            (root / 'data/target_drugs.txt').write_text('one\ntwo\n', encoding='utf-8')
            processed = []
            def process(target, *args, **kwargs):
                processed.append(target)
                raise RuntimeError('temporary API failure')
            with patch.object(collector, 'collect_target', side_effect=process):
                result = collector.run_batch(root, expected_count=2)
            self.assertEqual(result, 1)
            self.assertEqual(processed, ['one', 'two'])
            self.assertEqual(len(collector.load_state(root)['errors']), 2)

    def test_invalid_budgets_raise(self):
        for settings in ({'max_candidates': 0}, {'search_seconds': 0}, {'max_pages': 0}):
            with self.assertRaises(ValueError):
                collector.select_label('azithromycin', **settings)


if __name__ == '__main__':
    unittest.main()
