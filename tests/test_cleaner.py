"""Offline tests for conservative drug-label text cleaning."""

import json
from pathlib import Path
import tempfile
import unittest

from src.cleaner import clean_document, clean_text
from scripts.clean import clean_documents


class CleanerTests(unittest.TestCase):
    def test_unicode_nfc_preserves_units_and_notation(self):
        original = "Cafe\u0301 \u00b5g 10 mg/mL 10\u00b2"
        result = clean_text(original)
        self.assertEqual(result, "Caf\u00e9 \u00b5g 10 mg/mL 10\u00b2")

    def test_whitespace_and_blank_lines(self):
        self.assertEqual(clean_text("  Active   ingredient\r\n \r\n  325\t mg  \n"),
                         "Active ingredient\n325 mg")

    def test_control_characters_are_removed(self):
        self.assertEqual(clean_text("Directions\x00\nTwo tablets\x1f daily"),
                         "Directions\nTwo tablets daily")

    def test_bare_numbers_and_warnings_are_preserved(self):
        data = "5\nDo not take more than 5 tablets\nWarnings\nLiver damage"
        self.assertEqual(clean_text(data, page_number=2, total_pages=10), data)

    def test_explicit_boundary_page_label_removed(self):
        self.assertEqual(clean_text("Page 2 of 3\nStrength 250 mg\nPage 2", page_number=2, total_pages=3),
                         "Strength 250 mg")

    def test_internal_page_like_text_is_preserved(self):
        self.assertEqual(clean_text("Ingredients\nPage 2\nDirections", page_number=2, total_pages=3),
                         "Ingredients\nPage 2\nDirections")

    def test_empty_page_preserved(self):
        cleaned = clean_document([
            {"filename": "medicine.pdf", "page_number": 1, "text": "Content"},
            {"filename": "medicine.pdf", "page_number": 2, "text": "  \u0000 \n"},
        ])
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[1]["text"], "")

    def test_mismatched_page_numbers_rejected(self):
        with self.assertRaisesRegex(ValueError, "page_number"):
            clean_document([{"filename": "medicine.pdf", "page_number": 2, "text": "hello"}])

    def test_mismatched_filename_rejected(self):
        with self.assertRaisesRegex(ValueError, "filename"):
            clean_document([{"filename": "wrong.pdf", "page_number": 1, "text": "hello"}],
                           expected_filename="medicine.pdf")

    def test_non_string_text_rejected(self):
        with self.assertRaisesRegex(ValueError, "text must be a string"):
            clean_document([{"filename": "medicine.pdf", "page_number": 1, "text": None}])

    def test_clean_documents_end_to_end_and_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            output = root / "cleaned"
            raw.mkdir()
            original = [
                {"filename": "medicine.pdf", "page_number": 1, "text": " Pain  \n\nreliever "},
                {"filename": "medicine.pdf", "page_number": 2, "text": " \x00 "},
            ]
            original_bytes = (json.dumps(original) + "\n").encode()
            (raw / "medicine.json").write_bytes(original_bytes)
            first = clean_documents(raw, output)
            output_bytes = (output / "medicine.json").read_bytes()
            second = clean_documents(raw, output)
            self.assertEqual(output_bytes, (output / "medicine.json").read_bytes())
            self.assertEqual((raw / "medicine.json").read_bytes(), original_bytes)
            self.assertEqual(first, second)
            self.assertEqual(first["documents"], 1)
            self.assertEqual(first["pages"], 2)
            self.assertEqual(first["nonempty_pages"], 1)
            self.assertEqual(first["empty_pages"][0]["page_number"], 2)

    def test_stale_output_is_not_silently_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, out = root / "raw", root / "cleaned"
            raw.mkdir(); out.mkdir()
            (raw / "a.json").write_text(json.dumps([
                {"filename": "a.pdf", "page_number": 1, "text": "ok"}
            ]))
            (out / "old.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "stale"):
                clean_documents(raw, out)


if __name__ == "__main__":
    unittest.main()
