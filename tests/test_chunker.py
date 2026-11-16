"""Offline tests for manual chunking and its ingestion adapter."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.chunk import chunk_documents
from src.chunker import chunk_document, chunk_page


class ChunkerTests(unittest.TestCase):
    def test_short_page_one_chunk(self):
        result = chunk_page("Do not exceed 10 mg.", "drug.pdf", 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Do not exceed 10 mg.")
        self.assertEqual((result[0]["start_char"], result[0]["end_char"]), (0, 20))

    def test_empty_page_produces_no_chunks(self):
        self.assertEqual(chunk_page(" \n", "drug.pdf", 1), [])

    def test_no_gap_complete_coverage_and_bounded_length(self):
        original = "This text is meaningful. " * 130
        results = chunk_page(original, "drug.pdf", 4, chunk_size=1000, overlap=180)
        self.assertGreater(len(results), 1)
        self.assertEqual(results[0]["start_char"], 0)
        self.assertEqual(results[-1]["end_char"], len(original))
        for i, chunk in enumerate(results):
            self.assertEqual(chunk["text"], original[chunk["start_char"]:chunk["end_char"]])
            self.assertLessEqual(len(chunk["text"]), 1000)
            self.assertTrue(chunk["text"].strip())
            if i:
                prev = results[i - 1]
                self.assertLessEqual(chunk["start_char"], prev["end_char"])
                self.assertGreater(chunk["end_char"], prev["end_char"])

    def test_page_boundary_never_crossed(self):
        records = [{"filename": "x.pdf", "page_number": 1, "text": "A" * 1500},
                   {"filename": "x.pdf", "page_number": 2, "text": "B" * 1500}]
        chunks = chunk_document(records, chunk_size=1000, overlap=180)
        self.assertTrue(all(set(c["text"]) == ({"A"} if c["page_number"] == 1 else {"B"}) for c in chunks))
        self.assertEqual([c["chunk_index"] for c in chunks if c["page_number"] == 2], [1, 2])

    def test_ids_stable_and_unique(self):
        a = chunk_page("abc " * 500, "a.pdf", 1, chunk_size=1000, overlap=180)
        b = chunk_page("abc " * 500, "a.pdf", 1, chunk_size=1000, overlap=180)
        self.assertEqual(a, b)
        self.assertEqual(len({c["chunk_id"] for c in a}), len(a))
        self.assertNotEqual(a[0]["chunk_id"], chunk_page("abc " * 500, "a.pdf", 2)[0]["chunk_id"])

    def test_invalid_parameters(self):
        for size, overlap in [(0, 0), (10, -1), (10, 10), (10, 11), (True, 0), (10, False)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_page("text", "a.pdf", 1, chunk_size=size, overlap=overlap)

    def test_metadata_validation(self):
        with self.assertRaises(ValueError):
            chunk_document([{"filename": "a.pdf", "page_number": 2, "text": "x"}])
        with self.assertRaises(ValueError):
            chunk_document([{"filename": "a.pdf", "page_number": 1, "text": "x"},
                            {"filename": "b.pdf", "page_number": 2, "text": "y"}])

    def test_output_and_rerun(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            src, out = root / "cleaned", root / "chunks"
            src.mkdir()
            pages = [{"filename": "drug.pdf", "page_number": 1, "text": "Safety warning. " * 150},
                     {"filename": "drug.pdf", "page_number": 2, "text": ""}]
            (src / "drug.json").write_text(json.dumps(pages), encoding="utf-8")
            first = chunk_documents(src, out)
            saved = (out / "drug.json").read_bytes()
            second = chunk_documents(src, out)
            self.assertEqual(first, second)
            self.assertEqual(saved, (out / "drug.json").read_bytes())
            self.assertEqual(first["empty_pages"], 1)
            self.assertEqual(first["chunks"], first["unique_ids"])

    def test_stale_output_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            src, out = root / "cleaned", root / "chunks"
            src.mkdir()
            out.mkdir()
            (src / "drug.json").write_text(json.dumps([{"filename": "drug.pdf", "page_number": 1, "text": "Hi"}]))
            (out / "old.json").write_text("[]")
            with self.assertRaises(ValueError):
                chunk_documents(src, out)


if __name__ == "__main__":
    unittest.main()
