"""Offline tests for Step 2: PDF document loading and existing ingestion."""

import json
from pathlib import Path
import tempfile
import unittest

import pymupdf

from src.loader import load_pdf
from scripts.ingest import ingest_pdfs


def create_pdf(path: Path, page_texts: list[str]) -> None:
    """Create small synthetic PDFs; never touch data/documents/."""
    with pymupdf.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        pdf.save(path)


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_load_pdf_returns_one_record_per_page(self):
        path = self.root / "sample_drug.pdf"
        create_pdf(path, ["Acetaminophen 325 mg", "Liver warning"])
        records = load_pdf(path)
        self.assertEqual(len(records), 2)
        self.assertEqual([r["page_number"] for r in records], [1, 2])
        self.assertEqual([r["filename"] for r in records], [path.name] * 2)
        self.assertIn("Acetaminophen 325 mg", records[0]["text"])
        self.assertIn("Liver warning", records[1]["text"])

    def test_blank_page_is_preserved(self):
        path = self.root / "blank.pdf"
        create_pdf(path, ["Heading", "", "Last page"])
        records = load_pdf(path)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]["page_number"], 2)
        self.assertEqual(records[1]["text"], "")

    def test_missing_pdf_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_pdf(self.root / "does_not_exist.pdf")

    def test_wrong_extension_is_rejected(self):
        path = self.root / "not_a_pdf.txt"
        path.write_text("Hello", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_pdf(path)

    def test_corrupted_pdf_is_rejected(self):
        path = self.root / "corrupt.pdf"
        path.write_bytes(b"This is not a PDF")
        with self.assertRaises((RuntimeError, ValueError)):
            load_pdf(path)

    def test_password_protected_pdf_is_rejected(self):
        path = self.root / "private.pdf"
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "Secret")
            pdf.save(
                path,
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner-password",
                user_pw="user-password",
            )
        with self.assertRaisesRegex(ValueError, "Password-protected"):
            load_pdf(path)

    def test_existing_ingest_creates_json_without_changing_pdf(self):
        documents = self.root / "documents"
        raw = self.root / "raw"
        documents.mkdir()
        path = documents / "medicine.pdf"
        create_pdf(path, ["Drug uses", "Safety warning"])
        before = path.read_bytes()

        ingest_pdfs(documents, raw)

        output = raw / "medicine.json"
        records = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["page_number"], 1)
        self.assertEqual(records[1]["page_number"], 2)
        self.assertEqual(path.read_bytes(), before)

        # A second run reproduces exactly the same JSON file.
        first_json = output.read_bytes()
        ingest_pdfs(documents, raw)
        self.assertEqual(output.read_bytes(), first_json)

    def test_ingest_rejects_empty_directory(self):
        documents = self.root / "documents"
        documents.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "No PDF"):
            ingest_pdfs(documents, self.root / "raw")


if __name__ == "__main__":
    unittest.main()
