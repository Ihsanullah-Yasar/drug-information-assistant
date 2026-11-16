"""Unit tests for selective token-overflow repair. Network/model downloads not needed."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.chunker import chunk_page
from scripts.embed import read_chunks
from scripts.repair_token_overflow import repair_chunks, scan_and_repair, token_counts


class CharTokenizer:
    """Mock: each character counts as one token plus 2 special tokens."""
    def __call__(self, texts, **kwargs):
        return {"input_ids": [list(range(len(t) + 2)) for t in texts]}


def source_pages(texts: list[str], name: str) -> list[dict]:
    return [{"filename": f"{name}.pdf", "page_number": i, "text": t}
            for i, t in enumerate(texts, 1)]


def fixtures(root: Path, name="drug", texts=None):
    texts = texts or ["Safety warning! " * 80, "Allergy warning. " * 12]
    chunk_dir, clean_dir = root / "data/chunks", root / "data/cleaned"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    pages = source_pages(texts, name)
    chunks = [c for page in pages for c in chunk_page(page["text"], page["filename"], page["page_number"]) ]
    cp = chunk_dir / f"{name}.json"
    cp.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (clean_dir / f"{name}.json").write_text(json.dumps(pages), encoding="utf-8")
    return cp, chunks, pages


class TokenRepairTests(unittest.TestCase):
    def test_no_change_to_safe_chunks(self):
        chunks, pages = chunk_page("short.", "drug.pdf", 1), source_pages(["short."], "drug")
        repaired, result = repair_chunks(chunks, pages, "drug.json", CharTokenizer())
        self.assertEqual(chunks, repaired)
        self.assertEqual(result["affected_pages"], [])

    def test_only_oversized_page_changes(self):
        with TemporaryDirectory() as temp:
            _, chunks, pages = fixtures(Path(temp))
            page2 = [c for c in chunks if c["page_number"] == 2]
            new, report = repair_chunks(chunks, pages, "drug.json", CharTokenizer())
            self.assertEqual(report["affected_pages"], [1])
            self.assertEqual(page2, [c for c in new if c["page_number"] == 2])
            self.assertTrue(all(x <= 512 for x in token_counts([c["text"] for c in new], CharTokenizer())))
            self.assertEqual(new[0]["start_char"], 0)

    def test_dry_run_does_not_modify_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path, _, _ = fixtures(root)
            before = path.read_bytes()
            report = scan_and_repair(root, CharTokenizer())
            self.assertEqual(report["modified"], 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root / "data/chunks/_token_repair_backups").exists())

    def test_apply_preserves_backup_and_rerun(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path, original, _ = fixtures(root)
            safe, _, _ = fixtures(root, name="safe", texts=["short text."])
            safe_bytes = safe.read_bytes()
            old = path.read_bytes()
            report = scan_and_repair(root, CharTokenizer(), apply=True)
            self.assertEqual(report["modified"], 1)
            self.assertNotEqual(old, path.read_bytes())
            self.assertEqual(safe.read_bytes(), safe_bytes)
            backup = root / "data/chunks/_token_repair_backups/drug.json"
            self.assertEqual(backup.read_bytes(), old)
            read_chunks(path)
            result = scan_and_repair(root, CharTokenizer(), apply=True)
            self.assertEqual(result["modified"], 0)
            self.assertEqual(backup.read_bytes(), old)

    def test_bad_source_rejected_without_modification(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path, _, _ = fixtures(root)
            source_path = root / "data/cleaned/drug.json"
            pages = json.loads(source_path.read_text())
            pages[0]["text"] = "CORRUPTED" + pages[0]["text"]
            source_path.write_text(json.dumps(pages), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                scan_and_repair(root, CharTokenizer(), apply=True)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
