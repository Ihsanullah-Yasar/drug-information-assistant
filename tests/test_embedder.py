"""Offline tests for Step 5; no model download, GPU, or network required."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.chunker import chunk_page
from src.embedder import BGEEmbedder, EMBEDDING_DIM, QUERY_INSTRUCTION
from scripts.embed import generate_embeddings, read_chunks, validate_archive


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}


class FakeModel:
    max_seq_length = 512
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []
        self.bad_dim = False
        self.nan = False
        self.bad_norm = False

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        dim = 383 if self.bad_dim else 384
        result = np.zeros((len(texts), dim), dtype=np.float32)
        result[:, 0] = 1
        if self.nan:
            result[0, 0] = np.nan
        if self.bad_norm:
            result[:, 0] = 2
        return result


def save_test_chunks(directory: Path, name="a") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    chunks = chunk_page("Do not exceed the recommended dose. Indications and warnings. " * 36, f"{name}.pdf", 1)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    return path


class EmbedderTests(unittest.TestCase):
    def test_dimension_and_unit_norm(self):
        model = FakeModel()
        wrapper = BGEEmbedder(model=model)
        result = wrapper.encode_passages(["Drug label warnings", "Recommended dose"])
        self.assertEqual(result.shape, (2, EMBEDDING_DIM))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(result, axis=1), 1))
        self.assertEqual(model.calls[0], ["Drug label warnings", "Recommended dose"])

    def test_query_instruction_not_added_to_passage(self):
        model = FakeModel()
        wrapper = BGEEmbedder(model=model)
        wrapper.encode_query("What are the warnings?")
        self.assertEqual(model.calls[0], [QUERY_INSTRUCTION + "What are the warnings?"])
        wrapper.encode_passages(["Label text"])
        self.assertEqual(model.calls[1], ["Label text"])

    def test_rejects_empty_input(self):
        wrapper = BGEEmbedder(model=FakeModel())
        for value in ([], ["  "], "raw str"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                wrapper.encode_passages(value)

    def test_token_overflow_not_silently_truncated(self):
        model = FakeModel()
        wrapper = BGEEmbedder(model=model)
        with self.assertRaisesRegex(ValueError, "exceeding model limit"):
            wrapper.encode_passages(["warning " * 512])
        self.assertEqual(model.calls, [])

    def test_rejects_invalid_model_outputs(self):
        model = FakeModel()
        wrapper = BGEEmbedder(model=model)
        for attr in ("bad_dim", "nan", "bad_norm"):
            setattr(model, attr, True)
            with self.subTest(attr=attr), self.assertRaises(ValueError):
                wrapper.encode_passages(["Drug facts"])
            setattr(model, attr, False)

    def test_read_chunks_detects_modified_identity(self):
        with TemporaryDirectory() as temp:
            path = save_test_chunks(Path(temp))
            rows = json.loads(path.read_text())
            rows[0]["text"] = "MALICIOUS " + rows[0]["text"]
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "invalid page/index/offsets|altered"):
                read_chunks(path)

    def test_archive_roundtrip_and_second_run_reuses(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir, output_dir = root / "chunks", root / "embeddings"
            save_test_chunks(input_dir)
            m = FakeModel()
            first = generate_embeddings(input_dir, output_dir, embedder_factory=lambda: BGEEmbedder(model=m))
            self.assertEqual(first["documents"], 1)
            self.assertEqual(first["saved"], 1)
            chunks, sha = read_chunks(input_dir / "a.json")
            self.assertEqual(validate_archive(output_dir / "a.npz", chunks, sha), len(chunks))
            called = []
            second = generate_embeddings(input_dir, output_dir, embedder_factory=lambda: called.append(True))
            self.assertEqual(second["reused"], 1)
            self.assertFalse(called)

    def test_rebuilds_changed_chunks_without_overwriting_other_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inp, out = root / "chunks", root / "embeddings"
            path = save_test_chunks(inp)
            save_test_chunks(inp, "b")
            factory = lambda: BGEEmbedder(model=FakeModel())
            first = generate_embeddings(inp, out, embedder_factory=factory)
            self.assertEqual(first["saved"], 2)
            existing = (out / "b.npz").read_bytes()
            # Write a new valid chunk that has changed ID/text.
            changed = chunk_page("New medication label warning. " * 36, "a.pdf", 1)
            path.write_text(json.dumps(changed), encoding="utf-8")
            second = generate_embeddings(inp, out, embedder_factory=factory)
            self.assertEqual(second["saved"], 1)
            self.assertEqual(second["reused"], 1)
            self.assertEqual((out / "b.npz").read_bytes(), existing)

    def test_audit_only_requires_archives_without_loading_model(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inp, out = root / "chunks", root / "embeddings"
            save_test_chunks(inp)
            with self.assertRaises(FileNotFoundError):
                generate_embeddings(inp, out, audit_only=True)
            generate_embeddings(inp, out, embedder_factory=lambda: BGEEmbedder(model=FakeModel()))
            summary = generate_embeddings(inp, out, audit_only=True, embedder_factory=lambda: self.fail("should not load model"))
            self.assertEqual(summary["embeddings"], summary["unique_ids"])

    def test_rejects_tampered_archive(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inp, out = root / "chunks", root / "embeddings"
            save_test_chunks(inp)
            generate_embeddings(inp, out, embedder_factory=lambda: BGEEmbedder(model=FakeModel()))
            chunks, sha = read_chunks(inp / "a.json")
            archive = out / "a.npz"
            with np.load(archive, allow_pickle=False) as original:
                content = {key: original[key] for key in original.files}
            content["embeddings"] = np.ones_like(content["embeddings"])
            with archive.open("wb") as f:
                np.savez_compressed(f, **content)
            with self.assertRaises(ValueError):
                validate_archive(archive, chunks, sha)
            with self.assertRaisesRegex(ValueError, "Outdated or corrupt"):
                generate_embeddings(inp, out, audit_only=True)

    def test_limit_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inp, out = root / "chunks", root / "embeddings"
            save_test_chunks(inp, "a")
            save_test_chunks(inp, "b")
            res = generate_embeddings(inp, out, limit_files=1, embedder_factory=lambda: BGEEmbedder(model=FakeModel()))
            self.assertEqual(res["documents"], 1)
            self.assertTrue((out / "a.npz").exists())
            self.assertFalse((out / "b.npz").exists())

    def test_invalid_limit_batch_and_missing_path(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inp, out = root / "chunks", root / "embeddings"
            save_test_chunks(inp)
            with self.assertRaises(ValueError):
                generate_embeddings(inp, out, limit_files=0)
            with self.assertRaises(ValueError):
                generate_embeddings(inp, out, batch_size=0)
            with self.assertRaises(FileNotFoundError):
                generate_embeddings(root / "missing", out)


if __name__ == "__main__":
    unittest.main()
