"""Step 7: offline semantic retrieval tests, plus optional real-Chroma smoke test.

These tests deliberately do not download the BGE model or modify the user's
production data/chroma directory.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.retrieve import run_query
from src.embedder import EMBEDDING_DIM
from src.retriever import retrieve_chunks


class DummyEmbedder:
    def __init__(self, vector=None):
        self.calls = []
        if vector is None:
            vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            vector[0] = 1
        self.vector = vector

    def encode_query(self, question: str):
        self.calls.append(question)
        return self.vector


class FakeSearchCollection:
    def __init__(self, records=None):
        self.calls = []
        self.records = list(records) if records is not None else default_records()
        self.bad_response = None

    def count(self):
        return len(self.records)

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.bad_response is not None:
            return self.bad_response
        vector = np.array(kwargs["query_embeddings"][0], dtype=np.float32)
        rows = self.records
        if "where" in kwargs:
            filter_ = kwargs["where"]
            rows = [r for r in rows if all(r["md"].get(k) == v for k, v in filter_.items())]
        ranked = sorted(rows, key=lambda r: float(1 - np.dot(r["vector"], vector)))
        ranked = ranked[:kwargs["n_results"]]
        return {
            "ids": [[r["id"] for r in ranked]],
            "documents": [[r["text"] for r in ranked]],
            "metadatas": [[r["md"] for r in ranked]],
            "distances": [[float(1 - np.dot(r["vector"], vector)) for r in ranked]],
        }


def default_records():
    rows = []
    names = ["ibuprofen_200mg_demo.pdf", "acetaminophen_325mg_demo.pdf", "ibuprofen_200mg_demo.pdf"]
    for i, filename in enumerate(names):
        angle = i * 0.32
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vector[0] = np.cos(angle)
        vector[1] = np.sin(angle)
        text = f"This is the drug label extract number {i + 1}."
        rows.append({
            "id": f"chunk-{i}", "text": text, "vector": vector,
            "md": {
                "filename": filename, "page_number": i + 1,
                "chunk_index": 1, "start_char": 0, "end_char": len(text),
                "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc",
                "packager": "Demo Pharmaceuticals", "label_updated_on": "2026-11-13",
            },
        })
    return rows


class RetrieverTests(unittest.TestCase):
    def test_top_five_default_and_correct_cosine_ranking(self):
        db, model = FakeSearchCollection(), DummyEmbedder()
        results = retrieve_chunks("  Which drug label?  ", collection=db, embedder=model)
        self.assertEqual(len(results), 3)
        self.assertEqual([x["rank"] for x in results], [1, 2, 3])
        self.assertEqual([x["chunk_id"] for x in results], ["chunk-0", "chunk-1", "chunk-2"])
        self.assertEqual(db.calls[0]["n_results"], 3)
        self.assertNotIn("query_texts", db.calls[0])
        self.assertIn("query_embeddings", db.calls[0])
        self.assertEqual(model.calls, ["Which drug label?"])
        self.assertAlmostEqual(results[0]["cosine_similarity"], 1.0, places=5)
        self.assertEqual(results[0]["filename"], "ibuprofen_200mg_demo.pdf")
        self.assertEqual(results[0]["page_number"], 1)

    def test_exact_pdf_filter(self):
        db = FakeSearchCollection()
        results = retrieve_chunks("question", collection=db, embedder=DummyEmbedder(),
                                  filename="ibuprofen_200mg_demo.pdf", top_k=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(db.calls[-1]["where"], {"filename": "ibuprofen_200mg_demo.pdf"})
        self.assertTrue(all(x["filename"] == "ibuprofen_200mg_demo.pdf" for x in results))

    def test_filter_with_no_matches_returns_no_false_evidence(self):
        results = retrieve_chunks("question", collection=FakeSearchCollection(),
                                  embedder=DummyEmbedder(), filename="missing.pdf")
        self.assertEqual(results, [])

    def test_validates_query_and_topk_without_calling_model(self):
        db, model = FakeSearchCollection(), DummyEmbedder()
        for question in ("", " \n", None):
            with self.assertRaises(ValueError):
                retrieve_chunks(question, collection=db, embedder=model)
        for top_k in (0, -1, 51, 1.2, True):
            with self.assertRaises(ValueError):
                retrieve_chunks("question", collection=db, embedder=model, top_k=top_k)
        self.assertEqual(model.calls, [])

    def test_rejects_path_in_pdf_filename(self):
        for filename in ("../secret.pdf", "dir/a.pdf", r"C:\tmp\secret.pdf", "", "ibuprofen.json"):
            with self.assertRaises(ValueError):
                retrieve_chunks("question", collection=FakeSearchCollection(),
                                embedder=DummyEmbedder(), filename=filename)

    def test_rejects_empty_or_incomplete_index(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            retrieve_chunks("query", collection=FakeSearchCollection([]), embedder=DummyEmbedder())
        with self.assertRaisesRegex(ValueError, "Expected 7194"):
            retrieve_chunks("query", collection=FakeSearchCollection(),
                            embedder=DummyEmbedder(), expected_count=7194)

    def test_rejects_invalid_query_vectors(self):
        vectors = [np.zeros(EMBEDDING_DIM, dtype=np.float32),
                   np.ones(100, dtype=np.float32),
                   np.array([float("nan")] + [0] * (EMBEDDING_DIM-1)),
                   np.ones(EMBEDDING_DIM, dtype=np.float32)]
        for vector in vectors:
            with self.assertRaises(ValueError):
                retrieve_chunks("question", collection=FakeSearchCollection(),
                                embedder=DummyEmbedder(vector))

    def test_rejects_misaligned_columns(self):
        db = FakeSearchCollection()
        db.bad_response = {"ids": [["x"]], "documents": [["one"]],
                           "metadatas": [[]], "distances": [[0.1]]}
        with self.assertRaisesRegex(ValueError, "misaligned"):
            retrieve_chunks("question", collection=db, embedder=DummyEmbedder())

    def test_rejects_duplicate_ids(self):
        db = FakeSearchCollection()
        records = default_records()[:2]
        db.bad_response = {
            "ids": [["same", "same"]],
            "documents": [[r["text"] for r in records]],
            "metadatas": [[r["md"] for r in records]],
            "distances": [[0.1, 0.2]],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            retrieve_chunks("question", collection=db, embedder=DummyEmbedder())

    def test_rejects_bad_provenance_and_cross_drug_mismatch(self):
        db = FakeSearchCollection()
        db.records[0]["md"]["source_url"] = "https://evil.example/a"
        with self.assertRaisesRegex(ValueError, "untrusted"):
            retrieve_chunks("question", collection=db, embedder=DummyEmbedder())
        db = FakeSearchCollection()
        db.records[0]["md"]["page_number"] = 0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            retrieve_chunks("question", collection=db, embedder=DummyEmbedder())

    def test_rejects_nonfinite_and_unsorted_distances(self):
        db = FakeSearchCollection()
        records = default_records()[:2]
        for distances in ([float("nan"), 0.1], [0.3, 0.1], [-1.0, 0.1]):
            db.bad_response = {
                "ids": [[r["id"] for r in records]],
                "documents": [[r["text"] for r in records]],
                "metadatas": [[r["md"] for r in records]],
                "distances": [distances],
            }
            with self.assertRaises(ValueError):
                retrieve_chunks("question", collection=db, embedder=DummyEmbedder())

    def test_cli_saves_full_citable_text_json(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "results" / "query.json"
            model = DummyEmbedder()
            result = run_query(
                "What are common effects?", db_dir=Path(tmp) / "unused",
                collection=FakeSearchCollection(), embedder=model,
                top_k=2, expected_count=3, json_out=output,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(result), 2)
            self.assertEqual(payload["question"], "What are common effects?")
            self.assertEqual(payload["matches"][0]["text"], result[0]["text"])
            self.assertIn("source_url", payload["matches"][0])
            self.assertEqual(model.calls, ["What are common effects?"])

    def test_optional_real_chroma_query_in_child_process(self):
        if importlib.util.find_spec("chromadb") is None:
            self.skipTest("chromadb not installed in authoring environment")
        # Child exits before TemporaryDirectory cleanup to avoid WinError 32
        # from a Chroma/HNSW file still held open on Windows.
        code = r'''
import json, numpy as np, sys
from pathlib import Path
from src.vectordb import connect_collection
from src.retriever import retrieve_chunks
class Stub:
    def encode_query(self, text):
        v = np.zeros(384, dtype=np.float32)
        v[0] = 1
        return v
c = connect_collection(Path(sys.argv[1]))
texts = ["This is a test drug-label section.", "Other drug-label section."]
vecs = np.zeros((2,384), dtype=np.float32)
vecs[0,0] = 1
vecs[1,1] = 1
c.upsert(
    ids=["a"*64, "b"*64], documents=texts,
    embeddings=vecs.tolist(),
    metadatas=[{
       "filename": "drug.pdf", "page_number": i+1, "chunk_index": 1,
       "start_char": 0, "end_char": len(texts[i]),
       "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc",
       "packager":"Demo", "label_updated_on":"2026-11-13",
    } for i in range(2)],
)
rows = retrieve_chunks("question", collection=c, embedder=Stub(), top_k=2)
assert len(rows) == 2 and rows[0]["page_number"] == 1
assert rows[0]["cosine_similarity"] > rows[1]["cosine_similarity"]
print(json.dumps({"count":c.count(), "first":rows[0]["filename"]}))
'''
        with TemporaryDirectory() as tmp:
            project = Path(__file__).resolve().parents[1]
            proc = subprocess.run(
                [sys.executable, "-c", code, str(Path(tmp) / "chroma")],
                cwd=project, capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn('"count": 2', proc.stdout)


if __name__ == "__main__":
    unittest.main()
