"""Step 6 offline tests. FakeChroma mimics the used public Chroma collection API.

No network, model downloads, real Chroma installation, or corpus files needed.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

import numpy as np

from scripts.embed import read_chunks, write_archive
from scripts.index_vectors import index_vectors, read_sources
from src.chunker import chunk_page
from src.embedder import EMBEDDING_DIM
from src.vectordb import (
    COLLECTION_METADATA, COLLECTION_NAME, connect_collection,
    vector_sha256,
)


class FakeCollection:
    def __init__(self, metadata=None, configuration=None):
        self.metadata = dict(metadata or COLLECTION_METADATA)
        self.configuration = configuration or {"hnsw": {"space": "cosine"}}
        self.items = {}
        self.write_calls = 0

    def get(self, ids=None, where=None, include=None, limit=None, offset=None):
        records = list(self.items.items())
        if ids is not None:
            allowed = set(ids)
            records = [(id_, r) for id_, r in records if id_ in allowed]
        if where is not None:
            records = [(id_, r) for id_, r in records if all(r["metadata"].get(k) == v for k, v in where.items())]
        if offset is not None:
            records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return {
            "ids": [id_ for id_, _ in records],
            "documents": [r["text"] for _, r in records],
            "metadatas": [r["metadata"] for _, r in records],
            "embeddings": [r["vector"] for _, r in records],
        }

    def upsert(self, *, ids, documents, metadatas, embeddings):
        self.write_calls += 1
        for id_, doc, md, vector in zip(ids, documents, metadatas, embeddings, strict=True):
            self.items[id_] = {"text": doc, "metadata": dict(md), "vector": np.array(vector, dtype=np.float32)}

    def delete(self, *, ids):
        for id_ in ids:
            self.items.pop(id_, None)

    def count(self):
        return len(self.items)


class FakeClient:
    def __init__(self):
        self.collection = None

    def get_or_create_collection(self, *, name, metadata, configuration, embedding_function=None):
        if self.collection is None:
            assert name == COLLECTION_NAME
            self.collection = FakeCollection(metadata=metadata, configuration=configuration)
        return self.collection

    def get_collection(self, *, name):
        if self.collection is None:
            raise ValueError("missing collection")
        return self.collection


def make_sample_data(root: Path, names=("a", "b")):
    chunks_dir, embedding_dir = root / "chunks", root / "embeddings"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(names):
        filename = f"{name}.pdf"
        text = ("IMPORTANT: Do not mix with other medication. Confirm the warnings. " * 38)
        chunks = chunk_page(text, filename, 1, chunk_size=360, overlap=50)
        path = chunks_dir / f"{name}.json"
        path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        actual_chunks, digest = read_chunks(path)
        vectors = np.zeros((len(chunks), EMBEDDING_DIM), dtype=np.float32)
        vectors[:, i] = 1.0
        write_archive(embedding_dir / f"{name}.npz", actual_chunks, vectors, digest)
        rows.append({
            "filename": filename, "product_name": name, "packager": f"Packager {name}",
            "label_updated_on": "November 13, 2026",
            "source_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={uuid4()}",
        })
    sources = root / "sources.csv"
    with sources.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return chunks_dir, embedding_dir, root / "chroma", sources


class VectorDbTests(unittest.TestCase):
    def index(self, dirs, client, **opts):
        return index_vectors(*dirs, client=client, **opts)

    def test_initial_index_and_idempotent_rerun(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp))
            client = FakeClient()
            first = self.index(dirs, client, batch_size=3)
            self.assertEqual(first["documents"], 2)
            self.assertEqual(first["vectors"], client.collection.count())
            calls = client.collection.write_calls
            second = self.index(dirs, client)
            self.assertEqual(second["reused"], 2)
            self.assertEqual(calls, client.collection.write_calls)
            self.index(dirs, client, audit_only=True)

    def test_provenance_metadata_and_384_dimensions(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp), names=("a",))
            client = FakeClient()
            self.index(dirs, client)
            example = next(iter(client.collection.items.values()))
            md = example["metadata"]
            self.assertEqual(md["filename"], "a.pdf")
            self.assertEqual(md["page_number"], 1)
            self.assertIn("dailymed.nlm.nih.gov", md["source_url"])
            self.assertEqual(len(example["vector"]), 384)
            self.assertEqual(md["vector_sha256"], vector_sha256(example["vector"]))
            self.assertGreater(md["end_char"], md["start_char"])

    def test_partial_index_is_not_mistaken_for_full_audit(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp))
            client = FakeClient()
            partial = self.index(dirs, client, limit_files=1)
            self.assertEqual(partial["documents"], 1)
            self.index(dirs, client, limit_files=1, audit_only=True)
            with self.assertRaisesRegex(ValueError, "Missing, stale"):
                self.index(dirs, client, audit_only=True)
            self.index(dirs, client)
            self.index(dirs, client, audit_only=True)

    def test_changed_chunk_removes_old_ids(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp))
            client = FakeClient()
            self.index(dirs, client)
            prior_a_ids = set(client.collection.get(where={"filename": "a.pdf"})["ids"])
            chunk_path = dirs[0] / "a.json"
            chunks = chunk_page("New label text about allergies. " * 40, "a.pdf", 1, chunk_size=360, overlap=50)
            chunk_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
            actual, digest = read_chunks(chunk_path)
            vectors = np.zeros((len(actual), EMBEDDING_DIM), dtype=np.float32)
            vectors[:, 0] = 1
            write_archive(dirs[1] / "a.npz", actual, vectors, digest)
            self.index(dirs, client)
            current_a_ids = set(client.collection.get(where={"filename": "a.pdf"})["ids"])
            self.assertFalse(prior_a_ids.intersection(current_a_ids))
            self.assertEqual(current_a_ids, {c["chunk_id"] for c in chunks})
            self.index(dirs, client, audit_only=True)

    def test_changed_source_url_refreshes_metadata(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp), names=("a",))
            client = FakeClient()
            self.index(dirs, client)
            old = next(iter(client.collection.items.values()))["metadata"]["source_url"]
            rows = read_sources(dirs[3])
            rows["a.pdf"]["source_url"] = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00000000-0000-4000-8000-000000000001"
            with dirs[3].open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
                writer.writeheader()
                writer.writerows(rows.values())
            with self.assertRaisesRegex(ValueError, "Missing, stale"):
                self.index(dirs, client, audit_only=True)
            self.index(dirs, client)
            new = next(iter(client.collection.items.values()))["metadata"]["source_url"]
            self.assertNotEqual(old, new)

    def test_tampered_chroma_vector_audit_fails_then_repairs(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp), names=("a",))
            client = FakeClient()
            self.index(dirs, client)
            record = next(iter(client.collection.items.values()))
            record["vector"] = np.full((384,), 0.3, dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "Missing, stale"):
                self.index(dirs, client, audit_only=True)
            self.index(dirs, client)
            self.index(dirs, client, audit_only=True)

    def test_missing_or_tampered_embedding_archive_rejected(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp), names=("a",))
            (dirs[1] / "a.npz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Complete Step 5"):
                self.index(dirs, FakeClient())

    def test_wrong_collection_metadata_rejected(self):
        client = FakeClient()
        client.collection = FakeCollection(metadata={**COLLECTION_METADATA, "model_name": "other-model"})
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            connect_collection(Path("unused"), client=client)

    def test_wrong_cosine_configuration_rejected(self):
        client = FakeClient()
        client.collection = FakeCollection(configuration={"hnsw": {"space": "l2"}})
        with self.assertRaisesRegex(ValueError, "cosine"):
            connect_collection(Path("unused"), client=client)

    def test_unrelated_extra_record_detected_on_full_audit(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp), names=("a",))
            client = FakeClient()
            self.index(dirs, client)
            client.collection.items["unexpected"] = {
                "text": "unrelated", "metadata": {"filename": "unrelated.pdf"},
                "vector": np.zeros((384,), dtype=np.float32),
            }
            with self.assertRaisesRegex(ValueError, "collection contains"):
                self.index(dirs, client, audit_only=True)

    def test_bad_csv_or_unlisted_source_rejected(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp), names=("a",))
            with dirs[3].open("a", encoding="utf-8") as f:
                f.write(dirs[3].read_text(encoding="utf-8").splitlines()[1] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate/unsafe"):
                read_sources(dirs[3])

    @unittest.skipUnless(importlib.util.find_spec("chromadb"), "chromadb is not installed here")
    def test_real_chromadb_persistence_and_vector_query(self):
        """Check real persistence in child processes to release Windows HNSW file locks.

        Chroma's native HNSW index may keep data_level0.bin open as long as
        the Python process is alive. Windows cannot delete an open file.
        Running the write and reopen phases in separate Python processes
        ensures each process exits before TemporaryDirectory cleans up.
        """
        worker = r'''
import json
import sys
from pathlib import Path

import numpy as np
from scripts.index_vectors import index_vectors
from src.embedder import EMBEDDING_DIM
from src.vectordb import connect_collection

phase = sys.argv[1]
dirs = tuple(Path(arg) for arg in sys.argv[2:])
expected_count = sum(
    len(json.loads(path.read_text(encoding="utf-8")))
    for path in dirs[0].glob("*.json")
)
if phase == "write":
    summary = index_vectors(*dirs, batch_size=4)
    if summary["collection_count"] != expected_count:
        raise AssertionError("Index did not store every expected vector")
elif phase == "reopen":
    collection = connect_collection(dirs[2], create=False)
    if collection.count() != expected_count:
        raise AssertionError("Persisted collection count changed across processes")
    query = np.zeros((EMBEDDING_DIM,), dtype=np.float32)
    query[0] = 1.0
    result = collection.query(
        query_embeddings=[query.tolist()],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    if result["metadatas"][0][0]["filename"] != "a.pdf":
        raise AssertionError("Vector search returned the wrong source document")
    if abs(float(result["distances"][0][0])) >= 1e-4:
        raise AssertionError("Cosine distance differs from the expected zero")
    index_vectors(*dirs, audit_only=True)
else:
    raise ValueError("Unknown integration test phase: " + phase)
'''
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp))
            project_root = Path(__file__).resolve().parents[1]
            for phase in ("write", "reopen"):
                result = subprocess.run(
                    [sys.executable, "-c", worker, phase, *(str(p) for p in dirs)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode != 0:
                    self.fail(
                        f"Chroma {phase} subprocess exited with code {result.returncode}\n"
                        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                    )
        # Both child processes exited before the temp directory was deleted;
        # Windows' file locks are therefore released by the operating system.

    def test_bad_parameters_rejected(self):
        with TemporaryDirectory() as tmp:
            dirs = make_sample_data(Path(tmp), names=("a",))
            with self.assertRaisesRegex(ValueError, "limit_files"):
                self.index(dirs, FakeClient(), limit_files=0)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                self.index(dirs, FakeClient(), batch_size=0)


if __name__ == "__main__":
    unittest.main()
