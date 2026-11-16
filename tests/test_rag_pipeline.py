"""Step 10 local UI orchestration tests (no model download, no Ollama, no Chroma writes)."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from src.prompt_builder import ABSTENTION
from src.rag_pipeline import LocalDrugRAG, available_pdf_labels, _evidence_rows
from scripts.app import format_run

IBU = "ibuprofen_200mg_haleon_us_holdings_llc.pdf"
ACET = "acetaminophen_325mg_health_pharma.pdf"
URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc123"
QUESTION = "What safety warnings are listed for ibuprofen?"


def fake_match(rank=1, *, filename=IBU, text=None, distance=None):
    text = text or f"Drug label warning on page {rank}: severe bleeding risk."
    distance = rank * .08 if distance is None else distance
    return {
        "rank": rank, "chunk_id": f"chunk-{rank}", "text": text,
        "filename": filename, "page_number": rank + 1,
        "chunk_index": 1, "start_char": 0, "end_char": len(text),
        "source_url": URL,
        "cosine_distance": distance, "cosine_similarity": 1 - distance,
    }


class FakeCollection:
    def __init__(self, count=7194):
        self.total = count
        self.calls = 0

    def count(self):
        self.calls += 1
        return self.total


class RagPipelineTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = self.root / "data"
        data.mkdir()
        with (data / "sources.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "source_url"])
            writer.writerow([IBU, URL])
            writer.writerow([ACET, URL])
        (data / "target_drugs.txt").write_text("ibuprofen\nacetaminophen\n", encoding="utf-8")
        self.collection = FakeCollection()
        self.embedder = Mock()
        self.generated = []

        def generator(package, **opts):
            self.generated.append((package, opts))
            return {
                "status": "answered", "answer": "The label warns of bleeding [S1].",
                "reason": "Citation IDs validated (not medical claim entailment)",
                "citations": ["S1"],
                "usage": {"prompt_eval_count": 1200, "eval_count": 25},
            }

        self.service = LocalDrugRAG(
            self.root, collection=self.collection, embedder=self.embedder,
            generator=generator,
        )
        self.results = [fake_match(1), fake_match(2), fake_match(3), fake_match(4), fake_match(5)]

    def _ask_with(self, results=None, question=QUESTION, **kwargs):
        with patch("src.rag_pipeline.retrieve_chunks", return_value=(self.results if results is None else results)) as retr:
            out = self.service.ask(question, **kwargs)
            return out, retr

    def test_pdf_catalog_sorted_and_valid(self):
        self.assertEqual(available_pdf_labels(self.service.sources_csv), [ACET, IBU])
        with (self.root / "data/sources.csv").open("a", encoding="utf-8") as f:
            f.write("../bad.pdf,whatever\n")
        with self.assertRaisesRegex(ValueError, "Invalid filename"):
            available_pdf_labels(self.service.sources_csv)

    def test_rejects_duplicate_and_empty_catalog(self):
        self.service.sources_csv.write_text("filename\n" + IBU + "\n" + IBU + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            available_pdf_labels(self.service.sources_csv)
        self.service.sources_csv.write_text("filename\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            available_pdf_labels(self.service.sources_csv)

    def test_end_to_end_reuses_existing_components(self):
        response, retrieve = self._ask_with()
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["filename"], IBU)
        self.assertEqual(response["retrieved_count"], 5)
        self.assertEqual(response["included_count"], 3)
        self.assertEqual(response["citations"], ["S1"])
        self.assertEqual(len(response["sources"]), 3)
        self.assertTrue(response["sources"][0]["used_in_answer"])
        self.assertFalse(response["sources"][1]["used_in_answer"])
        self.assertIn("bleeding risk", response["sources"][0]["excerpt"])
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args.kwargs["top_k"], 5)
        self.assertEqual(retrieve.call_args.kwargs["filename"], IBU)
        self.assertEqual(retrieve.call_args.kwargs["expected_count"], 7194)
        package, opts = self.generated[0]
        self.assertEqual(len(package["sources"]), 3)
        self.assertIn("severe bleeding risk", package["messages"][1]["content"])
        self.assertEqual(opts["model"], "qwen2.5:7b-instruct")
        self.assertEqual(opts["num_ctx"], 8192)
        self.assertEqual(opts["max_output_tokens"], 512)

    def test_uses_selected_source_for_unnamed_drug(self):
        result, retr = self._ask_with(
            results=[fake_match(1, filename=ACET)],
            question="What safety warnings are listed?", filename=ACET,
        )
        self.assertEqual(result["status"], "answered")
        self.assertEqual(retr.call_args.kwargs["filename"], ACET)

    def test_unknown_drug_returns_safe_scope_prompt_without_embedding(self):
        result, retr = self._ask_with(question="What general medicines do you know?")
        self.assertEqual(result["status"], "needs_drug_scope")
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(self.collection.calls, 0)
        self.assertFalse(self.generated)
        retr.assert_not_called()

    def test_two_drugs_rejected_before_retrieval(self):
        with patch("src.rag_pipeline.retrieve_chunks") as retr:
            with self.assertRaisesRegex(ValueError, "Multiple drugs"):
                self.service.ask("Compare ibuprofen and acetaminophen")
            retr.assert_not_called()
        self.assertFalse(self.generated)

    def test_invalid_label_cannot_inject_arbitrary_path(self):
        for filename in ("../private.pdf", "fake.pdf", "C:/tmp/ibuprofen.pdf"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "source catalog"):
                    self.service.ask(QUESTION, filename=filename)
        self.assertFalse(self.generated)

    def test_empty_or_long_question_rejected_early(self):
        for q in (None, "", "  ", "x" * 1001):
            with self.subTest(question=str(q)[:10]):
                with self.assertRaises(ValueError):
                    self.service.ask(q)
        self.assertEqual(self.collection.calls, 0)

    def test_bad_max_sources_rejected(self):
        for n in (0, -1, 6, True, 1.2, "3"):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    self.service.ask(QUESTION, max_sources=n)

    def test_incomplete_index_rejected_before_embeddings(self):
        self.collection.total = 42
        with patch("src.rag_pipeline.retrieve_chunks") as retr:
            with self.assertRaisesRegex(ValueError, "expected 7194"):
                self.service.ask(QUESTION)
            retr.assert_not_called()

    def test_empty_retrieval_abstains_without_calling_generator(self):
        result, retr = self._ask_with(results=[])
        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(result["included_count"], 0)
        self.assertEqual(self.generated, [])
        retr.assert_called_once()

    def test_prompt_budget_shrinks_whole_passage_count(self):
        # Each passage is ~1500 characters. 3 selected passages do not fit
        # in 6000 conservative context units, but 1 or 2 may.
        self.service.num_ctx = 6000
        large = [fake_match(i, text=("Ibuprofen warning facts. " * 65)) for i in range(1, 6)]
        result, _ = self._ask_with(results=large)
        self.assertEqual(result["status"], "answered")
        self.assertLess(result["included_count"], 3)
        self.assertGreaterEqual(result["included_count"], 1)
        package, _ = self.generated[0]
        included = package["sources"]
        body = package["messages"][1]["content"]
        for passage in included:
            self.assertIn("Ibuprofen warning facts. " * 65, body)

    def test_single_evidence_chunk_too_large_fails_closed(self):
        self.service.num_ctx = 4096
        long_text = "Bleeding warning. " * 450
        result, _ = self._ask_with(results=[fake_match(1, text=long_text)])
        self.assertIn(result["status"], {"context_budget_exceeded", "context_budget_too_small"})
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(self.generated, [])

    def test_cross_drug_chunk_rejected_before_generation(self):
        wrong = fake_match(1, filename=ACET)
        with self.assertRaisesRegex(ValueError, "different drug PDF"):
            self._ask_with(results=[wrong])
        self.assertEqual(self.generated, [])

    def test_generator_citation_not_in_prompt_fails_closed(self):
        self.service.generator = lambda *args, **kw: {
            "status": "answered", "answer": "Bad [S99]", "citations": ["S99"],
            "usage": {"prompt_eval_count": 1},
        }
        with self.assertRaisesRegex(RuntimeError, "outside the evidence"):
            self._ask_with()

    def test_abstention_from_generator_keeps_evidence_as_uncited(self):
        self.service.generator = lambda *args, **kw: {
            "status": "abstained", "answer": ABSTENTION, "citations": [],
            "usage": {"prompt_eval_count": 100},
        }
        result, _ = self._ask_with()
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(result["status"], "abstained")
        self.assertFalse(any(s["used_in_answer"] for s in result["sources"]))

    def test_source_rows_use_only_selected_evidence_and_valid_urls(self):
        result, _ = self._ask_with()
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["sources"][0]["source_url"], URL)
        table = format_run(result)[2]
        self.assertEqual(len(table), 3)
        self.assertEqual(table[0][0], "[S1]")
        self.assertEqual(table[0][-1], "yes")
        self.assertEqual(table[1][-1], "no")

    def test_ui_format_has_explicit_medical_limitations(self):
        result, _ = self._ask_with()
        answer, status, table = format_run(result)
        self.assertIn("bleeding", answer)
        self.assertIn("not automatically fact-checked", status)
        self.assertEqual(table[0][0], "[S1]")

    def test_ui_startup_does_not_load_any_neural_models(self):
        if importlib.util.find_spec("gradio") is None:
            self.skipTest("gradio not installed; UI smoke test is optional")
        from scripts.app import build_app
        with patch("src.rag_pipeline.BGEEmbedder") as embedder, patch(
            "src.rag_pipeline.connect_collection") as vector_db:
            ui = build_app(service=self.service)
            self.assertIsNotNone(ui)
            embedder.assert_not_called()
            vector_db.assert_not_called()
            ui.close()

    def test_options_validation(self):
        for kwargs in ({"expected_count": 0}, {"num_ctx": 512},
                       {"num_ctx": 1024, "max_output_tokens": 900},
                       {"device": "gpu"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LocalDrugRAG(self.root, **kwargs)


if __name__ == "__main__":
    import unittest
    unittest.main()
