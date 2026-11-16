"""Phase 1 UI tests; no Ollama, embeddings, Chroma, or network required."""
from __future__ import annotations

from pathlib import Path
import csv
import importlib.util
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from src.chat_ui import (
    AUTO_LABEL, WAITING, start_turn, check_label_scope,
    append_result, append_error, reset_chat, sources_message,
)

IBU = "ibuprofen_200mg_haleon_us_holdings_llc.pdf"
ACET = "acetaminophen_325mg_health_pharma.pdf"
URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc"


class ChatUiTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = self.root / "data"
        data.mkdir()
        with (data / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["filename"])
            w.writerow([IBU])
            w.writerow([ACET])
        (data / "target_drugs.txt").write_text("ibuprofen\nacetaminophen\n", encoding="utf-8")
        self.svc = Mock()
        self.svc.sources_csv = data / "sources.csv"
        self.svc.targets_txt = data / "target_drugs.txt"

    def test_new_user_turn_and_pending_are_visible(self):
        cleared, chat, pending = start_turn(" What are ibuprofen warnings? ", [])
        self.assertEqual(cleared, "")
        self.assertEqual(pending, "What are ibuprofen warnings?")
        self.assertEqual([m["role"] for m in chat], ["user", "assistant"])
        self.assertEqual(chat[1]["content"], WAITING)

    def test_gradio6_content_blocks_still_update_pending(self):
        """Gradio 6 Chatbot preprocess returns content as [{type,text}] blocks."""
        from src.chat_ui import apply_stream_event, stream_chat_turn

        graded = [
            {"role": "user", "content": [{"type": "text", "text": "what fluoxetine is used for ?"}],
             "metadata": None, "options": None},
            {"role": "assistant", "content": [{"type": "text", "text": WAITING}],
             "metadata": None, "options": None},
        ]
        updated = apply_stream_event(
            graded, {"type": "status", "message": "Searching the selected drug label…"}
        )
        self.assertTrue(updated[-1]["content"].startswith("**Working** · "))

        class FakeService:
            sources_csv = None
            targets_txt = None

            def ask_stream(self, question, **kwargs):
                yield {"type": "status", "message": "Searching"}
                yield {"type": "final", "result": {
                    "status": "answered", "answer": "Used for MDD [S1]", "sources": [],
                }}

        outs = list(stream_chat_turn(
            graded, "what fluoxetine is used for ?", AUTO_LABEL, 3, FakeService(),
        ))
        self.assertEqual(outs[-1][-1]["content"], "Used for MDD [S1]")

    def test_blank_does_not_generate_turn(self):
        existing = [{"role": "user", "content": "Hi"}]
        cleared, chat, pending = start_turn("  ", existing)
        self.assertEqual(pending, "")
        self.assertEqual(chat, existing)
        self.assertIsNot(chat, existing)

    def test_long_question_uses_safe_error(self):
        _, chat, pending = start_turn("X" * 1001, [])
        self.assertEqual(pending, "")
        self.assertIn("shorten", chat[-1]["content"])
        self.assertEqual(len(chat[0]["content"]), 1001)

    def test_selected_pdf_detects_conflict(self):
        with self.assertRaisesRegex(ValueError, "different drug"):
            check_label_scope("What warnings does ibuprofen have?", ACET, self.svc)
        self.assertEqual(check_label_scope("What warnings does ibuprofen have?", IBU, self.svc), IBU)

    def test_two_drugs_are_not_combined_even_with_manual_selection(self):
        with self.assertRaisesRegex(ValueError, "Multiple drugs"):
            check_label_scope("Compare ibuprofen and acetaminophen", IBU, self.svc)

    def test_auto_scope_is_delegated_to_backend(self):
        self.assertIsNone(check_label_scope("What warnings does ibuprofen have?", AUTO_LABEL, self.svc))
        self.assertIsNone(check_label_scope("What warnings apply?", AUTO_LABEL, self.svc))

    def test_source_panel_compact_closed_and_has_citations(self):
        run = {
            "filename": IBU, "retrieved_count": 5, "included_count": 2,
            "sources": [
                {"citation": "[S1]", "page": 2, "excerpt": "Bleeding warning.",
                 "source_url": URL, "used_in_answer": True},
                {"citation": "[S2]", "page": 3, "excerpt": "Allergy warning.",
                 "source_url": URL, "used_in_answer": False},
            ],
        }
        m = sources_message(run)
        self.assertEqual(m["role"], "assistant")
        self.assertEqual(m["metadata"]["status"], "done")
        self.assertIn("2 passages", m["metadata"]["title"])
        self.assertIn("DailyMed", m["content"])
        self.assertIn("Not cited", m["content"])

    def test_untrusted_source_url_never_becomes_link(self):
        run = {"sources": [{"citation": "[S1]", "page": 1, "excerpt": "x",
                             "source_url": "javascript:alert(1)", "used_in_answer": True}]}
        m = sources_message(run)
        self.assertIn("Source link unavailable", m["content"])
        self.assertNotIn("javascript:", m["content"])

    def test_replaces_pending_and_attaches_sources(self):
        _, chat, _ = start_turn("What about ibuprofen?", [])
        result = {"status": "answered", "answer": "Stomach bleeding [S1].", "sources": [
            {"citation": "[S1]", "page": 2, "excerpt": "Bleeding.",
             "source_url": URL, "used_in_answer": True}
        ]}
        finished = append_result(chat, result)
        self.assertEqual(finished[1]["content"], "Stomach bleeding [S1].")
        self.assertEqual(len(finished), 3)
        self.assertIn("Sources", finished[-1]["metadata"]["title"])
        self.assertEqual(chat[-1]["content"], WAITING)  # immutable update

    def test_rejected_answer_keeps_abstention_with_status(self):
        _, chat, _ = start_turn("Question", [])
        finished = append_result(chat, {"status": "no_evidence",
                                        "answer": "I don't have enough information in the provided documents.",
                                        "reason": "No evidence available", "sources": []})
        self.assertIn("No evidence available", finished[-1]["content"])
        self.assertEqual(len(finished), 2)

    def test_network_error_does_not_show_untrusted_response(self):
        _, chat, _ = start_turn("Question", [])
        result = append_error(chat, RuntimeError("secret response full of unrelated medical advice"))
        self.assertIn("Unable to answer safely", result[-1]["content"])
        self.assertNotIn("secret response", result[-1]["content"])

    def test_oom_error_shows_memory_hint_without_raw_dump(self):
        _, chat, _ = start_turn("Question", [])
        result = append_error(chat, RuntimeError(
            'Ollama HTTP 500: {"error":"llama-server reported out-of-memory during startup"}'
        ))
        content = result[-1]["content"]
        self.assertIn("Unable to answer safely", content)
        self.assertIn("out of memory", content.lower())
        self.assertNotIn("llama-server reported", content)

    def test_reset_clears_session(self):
        self.assertEqual(reset_chat(), ("", [], "", AUTO_LABEL, 3))

    def test_ui_build_does_not_load_models(self):
        if importlib.util.find_spec("gradio") is None:
            self.skipTest("Optional Gradio smoke test")
        from scripts.app import build_app
        # RAG service is mocked; no GPU or Ollama required.
        self.svc.ask = Mock()
        ui = build_app(service=self.svc)
        self.assertIsNotNone(ui)
        self.svc.ask.assert_not_called()
        ui.close()

    def test_multi_turn_chat_history_preserved(self):
        _, h1, _ = start_turn("First question", [])
        h2 = append_result(h1, {"status": "answered", "answer": "First answer [S1]", "sources": []})
        _, h3, pending = start_turn("Second question", h2)
        self.assertEqual(len(h3), 4)
        self.assertEqual(pending, "Second question")
        self.assertEqual(h3[1]["content"], "First answer [S1]")


if __name__ == "__main__":
    import unittest
    unittest.main()
