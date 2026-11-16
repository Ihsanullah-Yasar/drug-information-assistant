"""Regression checks: false abstentions and abstention-only reference tags.

No real Ollama call. Tests exercise prompts/response parsing only.
"""
import unittest

from src.prompt_builder import ABSTENTION, SYSTEM_INSTRUCTIONS, build_prompt_package
from src.ollama_generator import _check_answer, validate_prompt_package

URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=example"
PDF = "ibuprofen_200mg_haleon_us_holdings_llc.pdf"


def sample_retrieval():
    text = ("Allergy alert: ibuprofen may cause a severe allergic reaction, including "
            "rash or wheezing. Stomach bleeding warning: This product may cause "
            "severe stomach bleeding.")
    return {
        "question": "What safety warnings are listed for ibuprofen?",
        "filename_filter": PDF,
        "top_k": 5,
        "matches": [{
            "rank": 1, "chunk_id": "sample-chunk-id", "text": text,
            "filename": PDF, "page_number": 2, "chunk_index": 1,
            "start_char": 0, "end_char": len(text),
            "source_url": URL, "cosine_distance": .25,
            "cosine_similarity": .75,
        }],
    }


class AnsweringFixTests(unittest.TestCase):
    def test_rebuilt_prompt_still_passes_integrity_checks(self):
        package = build_prompt_package(sample_retrieval())
        self.assertTrue(validate_prompt_package(package))
        self.assertEqual(package["sources"][0]["page_number"], 2)
        self.assertEqual(package["messages"][0]["role"], "system")
        self.assertIn('"source_id": "S1"', package["messages"][1]["content"])

    def test_prompt_encourages_supported_partial_answer(self):
        self.assertIn("even if the passages do not cover every possible answer", SYSTEM_INSTRUCTIONS)
        self.assertIn("ABSTAIN ONLY if no passage", SYSTEM_INSTRUCTIONS)
        self.assertIn("warn", SYSTEM_INSTRUCTIONS.lower())
        self.assertIn("without citations", SYSTEM_INSTRUCTIONS)
        self.assertIn("untrusted document data", SYSTEM_INSTRUCTIONS)

    def test_raw_real_user_response_is_pure_abstention_without_citations(self):
        output = ABSTENTION + " [S1], [S2], [S3]"
        status, reason, answer, ids = _check_answer(output, {"S1", "S2", "S3"})
        self.assertEqual(status, "abstained")
        self.assertEqual(answer, ABSTENTION)
        self.assertEqual(ids, [])
        self.assertIn("citations removed", reason)

    def test_abstention_standalone_unchanged(self):
        status, _reason, answer, ids = _check_answer(ABSTENTION, {"S1"})
        self.assertEqual((status, answer, ids), ("abstained", ABSTENTION, []))

    def test_abstention_with_unknown_citation_is_rejected(self):
        status, _reason, answer, ids = _check_answer(ABSTENTION + " [S99]", {"S1"})
        self.assertEqual(status, "rejected_unknown_citation")
        self.assertEqual(answer, ABSTENTION)
        self.assertEqual(ids, [])

    def test_abstention_followed_by_medical_claim_remains_rejected(self):
        for output in (ABSTENTION + " Ibuprofen causes X [S1].",
                       "Ibuprofen causes X [S1]. " + ABSTENTION,
                       ABSTENTION + " [S1] Do not take any medications."):
            with self.subTest(output=output):
                status, _reason, answer, ids = _check_answer(output, {"S1"})
                self.assertEqual(status, "rejected_mixed_abstention")
                self.assertEqual(answer, ABSTENTION)
                self.assertEqual(ids, [])

    def test_abstention_with_malformed_citation_not_accepted(self):
        status, _reason, answer, ids = _check_answer(ABSTENTION + " [S0]", {"S1"})
        self.assertEqual(status, "rejected_mixed_abstention")
        self.assertEqual((answer, ids), (ABSTENTION, []))

    def test_combined_citation_brackets_are_accepted(self):
        draft = (
            "The provided passages state that ampicillin is indicated for infections "
            "such as respiratory tract infections and bacterial meningitis [S1, S3]."
        )
        status, reason, answer, ids = _check_answer(draft, {"S1", "S2", "S3"})
        self.assertEqual(status, "answered")
        self.assertEqual(ids, ["S1", "S3"])
        self.assertIn("[S1]", answer)
        self.assertIn("[S3]", answer)
        self.assertNotIn("[S1, S3]", answer)
        self.assertIn("Citation IDs validated", reason)

    def test_supported_cited_answer_remains_accepted(self):
        result = _check_answer("The supplied passage warns of severe stomach bleeding [S1].", {"S1"})
        self.assertEqual(result[0], "answered")
        self.assertEqual(result[3], ["S1"])

    def test_uncited_factual_answer_remains_rejected(self):
        result = _check_answer("The supplied passage warns of stomach bleeding.", {"S1"})
        self.assertEqual(result[0], "rejected_uncited")


if __name__ == "__main__":
    unittest.main()
