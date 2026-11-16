"""Offline tests for deterministic drug-aware question routing."""
from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.drug_router import named_drugs, resolve_drug_filename, _catalog


class DrugRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / "data"
        root.mkdir()
        self.csv = root / "sources.csv"
        self.txt = root / "target_drugs.txt"
        self.names = [
            "ibuprofen_200mg_haleon_us_holdings_llc.pdf",
            "acetaminophen_325mg_health_pharma.pdf",
            "valacyclovir_hydrochloride_500mg_nucare_pharmaceuticals_inc.pdf",
            "penicillin_v_potassium_500mg_remedyrepack_inc.pdf",
            "aspirin_81mg_sample.pdf",
            "diclofenac_sodium_100mg_nucare.pdf",
        ]
        self.txt.write_text("ibuprofen\nacetaminophen\nvalacyclovir\npenicillin v\naspirin\ndiclofenac\n", encoding="utf-8")
        with self.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["filename", "product_name", "source_url"])
            writer.writeheader()
            for name in self.names:
                writer.writerow({"filename": name, "product_name": "sample", "source_url": "https://dailymed.nlm.nih.gov/"})

    def route(self, question, **kwargs):
        return resolve_drug_filename(question, sources_csv=self.csv,
                                     targets_txt=self.txt, **kwargs)

    def test_ibuprofen_named_question_auto_filters(self):
        self.assertEqual(self.route("What are the adverse reactions of Ibuprofen?"), self.names[0])

    def test_medical_salt_filename_maps_to_active_ingredient(self):
        self.assertEqual(self.route("What does VALACYCLOVIR do?"), self.names[2])

    def test_multiword_drug_name_matching(self):
        self.assertEqual(self.route("What are penicillin V warnings?"), self.names[3])

    def test_longest_match_wins(self):
        cat = {("penicillin",): ("a.pdf",), ("penicillin", "v"): ("b.pdf",)}
        self.assertEqual(named_drugs("About penicillin v", cat), [("penicillin", "v")])

    def test_does_not_fuzzy_match_another_drug(self):
        self.assertIsNone(self.route("What is a recommended antibiotic?"))
        self.assertIsNone(self.route("What about ibuprofenlike names?"))

    def test_multiple_named_drugs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Multiple drugs"):
            self.route("Compare adverse reactions of ibuprofen and aspirin")

    def test_explicit_filename_overrides_automatic(self):
        self.assertEqual(self.route("How does aspirin interact with ibuprofen?",
                                    filename=self.names[0]), self.names[0])

    def test_all_drugs_opt_out(self):
        self.assertIsNone(self.route("ibuprofen and aspirin", all_drugs=True))
        with self.assertRaisesRegex(ValueError, "either"):
            self.route("ibuprofen", all_drugs=True, filename=self.names[0])

    def test_missing_or_ambiguous_source_fails_closed(self):
        self.txt.write_text(self.txt.read_text() + "notindataset\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "0 source PDFs"):
            self.route("Tell me about notindataset")
        with self.csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["ibuprofen_600mg_other.pdf", "sample", "https://dailymed.nlm.nih.gov/"])
        with self.assertRaisesRegex(ValueError, "2 source PDFs"):
            self.route("What are ibuprofen warnings?")

    def test_invalid_duplicate_csv_rejected(self):
        with self.csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([self.names[0], "sample", "https://dailymed.nlm.nih.gov/"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.route("ibuprofen")

    def test_missing_catalog_explained(self):
        self.csv.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "sources.csv"):
            self.route("ibuprofen")
        # Explicit filename still works without the local router catalog.
        self.assertEqual(self.route("ibuprofen", filename=self.names[0]), self.names[0])

    def test_cli_wires_auto_resolution_before_model_load(self):
        from scripts import retrieve as cli
        with patch.object(cli, "PROJECT_ROOT", Path(self.temp.name)), \
             patch.object(cli, "run_query", return_value=[] ) as query:
            import sys
            with patch.object(sys, "argv", ["retrieve", "--question", "ibuprofen side effects", "--require-count", "7194"]):
                self.assertEqual(cli.main(), 0)
            self.assertEqual(query.call_args.kwargs["filename"], self.names[0])
            with patch.object(sys, "argv", ["retrieve", "--question", "ibuprofen side effects", "--all-drugs"]):
                self.assertEqual(cli.main(), 0)
            self.assertIsNone(query.call_args.kwargs["filename"])


if __name__ == "__main__":
    unittest.main()
