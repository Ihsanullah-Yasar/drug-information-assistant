"""Offline tests for the collection stage. Run: python -m unittest discover -s tests -v"""
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pymupdf

# The test lives in tests/, while the implementation lives in scripts/.
MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_labels.py"
spec = importlib.util.spec_from_file_location("collect_labels", MODULE_PATH)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def sample_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Drug label for education: sample text")
    data = doc.tobytes()
    doc.close()
    return data


def make_label(name: str, set_id: str, filename: str) -> dict:
    return {
        "Ingredient": name.upper(),
        "Strength": "500 mg per 1 tablet",
        "Strengths": [{"amount": "500", "unit": "mg", "basis_amount": "1", "basis_unit": "1", "form": "TABLET"}],
        "Packager": "Example Pharmaceutical",
        "Set ID": set_id,
        "Source page URL": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",
        "Exact product title (DailyMed API)": f"{name.upper()} tablet",
        "Verified target": name,
        "Category": "HUMAN OTC DRUG LABEL",
        "Evidence": [{"ingredient": name.upper(), "active_moieties": [], "form": "TABLET", "routes": ["ORAL"]}],
        "Oral tablet": True,
        "Verification": "test SPL verification",
        "Label updated on": "November 3, 2026",
        "Product name": f"{name.upper()} tablet",
    }


class CollectorTests(unittest.TestCase):
    def test_filename_sorted_decimals_and_windows_safe(self):
        value = collector.build_pdf_filename({
            "Ingredient": "Example Drug",
            "Packager": "Example, LLC",
            "Strength per tablet": "10 mg per tablet; 2.5 mg per tablet",
        })
        self.assertEqual(value, "example_drug_2.5mg_10mg_example_llc.pdf")
        for bad in ("../evil.pdf", "..\\evil.pdf", "CON.pdf", "data.csv", "x?.pdf"):
            with self.assertRaises(ValueError, msg=bad):
                collector.local_path(Path("/tmp"), bad)

    def test_pdf_validation(self):
        self.assertEqual(collector.pdf_page_count(sample_pdf()), 1)
        with self.assertRaises(ValueError):
            collector.pdf_page_count(b"not a pdf")
        blank = pymupdf.open()
        blank.new_page()
        with self.assertRaises(ValueError):
            collector.pdf_page_count(blank.tobytes())
        blank.close()

    def test_source_csv_quotes_and_idempotency(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            label = make_label("acetaminophen", collector.REVIEWED["acetaminophen"][0], "acetaminophen_325mg_health_pharma.pdf")
            label["Packager"] = 'Example, "Medical" LLC'
            row = collector.source_row(label, "acetaminophen_325mg_health_pharma.pdf")
            collector.register_source(root, row)
            collector.register_source(root, row)
            self.assertEqual(collector.read_sources(root), [row])
            self.assertEqual(len((root / "data/sources.csv").read_text().splitlines()), 2)

    def test_single_active_xml_validation(self):
        set_id = collector.REVIEWED["acetaminophen"][0]
        xml = f'''<document xmlns="urn:hl7-org:v3">
          <code displayName="HUMAN OTC DRUG LABEL"/><setId root="{set_id}"/>
          <author><assignedEntity><representedOrganization><name>Example LLC</name></representedOrganization></assignedEntity></author>
          <subject><manufacturedProduct><routeCode displayName="ORAL"/>
            <manufacturedProduct><formCode displayName="TABLET"/>
              <ingredient classCode="ACTIB"><quantity><numerator value="325" unit="mg"/>
                <denominator value="1" unit="1"/></quantity>
                <ingredientSubstance><name>ACETAMINOPHEN</name></ingredientSubstance>
              </ingredient>
            </manufacturedProduct>
          </manufacturedProduct></subject>
        </document>'''
        root = ET.fromstring(xml)
        label = collector.verify_label(root, {"setid": set_id, "title": "ACETAMINOPHEN tablet"}, "acetaminophen")
        self.assertEqual(label["Set ID"], set_id)
        self.assertTrue(label["Oral tablet"])
        self.assertEqual(label["Strengths"][0]["amount"], "325")
        with self.assertRaises(ValueError):
            collector.verify_label(root, {"setid": set_id}, "ibuprofen")

    def test_resume_three_existing_pdfs_and_add_missing_naproxen_csv(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            docs = root / "data" / "documents"
            docs.mkdir(parents=True)
            targets = ["acetaminophen", "ibuprofen", "naproxen"]
            (root / "data" / "target_drugs.txt").write_text("\n".join(targets) + "\n", encoding="utf-8")
            labels = {}
            for target in targets:
                set_id, filename = collector.REVIEWED[target]
                (docs / filename).write_bytes(sample_pdf())
                labels[target] = make_label(target, set_id, filename)
            for target in targets[:2]:
                collector.register_source(root, collector.source_row(labels[target], collector.REVIEWED[target][1]))

            original_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in docs.glob("*.pdf")}
            with patch.object(collector, "label_by_set_id", side_effect=lambda sid, target: labels[target]):
                self.assertEqual(collector.run_batch(root, expected_count=3), 0)
                self.assertEqual(collector.run_batch(root, expected_count=3), 0)
            self.assertEqual(len(collector.read_sources(root)), 3)
            self.assertEqual(original_files, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in docs.glob("*.pdf")})
            state = collector.load_state(root)
            self.assertEqual(len(state["downloads"]), 3)
            self.assertEqual(len(state["targets"]), 3)
            audit = collector.audit_collection(targets, root, state)
            self.assertEqual(audit["verified_targets"], 3)
            self.assertEqual(audit["matching_csv_rows"], 3)
            self.assertEqual(audit["distinct_set_ids"], 3)
            self.assertEqual(audit["total_pdf_pages"], 3)

    def test_audit_only_does_not_change_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "data").mkdir()
            (root / "data/target_drugs.txt").write_text("ibuprofen\n")
            self.assertEqual(collector.run_batch(root, expected_count=1, audit_only=True), 1)
            self.assertFalse((root / collector.STATE_FILE).exists())

if __name__ == "__main__":
    unittest.main()
