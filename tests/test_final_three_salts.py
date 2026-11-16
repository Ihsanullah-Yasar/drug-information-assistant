"""Offline regression tests for three exceptional DailyMed ingredient forms."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_labels.py"
spec = importlib.util.spec_from_file_location("collector_salt_patch", PATH)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

CASES = [
    ("valacyclovir", "VALACYCLOVIR HYDROCHLORIDE", "ACYCLOVIR"),
    ("oseltamivir", "OSELTAMIVIR PHOSPHATE", "OSELTAMIVIR ACID"),
    ("enalapril", "ENALAPRIL MALEATE", "ENALAPRILAT ANHYDROUS"),
]
SET_ID = "5d6a5880-07c8-b077-e063-6394a90ad302"


def make_spl(ingredient, moiety, *, combination=False, category="HUMAN PRESCRIPTION DRUG LABEL"):
    moiety_fragment = (f"<activeMoiety><activeMoiety><name>{moiety}</name>"
                       f"</activeMoiety></activeMoiety>" if moiety else "")
    active = f'''<ingredient classCode="ACTIB"><quantity><numerator value="500" unit="mg"/>
       <denominator value="1" unit="1"/></quantity>
       <ingredientSubstance><name>{ingredient}</name>{moiety_fragment}</ingredientSubstance>
       </ingredient>'''
    second = '''<ingredient classCode="ACTIB"><quantity><numerator value="15" unit="mg"/>
       <denominator value="1" unit="1"/></quantity>
       <ingredientSubstance><name>OTHER INGREDIENT</name></ingredientSubstance>
       </ingredient>''' if combination else ""
    return ET.fromstring(f'''<document xmlns="urn:hl7-org:v3">
      <code displayName="{category}"/><setId root="{SET_ID}"/>
      <author><assignedEntity><representedOrganization><name>Example Pharmacy</name>
      </representedOrganization></assignedEntity></author>
      <subject><manufacturedProduct><routeCode displayName="ORAL"/>
        <manufacturedProduct><formCode displayName="TABLET"/>{active}{second}
        </manufacturedProduct></manufacturedProduct></subject>
      </document>''')


class FinalThreeIngredientTests(unittest.TestCase):
    def test_accepts_all_three_exact_reviewed_pairs(self):
        for target, ingredient, moiety in CASES:
            with self.subTest(target=target):
                self.assertTrue(collector.matches_structured_ingredient(target, ingredient, [moiety]))

    def test_verifies_full_spl_for_each_reviewed_pair(self):
        for target, ingredient, moiety in CASES:
            with self.subTest(target=target):
                label = collector.verify_label(make_spl(ingredient, moiety),
                                               {"setid": SET_ID}, target)
                self.assertEqual(label["Ingredient"], ingredient)
                self.assertEqual(label["Evidence"][0]["active_moieties"], [moiety])
                self.assertEqual(label["Strengths"][0]["amount"], "500")

    def test_saved_state_uses_same_rule(self):
        for target, ingredient, moiety in CASES:
            with self.subTest(target=target):
                label = collector.verify_label(make_spl(ingredient, moiety),
                                               {"setid": SET_ID}, target)
                label["Product name"] = ingredient + " tablet"
                label["Label updated on"] = "November 5, 2026"
                file_name = collector.build_pdf_filename(label)
                record = {"set_id": SET_ID, "filename": file_name, "label": label,
                          "source_row": collector.source_row(label, file_name),
                          "status": "planned"}
                collector.check_record(target, record)

    def test_missing_moiety_rejected(self):
        for target, ingredient, _ in CASES:
            with self.subTest(target=target):
                self.assertFalse(collector.matches_structured_ingredient(target, ingredient, []))

    def test_wrong_moiety_rejected(self):
        for target, ingredient, _ in CASES:
            with self.subTest(target=target):
                self.assertFalse(collector.matches_structured_ingredient(target, ingredient,
                                                                          ["UNRELATED DRUG"]))

    def test_extra_unexpected_moiety_rejected(self):
        for target, ingredient, moiety in CASES:
            with self.subTest(target=target):
                self.assertFalse(collector.matches_structured_ingredient(
                    target, ingredient, [moiety, "UNRELATED DRUG"]))

    def test_other_salt_suffix_not_implicitly_accepted(self):
        self.assertFalse(collector.matches_structured_ingredient(
            "enalapril", "ENALAPRIL HYDROCHLORIDE", ["ENALAPRILAT ANHYDROUS"]))
        self.assertFalse(collector.matches_structured_ingredient(
            "oseltamivir", "OSELTAMIVIR SODIUM", ["OSELTAMIVIR ACID"]))
        self.assertFalse(collector.matches_structured_ingredient(
            "valacyclovir", "VALACYCLOVIR PHOSPHATE", ["ACYCLOVIR"]))

    def test_combination_drug_still_rejected(self):
        target, ingredient, moiety = CASES[0]
        with self.assertRaisesRegex(ValueError, "exactly one active ingredient"):
            collector.verify_label(make_spl(ingredient, moiety, combination=True),
                                   {"setid": SET_ID}, target)

    def test_nonhuman_category_still_rejected(self):
        target, ingredient, moiety = CASES[0]
        with self.assertRaisesRegex(ValueError, "human drug label category"):
            collector.verify_label(make_spl(ingredient, moiety, category="VETERINARY DRUG LABEL"),
                                   {"setid": SET_ID}, target)

    def test_case_and_whitespace_normalization(self):
        self.assertTrue(collector.matches_structured_ingredient(
            "Enalapril", " ENALAPRIL   MALEATE ", [" ENALAPRILAT  ANHYDROUS "]))


if __name__ == "__main__":
    unittest.main()
