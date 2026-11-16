"""Tests for active-moiety matching of a hydrated drug (no network)."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'collect_labels.py'
spec = importlib.util.spec_from_file_location('collector_ingredient_patch', PATH)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

SETID = '5ee63ebc-45e8-fab8-e063-6294a90a7b23'


def xml_with_active_moiety(actual='AZITHROMYCIN DIHYDRATE', moiety='AZITHROMYCIN ANHYDROUS'):
    moiety_xml = (f'<activeMoiety><activeMoiety><name>{moiety}</name></activeMoiety></activeMoiety>'
                  if moiety is not None else '')
    xml = f"""<document xmlns="urn:hl7-org:v3">
      <code displayName="HUMAN PRESCRIPTION DRUG LABEL"/><setId root="{SETID}"/>
      <author><assignedEntity><representedOrganization><name>Example Pharmacy</name>
      </representedOrganization></assignedEntity></author>
      <subject><manufacturedProduct><routeCode displayName="ORAL"/>
        <manufacturedProduct><formCode displayName="TABLET, FILM COATED"/>
          <ingredient classCode="ACTIB">
            <quantity><numerator value="250" unit="mg"/><denominator value="1" unit="1"/></quantity>
            <ingredientSubstance><name>{actual}</name>{moiety_xml}</ingredientSubstance>
          </ingredient>
        </manufacturedProduct>
      </manufacturedProduct></subject>
    </document>"""
    return ET.fromstring(xml)


class ActiveMoietyMatchingTests(unittest.TestCase):
    def test_accepts_azithromycin_anhydrous_moiety_in_spl(self):
        label = collector.verify_label(xml_with_active_moiety(), {'setid': SETID}, 'azithromycin')
        self.assertEqual(label['Ingredient'], 'AZITHROMYCIN DIHYDRATE')
        self.assertEqual(label['Evidence'][0]['active_moieties'], ['AZITHROMYCIN ANHYDROUS'])
        self.assertTrue(label['Oral tablet'])
        self.assertEqual(label['Strengths'][0]['amount'], '250')

    def test_rejects_hydrate_name_without_structured_active_moiety(self):
        with self.assertRaisesRegex(ValueError, 'not verified'):
            collector.verify_label(xml_with_active_moiety(moiety=None), {'setid': SETID}, 'azithromycin')

    def test_rejects_unrelated_active_moiety(self):
        with self.assertRaisesRegex(ValueError, 'not verified'):
            collector.verify_label(xml_with_active_moiety(moiety='ERYTHROMYCIN'), {'setid': SETID}, 'azithromycin')

    def test_exact_ingredient_remains_accepted(self):
        label = collector.verify_label(xml_with_active_moiety(actual='AZITHROMYCIN', moiety=None), {'setid': SETID}, 'azithromycin')
        self.assertEqual(label['Ingredient'], 'AZITHROMYCIN')

    def test_saved_state_check_uses_same_moiety_rule(self):
        label = collector.verify_label(xml_with_active_moiety(), {'setid': SETID}, 'azithromycin')
        label['Product name'] = 'AZITHROMYCIN DIHYDRATE tablet'
        label['Label updated on'] = 'November 1, 2026'
        filename = collector.build_pdf_filename(label)
        row = collector.source_row(label, filename)
        record = {'set_id': SETID, 'filename': filename, 'label': label,
                  'source_row': row, 'status': 'planned'}
        collector.check_record('azithromycin', record)

    def test_unsupported_salt_not_implicitly_equivalent(self):
        self.assertFalse(collector.matches_structured_ingredient('azithromycin', 'AZITHROMYCIN SODIUM', []))
        self.assertFalse(collector.matches_structured_ingredient('azithromycin', 'AZITHROMYCIN DIHYDRATE', []))

if __name__ == '__main__':
    unittest.main()
