import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
import pandas as pd
from lib.tggates_metadata import attach_pathology_findings
from lib.cluster_analysis import control_group_by_dose, control_group_by_finding


class LabelIntegrityTests(unittest.TestCase):
    def test_unknown_is_not_a_negative(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pathology.csv'
            pd.DataFrame([dict(EXP_ID='01', GROUP_ID='01', INDIVIDUAL_ID='1', ORGAN='Liver',
                               FINDING_TYPE='Necrosis', TOPOGRAPHY_TYPE='Hepatocyte',
                               GRADE_TYPE='slight', SP_FLG='false')]).to_csv(p, index=False)
            manifest = pd.DataFrame(dict(slide_id=['a', 'b', 'c'], exp_id=[1,1,2],
                group_id=[1,1,1], individual_id=[1,2,1], dose_level=['High','Control',None],
                individual_metadata_matched=[True,True,False]))
            out = attach_pathology_findings(manifest, p)
            self.assertTrue(out.has_finding.iloc[0])
            self.assertFalse(out.has_finding.iloc[1])
            self.assertTrue(pd.isna(out.has_finding.iloc[2]))
            self.assertEqual(out.pathology_label_status.tolist(), ['recorded_finding','no_finding_recorded','unknown'])
            with self.assertRaises(ValueError):
                control_group_by_dose(out)
            with self.assertRaises(ValueError):
                control_group_by_finding(out)
            self.assertEqual(control_group_by_dose(out.iloc[:2]).tolist(), [False, True])

    def test_legacy_missing_dose_is_rejected(self):
        m = pd.DataFrame(dict(dose_level=['Control', None], has_finding=[False, False]))
        with self.assertRaises(ValueError):
            control_group_by_dose(m)

    def test_partial_review_cannot_erase_existing_judgments(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'review.csv'
            original = b'cluster_id,visual_finding\n147,yes\n104,no\n'
            path.write_bytes(original)
            result = subprocess.run([sys.executable, str(root / 'scripts/exemplars_review.py'),
                                     '--out', d, '--limit', '1', '--seed', '99'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Review output already exists', result.stderr)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
