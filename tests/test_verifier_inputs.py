"""Regression checks for the standalone certificate acceptance boundary."""
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from verify_certificate import verify, CertificateError

ROOT = Path(__file__).resolve().parents[1]

class VerifierInputTests(unittest.TestCase):
    def setUp(self):
        cases = json.loads((ROOT / 'results/certificates.json').read_text())
        self.case = next(copy.deepcopy(c) for c in cases if c['selected'])

    def test_index_validation(self):
        for field, roots in [('i', 'zeros'), ('j', 'poles')]:
            for invalid in [-1, len(self.case[roots]), True, 0.0, '0']:
                case = copy.deepcopy(self.case)
                case['selected'][0][field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(CertificateError):
                        verify([case])

    def test_empty_circle_list_rejected(self):
        self.case['circles'] = []
        with self.assertRaises(CertificateError):
            verify([self.case])

    def test_duplicate_edge_rejected(self):
        self.case['selected'].append(copy.deepcopy(self.case['selected'][0]))
        with self.assertRaises(CertificateError):
            verify([self.case])

    def test_validation_survives_optimized_python(self):
        self.case['selected'][0]['epsilon'] = '0'
        program = ('import json,sys; from verify_certificate import verify; '
                   'verify(json.loads(sys.stdin.read()))')
        result = subprocess.run([sys.executable, '-O', '-c', program],
            input=json.dumps([self.case]), text=True, capture_output=True, cwd=ROOT)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CertificateError', result.stderr)

if __name__ == '__main__':
    unittest.main()
