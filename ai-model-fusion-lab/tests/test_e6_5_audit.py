"""test_e6_5_audit.py — Unit & Integration tests for EQUYLAPTA6.5."""
import json
import os
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestEquylapta65Audit(unittest.TestCase):

    def test_linear_cka_properties(self):
        from demo.run_equylapta6_5 import linear_cka
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 64))
        # Self-CKA must be 1.0
        self.assertAlmostEqual(linear_cka(X, X), 1.0, places=3)
        # Scaled matrix CKA must be 1.0 (scale invariance)
        self.assertAlmostEqual(linear_cka(X, 5.0 * X), 1.0, places=3)

    def test_report_integrity_passes(self):
        from report_integrity_check import verify_report_integrity
        results_path = os.path.join(ROOT, "results.json")
        self.assertTrue(os.path.exists(results_path))
        with open(results_path) as f:
            res = json.load(f)
        self.assertTrue(verify_report_integrity(res))

    def test_twenty_questions_schema(self):
        tq_path = os.path.join(ROOT, "twenty_questions_generated.json")
        self.assertTrue(os.path.exists(tq_path))
        with open(tq_path) as f:
            tq = json.load(f)
        self.assertEqual(len(tq), 20)
        for i in range(1, 21):
            qid = f"Q{i:02d}"
            self.assertIn(qid, tq)
            self.assertIn("question", tq[qid])
            self.assertIn("answer", tq[qid])
            self.assertIn("source_keys", tq[qid])
            self.assertIn("status", tq[qid])
            self.assertGreater(len(tq[qid]["source_keys"]), 0)

    def test_semantic_validation_passes(self):
        from semantic_report_validator import run_semantic_validation
        self.assertTrue(run_semantic_validation())

    def test_artifact_size_compliance(self):
        ws = os.path.dirname(ROOT)
        zip_candidates = [
            os.path.join(ws, "EQUYLAPTA6POINT5_CORRECTED.zip"),
            os.path.join(ws, "EQUYLAPTA6_5.zip")
        ]
        found = False
        for zp in zip_candidates:
            if os.path.exists(zp):
                found = True
                size_mb = os.path.getsize(zp) / (1024 * 1024)
                self.assertLess(size_mb, 120.0)
                self.assertLess(size_mb, 55.0)
        if not found:
            from generate_size_report import get_dir_size
            raw_mb = (get_dir_size(ROOT) + get_dir_size(os.path.join(ws, "fusionlab_data"))) / (1024 * 1024)
            self.assertLess(raw_mb, 120.0)


if __name__ == "__main__":
    unittest.main()
