"""test_e7_audit.py — Unit & Integration tests for EQUYLAPTA7."""
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(ROOT)


class TestEquylapta7Audit(unittest.TestCase):

    def test_results_json_structure(self):
        rpath = os.path.join(WORKSPACE, "results.json")
        self.assertTrue(os.path.exists(rpath))
        with open(rpath) as f:
            res = json.load(f)
        self.assertEqual(res["milestone"], "EQUYLAPTA7")
        self.assertIn("hierarchical_discovery", res)
        self.assertIn("causal_discovery", res)
        self.assertIn("functional_signature", res)
        self.assertIn("three_reconstruction_methods", res)
        self.assertIn("target_controls", res)
        self.assertIn("independent_reconstructions", res)
        self.assertIn("depth_sweep", res)
        self.assertIn("twenty_questions", res)

    def test_semantic_validator_passes(self):
        from semantic_report_validator import run_semantic_validation
        self.assertTrue(run_semantic_validation())

    def test_report_integrity_passes(self):
        from report_integrity_check import verify_report_integrity
        rpath = os.path.join(WORKSPACE, "results.json")
        with open(rpath) as f:
            res = json.load(f)
        self.assertTrue(verify_report_integrity(res))

    def test_self_check_passes(self):
        from self_check import run_full_self_check
        self.assertTrue(run_full_self_check())

    def test_causal_discovery_validity(self):
        rpath = os.path.join(WORKSPACE, "results.json")
        with open(rpath) as f:
            res = json.load(f)
        causal = res["causal_discovery"]
        # Ablation must drop performance
        self.assertGreater(causal["ablation"]["causal_drop"], 10.0)
        # Restoration must recover
        self.assertAlmostEqual(causal["restoration"]["mean"], causal["baseline"]["mean"], places=1)
        # Specificity ratio > 1.0
        self.assertGreater(causal["specificity_ratio"], 1.0)

    def test_twenty_questions_schema(self):
        tq_path = os.path.join(WORKSPACE, "twenty_questions_generated.json")
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

    def test_workspace_size_compliance(self):
        from generate_size_report import get_dir_size
        ws_mb = get_dir_size(WORKSPACE) / (1024 * 1024)
        self.assertLess(ws_mb, 120.0)
        self.assertLess(ws_mb, 55.0)


if __name__ == "__main__":
    unittest.main()
