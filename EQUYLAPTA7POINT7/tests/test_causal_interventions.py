"""tests/test_causal_interventions.py

Unit test for surgical ablation and restoration operations.
"""
from __future__ import annotations

import unittest
from discovery.causal_localization import evaluate_native_localization


class TestCausalInterventions(unittest.TestCase):
    def test_native_math_ablation(self):
        res = evaluate_native_localization("e4-math-4L", "math", (0, 2), discovery_seed=777)
        self.assertEqual(res["status"], "LOCALIZED")
        self.assertGreaterEqual(res["causal_drop_pp"], 20.0)
        self.assertLessEqual(res["restoration_error_pp"], 1e-4)
        self.assertGreaterEqual(res["specificity_ratio"], 1.2)


if __name__ == "__main__":
    unittest.main()
