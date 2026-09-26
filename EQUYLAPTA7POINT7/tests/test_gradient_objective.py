"""tests/test_gradient_objective.py

Unit test for the two-branch functional effect analytical gradient engine.
"""
from __future__ import annotations

import unittest
from validation.gradient_check import verify_functional_effect_gradient


class TestGradientObjective(unittest.TestCase):
    def test_gradient_exactness(self):
        res = verify_functional_effect_gradient(epsilon=1e-4, tolerance=0.05)
        self.assertTrue(res["gradient_check_passed"])
        self.assertLess(res["max_relative_error"], 0.05)
        self.assertEqual(res["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
