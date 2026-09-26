"""tests/test_gradient_check.py

Unit test for two-branch analytical gradient verification against finite differences.
"""
from __future__ import annotations

import unittest
from validation.gradient_check import verify_functional_effect_gradient


class TestGradientCheck(unittest.TestCase):
    def test_gradient_verification(self):
        res = verify_functional_effect_gradient(epsilon=1e-4, tolerance=0.05)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["gradient_check_passed"])
        self.assertLess(res["max_relative_error"], 0.05)


if __name__ == "__main__":
    unittest.main()
