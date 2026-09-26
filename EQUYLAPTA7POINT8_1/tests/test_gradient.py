"""tests/test_gradient.py

Unit test for true two-branch training gradient against finite differences.
"""
from __future__ import annotations

import unittest
from src.gradient_validation import run_training_gradient_validation


class TestTrainingGradient(unittest.TestCase):
    def test_gradient_validation(self):
        res = run_training_gradient_validation(epsilon=1e-2, tolerance=0.05)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["gradient_check_passed"])
        self.assertLess(res["max_relative_error"], 0.05)
        self.assertGreater(res["total_parameters_tested"], 5)


if __name__ == "__main__":
    unittest.main()
