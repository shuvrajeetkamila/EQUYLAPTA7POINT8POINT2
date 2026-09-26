"""tests/test_fdc_minimizer.py

Unit test for progressive inclusion search and minimal circuit discovery.
"""
from __future__ import annotations

import unittest
from circuits.circuit_minimizer import run_minimal_circuit_search


class TestFDCMinimizer(unittest.TestCase):
    def test_minimal_circuit_search(self):
        res = run_minimal_circuit_search("e4-math-4L", "math", seed=777)
        self.assertEqual(res["minimal_fdc_elements"], 4)
        self.assertIn("L0_head_2", res["minimal_fdc_nodes"])
        self.assertIn("L0_mlp", res["minimal_fdc_nodes"])
        # Check necessity of core components
        for node, n_res in res["necessity_evaluations"].items():
            self.assertTrue(n_res["is_necessary"])


if __name__ == "__main__":
    unittest.main()
