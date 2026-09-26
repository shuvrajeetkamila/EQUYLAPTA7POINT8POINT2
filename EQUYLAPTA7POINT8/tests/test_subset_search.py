"""tests/test_subset_search.py

Unit test for exhaustive 16-subset search and global minimality.
"""
from __future__ import annotations

import unittest
from discovery.subset_search import run_exhaustive_subset_search
from circuits.circuit_minimizer import get_standard_circuits
from circuits.circuit_validation import validate_circuit_invariants


class TestSubsetSearch(unittest.TestCase):
    def test_exhaustive_subsets(self):
        res = run_exhaustive_subset_search()
        self.assertEqual(res["total_subsets_evaluated"], 16)
        self.assertEqual(len(res["subsets"]), 16)

        # Full circuit should recover 100% of capability
        full = res["full_circuit_100pct"]
        self.assertEqual(full["size"], 4)
        self.assertEqual(full["percent_capability_recovered"], 100.0)

        # Minimal nucleus should be size 2
        min_nuc = res["minimal_nucleus_50pct"]
        self.assertEqual(min_nuc["size"], 2)
        self.assertGreaterEqual(min_nuc["percent_capability_recovered"], 50.0)

    def test_circuit_invariants(self):
        circuits = get_standard_circuits()
        val = validate_circuit_invariants(circuits)
        self.assertTrue(val["all_invariants_satisfied"])
        self.assertTrue(val["details"]["oversized_distinct_from_fdc"])


if __name__ == "__main__":
    unittest.main()
