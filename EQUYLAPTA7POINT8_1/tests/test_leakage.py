"""tests/test_leakage.py

Automated leakage verification test (Section 27).
Ensures zero token/sequence overlap between characterization,
validation, and held-out test splits.
"""
from __future__ import annotations

import unittest
from src.data_splits import get_disjoint_splits


class TestDataLeakage(unittest.TestCase):
    def test_split_disjointness(self):
        splits = get_disjoint_splits(domain="math", n=20)
        s_char = splits["characterization"]
        s_val = splits["validation"]
        s_held = splits["heldout"]

        prompts_char = set(it.prompt for it in s_char.items)
        prompts_val = set(it.prompt for it in s_val.items)
        prompts_held = set(it.prompt for it in s_held.items)

        # 1. No overlap between characterization and held-out
        char_held_overlap = prompts_char.intersection(prompts_held)
        self.assertEqual(len(char_held_overlap), 0, f"Leakage detected between characterization and held-out: {char_held_overlap}")

        # 2. No overlap between validation and held-out
        val_held_overlap = prompts_val.intersection(prompts_held)
        self.assertEqual(len(val_held_overlap), 0, f"Leakage detected between validation and held-out: {val_held_overlap}")

        # 3. No overlap between characterization and validation
        char_val_overlap = prompts_char.intersection(prompts_val)
        self.assertEqual(len(char_val_overlap), 0, f"Leakage detected between characterization and validation: {char_val_overlap}")


if __name__ == "__main__":
    unittest.main()
