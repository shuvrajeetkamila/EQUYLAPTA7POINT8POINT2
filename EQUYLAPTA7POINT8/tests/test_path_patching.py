"""tests/test_path_patching.py

Unit test for donor path patching and mediation estimation.
"""
from __future__ import annotations

import unittest
from discovery.path_patching import run_path_patching_battery
from discovery.mediation import compute_conditional_mediation


class TestPathPatching(unittest.TestCase):
    def test_path_patching_conditions(self):
        res = run_path_patching_battery()
        conds = res["conditions"]
        self.assertIn("condition_1_intact", conds)
        self.assertIn("condition_2_source_intervention", conds)
        self.assertIn("condition_4_mediator_restored", conds)

        # Intact should be higher than source intervention
        self.assertGreater(conds["condition_1_intact"]["accuracy"],
                           conds["condition_2_source_intervention"]["accuracy"])

        # Mediator restored should recover capability over source intervention
        self.assertGreaterEqual(conds["condition_4_mediator_restored"]["accuracy"],
                                conds["condition_2_source_intervention"]["accuracy"])

        # Mediation metrics
        med = compute_conditional_mediation(res)
        self.assertGreaterEqual(med["total_effect_pp"], 0.0)
        self.assertGreaterEqual(med["proportion_mediated_pct"], 0.0)
        self.assertLessEqual(med["proportion_mediated_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
