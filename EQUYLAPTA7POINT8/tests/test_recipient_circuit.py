"""tests/test_recipient_circuit.py

Unit test for recipient circuit discovery, reconstruction, and ablation.
"""
from __future__ import annotations

import unittest
from models.synthetic import load_model
from benchmarking.suites import micro_suite
from transfer.architecture_translator import ArchitectureTranslator
from causal.ablation import evaluate_recipient_ablations
from causal.path_patch import discover_recipient_causal_pathway


class TestRecipientCircuit(unittest.TestCase):
    def test_recipient_circuit_transfer(self):
        m_src = load_model("e4-math-4L")
        m_tgt = load_model("e6-base-4L")
        suite = micro_suite("math", n=20, seed=888)

        trans = ArchitectureTranslator(
            donor_model=m_src,
            recipient_model=m_tgt,
            active_elements=["L0_head_2", "L0_mlp"],
            condition_name="Condition_C_Recipient_Reconstructed_Circuit",
            seed=42
        )

        # Invariant: donor components must be strictly frozen
        self.assertTrue(trans.verify_all_frozen())

        # Test ablations
        abl_res = evaluate_recipient_ablations(trans, suite)
        self.assertIn("active_accuracy", abl_res)
        self.assertIn("ablated_accuracy", abl_res)
        self.assertIn("circuit_causal_drop_pp", abl_res)

        # Test pathway discovery
        path_res = discover_recipient_causal_pathway(trans, suite)
        self.assertIn("primary_recipient_mediator", path_res)
        self.assertEqual(path_res["primary_recipient_mediator"], "Recipient_L0_mlp")


if __name__ == "__main__":
    unittest.main()
