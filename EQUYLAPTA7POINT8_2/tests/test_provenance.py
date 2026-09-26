"""tests/test_provenance.py

Unit test for parameter provenance, frozen invariants, and zero unauthorized weight modification.
"""
from __future__ import annotations

import os
import sys
import unittest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.synthetic import load_model
from src.transfer import ArchitectureTranslator


class TestParameterProvenance(unittest.TestCase):
    def test_frozen_invariants(self):
        m_src = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e4-math-4L"))
        m_tgt = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e6-base-4L"))

        trans = ArchitectureTranslator(
            donor_model=m_src,
            recipient_model=m_tgt,
            active_elements=["L0_head_2", "L0_mlp"],
            condition_name="Condition_E_Recipient_Reconstructed_FDC",
            seed=42
        )

        # Invariant check
        self.assertTrue(trans.verify_all_frozen())

        fp = trans.get_parameter_footprint()
        self.assertGreater(fp["donor_frozen_params"], 0)
        self.assertGreater(fp["adapter_trainable_params"], 0)


if __name__ == "__main__":
    unittest.main()
