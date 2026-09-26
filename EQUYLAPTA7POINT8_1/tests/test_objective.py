"""tests/test_objective.py

Unit test for two-branch functional causal loss computation.
"""
from __future__ import annotations

import unittest
import numpy as np
from models.synthetic import load_model
from src.corrected_functional_objective import compute_donor_causal_signature, compute_functional_loss


class TestFunctionalObjective(unittest.TestCase):
    def test_loss_computation(self):
        m_src = load_model("e4-math-4L")
        m_tgt = load_model("e6-base-4L")
        ids = np.array([[5, 12, 19, 33]], dtype=np.int64)

        delta_target = compute_donor_causal_signature(m_src, ids, (0, 2))
        self.assertEqual(delta_target.shape[0], m_src.spec.vocab)

        def ablate():
            m_tgt.params["L0.attn.o.W"][:16, :] = 0.0

        def restore():
            m_tgt.params["L0.attn.o.W"][:16, :] = 0.01

        loss, diff, l_int, l_abl = compute_functional_loss(m_tgt, ids, delta_target, ablate, restore)
        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(diff.shape[0], m_tgt.spec.vocab)


if __name__ == "__main__":
    unittest.main()
