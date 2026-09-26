"""EQUYLAPTA6 — Cross-Architecture Functional Translation Unit Tests."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.synthetic import make_micro
from demo.run_equylapta6 import (
    get_arch_report, compare_architectures,
    destroy_target_payload, extract_source_functional_signature
)


class TestE6FunctionalTranslation(unittest.TestCase):
    def setUp(self):
        self.src = make_micro("test-src", d_model=64, n_layers=2, n_heads=4)
        self.tgt = make_micro("test-tgt", d_model=96, n_layers=2, n_heads=6)

    def test_architecture_difference_detected(self):
        arch_a = get_arch_report(self.src)
        arch_b = get_arch_report(self.tgt)
        diff = compare_architectures(arch_a, arch_b)
        
        self.assertEqual(arch_a["hidden"], 64)
        self.assertEqual(arch_b["hidden"], 96)
        self.assertEqual(arch_a["heads"], 4)
        self.assertEqual(arch_b["heads"], 6)
        self.assertEqual(arch_a["mlp_hidden"], 256)
        self.assertEqual(arch_b["mlp_hidden"], 384)
        self.assertFalse(diff["naive_tensor_compatibility"])
        self.assertTrue(len(diff["different"]) >= 4)

    def test_naive_weight_transfer_incompatible_shapes(self):
        src_qkv = self.src.params["L0.attn.qkv.W"]
        tgt_qkv = self.tgt.params["L0.attn.qkv.W"]
        
        self.assertEqual(src_qkv.shape, (64, 192))
        self.assertEqual(tgt_qkv.shape, (96, 288))
        self.assertNotEqual(src_qkv.shape, tgt_qkv.shape)
        
        # Incompatible shapes break model forward if assigned
        self.tgt.params["L0.attn.qkv.W"] = src_qkv
        with self.assertRaises(ValueError):
            self.tgt.forward(np.array([[1, 2, 3]], dtype=np.int64))

    def test_functional_destruction_preserves_norms_and_shapes(self):
        payload = {
            "L0.attn.qkv.W": np.random.normal(0, 1, (96, 288)).astype(np.float32),
            "L0.attn.o.W": np.random.normal(0, 1, (96, 96)).astype(np.float32),
            "L0.attn.qkv.b": np.zeros(288, dtype=np.float32),
        }
        orig_norm_qkv = float(np.linalg.norm(payload["L0.attn.qkv.W"]))
        orig_norm_o = float(np.linalg.norm(payload["L0.attn.o.W"]))

        destroyed = destroy_target_payload(payload, seed=42)

        # Shapes strictly preserved
        self.assertEqual(destroyed["L0.attn.qkv.W"].shape, (96, 288))
        self.assertEqual(destroyed["L0.attn.o.W"].shape, (96, 96))
        self.assertEqual(destroyed["L0.attn.qkv.b"].shape, (288,))

        # Norms strictly preserved
        self.assertAlmostEqual(float(np.linalg.norm(destroyed["L0.attn.qkv.W"])), orig_norm_qkv, places=4)
        self.assertAlmostEqual(float(np.linalg.norm(destroyed["L0.attn.o.W"])), orig_norm_o, places=4)

        # Organization scrambled (not identical)
        self.assertFalse(np.allclose(destroyed["L0.attn.qkv.W"], payload["L0.attn.qkv.W"]))

    def test_functional_signature_structure(self):
        calib = ["1 + 1 = 2", "2 + 2 = 4"]
        probe = ["hello world"]
        sig = extract_source_functional_signature(self.src, calib, probe)

        self.assertIn("circuit_id", sig)
        self.assertIn("records", sig)
        self.assertTrue(len(sig["records"]) >= 2)
        rec0 = sig["records"][0]
        self.assertIn("top1_token", rec0)
        self.assertIn("probs", rec0)
        self.assertIn("entropy", rec0)
        self.assertAlmostEqual(sum(rec0["probs"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
