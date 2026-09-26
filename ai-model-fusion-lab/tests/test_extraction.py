"""EQUYLAPTA4 — CircuitPack extraction tests (compact transfer object)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.synthetic import load_model  # noqa: E402
from pattern_genome.extraction import CircuitPack  # noqa: E402


def _fake_circuit(head_idx=2):
    return {"circuit_id": "test-circuit", "kind": "convergence",
            "pattern_ids": ["P024"],
            "members": [
                {"level": "HEAD", "layer": 0, "module": "attn",
                 "index": head_idx},
                {"level": "MODULE", "layer": 0, "module": "mlp"}]}


class TestCircuitPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = load_model("deep-math")
        except Exception:
            cls.model = None

    def setUp(self):
        if self.model is None:
            self.skipTest("deep-math model not trained yet")

    def test_roundtrip_values_within_fp16(self):
        pack = CircuitPack.from_model(self.model, _fake_circuit(), "math")
        err = pack.verify_against_model(self.model)
        self.assertLess(err, 0.05, f"fp16 roundtrip err {err}")

    def test_manifest_is_complete(self):
        pack = CircuitPack.from_model(self.model, _fake_circuit(), "math")
        m = pack.manifest
        for field in ("circuit_id", "source_model", "source_spec", "members",
                      "tensor_shapes", "n_params_fp16", "fp16_max_rel_error"):
            self.assertIn(field, m)
        self.assertEqual(m["n_params_fp16"],
                         sum(int(np.asarray(v).size)
                             for p in pack.tensors.values()
                             for v in p.values()))
        self.assertLess(m["n_params_fp16"], 100_000, "pack must be a circuit,"
                        " not the model")

    def test_pack_is_compact(self):
        import io
        pack = CircuitPack.from_model(self.model, _fake_circuit(), "math")
        buf = io.BytesIO()
        np.savez_compressed(buf, **{f"{k}||{n}": v
                                    for k, p in pack.tensors.items()
                                    for n, v in p.items()})
        self.assertLess(buf.tell(), 200_000, "pack should be tens of KB")

    def test_save_load_roundtrip(self):
        import tempfile
        pack = CircuitPack.from_model(self.model, _fake_circuit(), "math")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "pack")
            pack.save(path)
            pack2 = CircuitPack.load(path + ".npz")
        self.assertEqual(pack2.manifest["circuit_id"], "test-circuit")
        for k in pack.tensors:
            for n in pack.tensors[k]:
                np.testing.assert_array_equal(pack.tensors[k][n],
                                              pack2.tensors[k][n])

    def test_shuffled_preserves_shapes_and_norms_not_values(self):
        pack = CircuitPack.from_model(self.model, _fake_circuit(), "math")
        sh = pack.shuffled_payload(seed=3)
        changed = 0
        for k, p in pack.tensors.items():
            for n, v in p.items():
                s = sh[k][n]
                self.assertEqual(s.shape, np.asarray(v).shape)
                self.assertAlmostEqual(float(np.sqrt(np.mean(
                    np.asarray(v, dtype=np.float64) ** 2))),
                    float(np.sqrt(np.mean(s.astype(np.float64) ** 2))),
                    places=4)
                if not np.allclose(np.asarray(v, dtype=np.float32),
                                   s.astype(np.float32)):
                    changed += 1
        self.assertGreater(changed, 0, "shuffle must alter the payload")


if __name__ == "__main__":
    unittest.main()
