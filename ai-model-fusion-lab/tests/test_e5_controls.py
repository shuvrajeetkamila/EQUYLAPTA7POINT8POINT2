"""EQUYLAPTA5 — location/content control helper tests."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MEMBERS = [{"level": "HEAD", "layer": 0, "module": "attn", "index": i}
           for i in range(4)] + [{"level": "MODULE", "layer": 0,
                                  "module": "attn"}]


class TestE5Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "demo"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_equylapta5", os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "demo", "run_equylapta5.py"))
        cls.e5 = importlib.util.module_from_spec(spec)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            spec.loader.exec_module(cls.e5)

    def test_relocate_moves_every_member(self):
        rm = self.e5.relocate(MEMBERS, 2)
        self.assertTrue(all(m["layer"] == 2 for m in rm))
        self.assertEqual([m["index"] for m in rm if m["level"] == "HEAD"],
                         [0, 1, 2, 3])

    def test_random_relocate_never_all_correct(self):
        for s in (21, 22, 23, 24, 25):
            rm = self.e5.random_relocate(MEMBERS, s)
            self.assertTrue(any(m["layer"] != 0 for m in rm),
                            f"seed {s} produced all-correct locations")

    def test_repack_preserves_values(self):
        payload = {"L0.h0": {"o.W": np.arange(48, dtype=np.float32)
                             .reshape(16, 3)}}
        rm = self.e5.relocate([MEMBERS[0]], 3)
        rp = self.e5.repack(payload, [MEMBERS[0]], rm)
        self.assertIn("L3.h0", rp)
        np.testing.assert_array_equal(rp["L3.h0"]["o.W"],
                                      payload["L0.h0"]["o.W"])

    def test_stats_matched_random_preserves_mean_std(self):
        rng = np.random.default_rng(0)
        payload = {"L0.h0": {"o.W": rng.normal(0.5, 0.2, (32, 16))
                             .astype(np.float32)}}
        out = self.e5.stats_matched_random(payload, 66)["L0.h0"]["o.W"]
        self.assertAlmostEqual(float(np.mean(out)), 0.5, delta=0.02)
        self.assertAlmostEqual(float(np.std(out)), 0.2, delta=0.02)

    def test_basis_mismatch_preserves_norms_not_alignment(self):
        rng = np.random.default_rng(1)
        w = rng.normal(0, 1, (16, 64)).astype(np.float32)
        payload = {"L0.h0": {"o.W": w, "o.b": np.zeros(16, np.float32)}}
        out = self.e5.basis_mismatch(payload, 77)["L0.h0"]
        # column norms preserved (permutation), Frobenius norm identical
        np.testing.assert_allclose(
            np.sort(np.linalg.norm(out["o.W"], axis=0)),
            np.sort(np.linalg.norm(w, axis=0)), rtol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(out["o.W"])),
                               float(np.linalg.norm(w)), places=3)
        # biases untouched (per-output-channel constants)
        np.testing.assert_array_equal(out["o.b"], payload["L0.h0"]["o.b"])
        # alignment destroyed (columns reordered)
        self.assertFalse(np.allclose(out["o.W"], w))

    def test_shuffled_preserves_magnitude_multiset(self):
        rng = np.random.default_rng(2)
        w = rng.normal(0, 1, (8, 8)).astype(np.float32)
        out = self.e5._shuffled({"L0.h0": {"o.W": w}}, 43)["L0.h0"]["o.W"]
        np.testing.assert_allclose(np.sort(out.ravel()),
                                   np.sort(w.ravel()), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
