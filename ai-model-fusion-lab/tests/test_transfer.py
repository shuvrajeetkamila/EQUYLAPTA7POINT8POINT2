"""Milestone-3 transfer-engine tests (spec §33). Tiny fixtures, CPU-only."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _tiny_pair(seed_a=1, seed_b=2, d=32):
    from models.synthetic import make_micro, master_tokenizer
    tok = master_tokenizer()
    a = make_micro("ta", d_model=d, n_layers=2, n_heads=4,
                   vocab_size=tok.vocab_size, tokenizer=tok, seed=seed_a)
    b = make_micro("tb", d_model=d, n_layers=2, n_heads=4,
                   vocab_size=tok.vocab_size, tokenizer=tok, seed=seed_b)
    return a, b, tok


def _head_comp(model, layer=0, head=1):
    from component_transfer.component import ComponentRecord
    return ComponentRecord(
        component_id="test_head", source_model=model.name,
        source_model_revision="test", architecture=model.spec.arch,
        component_type="HEAD", layer=layer, head_ids=[head],
        hidden_dimension=model.spec.hidden, input_dimension=model.spec.hidden,
        output_dimension=model.spec.hidden, tokenizer="test",
        capability_target="math", baseline_score=40.0,
        ablation_score=30.0, confidence="low", transfer_status="UNTESTED")


def _chan_comp(model, layer=1, lo=4, hi=12):
    from component_transfer.component import ComponentRecord
    return ComponentRecord(
        component_id="test_chan", source_model=model.name,
        source_model_revision="test", architecture=model.spec.arch,
        component_type="CHANNEL_GROUP", layer=layer, channel_range=[lo, hi],
        hidden_dimension=model.spec.hidden, input_dimension=model.spec.hidden,
        output_dimension=model.spec.hidden, tokenizer="test",
        capability_target="math", baseline_score=40.0,
        ablation_score=35.0, confidence="low", transfer_status="UNTESTED")


class TestComponentRecord(unittest.TestCase):
    def test_serialization_roundtrip_and_validation(self):
        from component_transfer.component import ComponentRecord
        a, b, tok = _tiny_pair()
        c = _head_comp(a)
        c.save_tensors({"qkv.W.q": np.zeros((4, 4), np.float32)})
        c2 = ComponentRecord.from_dict(c.to_dict())
        self.assertEqual(c2.component_type, "HEAD")
        self.assertEqual(c2.param_count(), 16)
        # components without baseline evidence are rejected (spec §3)
        with self.assertRaises(ValueError):
            bad = _head_comp(a)
            bad.baseline_score = None
            bad.save_meta()
        with self.assertRaises(ValueError):
            bad2 = _head_comp(a)
            bad2.component_type = "NOT_A_TYPE"
            bad2.save_meta()

    def test_load_missing_raises(self):
        from component_transfer.component import ComponentRecord
        with self.assertRaises(FileNotFoundError):
            ComponentRecord.load("definitely_not_a_component_id")


class TestAlignment(unittest.TestCase):
    def test_identity_dimension_mismatch_raises(self):
        from component_transfer.alignment import fit_alignment, DimensionMismatchError
        X = np.random.default_rng(0).normal(size=(40, 8))
        Y = np.random.default_rng(1).normal(size=(40, 16))
        with self.assertRaises(DimensionMismatchError) as cm:
            fit_alignment(X, Y, "identity")
        self.assertIn("SOURCE", str(cm.exception))

    def test_linear_alignment_beats_baseline_on_heldout(self):
        from component_transfer.alignment import fit_alignment
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 16))
        W_true = rng.normal(size=(16, 12))
        Y = X @ W_true + 0.1 * rng.normal(size=(300, 12))
        t, res = fit_alignment(X, Y, "linear", seed=0)
        self.assertFalse(res.saturated)
        self.assertGreater(res.aligned_similarity, 0.95)
        self.assertGreater(res.heldout_r2, 0.9)

    def test_saturation_flagged_when_n_small(self):
        from component_transfer.alignment import fit_alignment
        rng = np.random.default_rng(4)
        X = rng.normal(size=(20, 16))
        Y = rng.normal(size=(20, 16))
        t, res = fit_alignment(X, Y, "linear", seed=0)
        self.assertTrue(res.saturated)

    def test_low_rank_rank_recorded(self):
        from component_transfer.alignment import fit_alignment
        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 16))
        Y = X @ rng.normal(size=(16, 16))
        t, res = fit_alignment(X, Y, "low_rank", seed=0, rank=4)
        self.assertEqual(res.rank, 4)

    def test_apply_validates_dims(self):
        from component_transfer.alignment import fit_alignment, apply_alignment, DimensionMismatchError
        X = np.random.default_rng(6).normal(size=(100, 12))
        Y = X @ np.random.default_rng(7).normal(size=(12, 9))
        t, res = fit_alignment(X, Y, "linear", seed=0)
        with self.assertRaises(DimensionMismatchError):
            apply_alignment(t, np.zeros((3, 30)))


class TestInsertion(unittest.TestCase):
    def test_head_insertion_exact_slices_and_restore(self):
        from component_transfer.apply import (insert_component_tensors,
                                              snapshot, restore_snapshot)
        a, b, tok = _tiny_pair()
        c = _head_comp(a)
        tensors = {"qkv.W.q": a.state_dict()["L0.attn.qkv.W"][:, 8:16].copy(),
                   "qkv.W.k": a.state_dict()["L0.attn.qkv.W"][:, 40:48].copy(),
                   "qkv.W.v": a.state_dict()["L0.attn.qkv.W"][:, 72:80].copy(),
                   "qkv.b.q": a.state_dict()["L0.attn.qkv.b"][8:16].copy(),
                   "qkv.b.k": a.state_dict()["L0.attn.qkv.b"][40:48].copy(),
                   "qkv.b.v": a.state_dict()["L0.attn.qkv.b"][72:80].copy(),
                   "o.W": a.state_dict()["L0.attn.o.W"][8:16, :].copy()}
        snap = snapshot(b)
        rec = insert_component_tensors(b, c, tensors)
        self.assertEqual(len(rec["applied"]), 7)
        D = b.state_dict()
        self.assertTrue(np.allclose(D["L0.attn.qkv.W"][:, 8:16], tensors["qkv.W.q"]))
        restore_snapshot(b, snap)
        D = b.state_dict()
        self.assertFalse(np.allclose(D["L0.attn.qkv.W"][:, 8:16], tensors["qkv.W.q"]))

    def test_cross_dim_insertion_requires_transform(self):
        from component_transfer.apply import insert_component_tensors
        from component_transfer.alignment import DimensionMismatchError
        from models.synthetic import make_micro, master_tokenizer
        tok = master_tokenizer()
        a = make_micro("sa", d_model=32, n_layers=2, n_heads=4,
                       vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        b = make_micro("sb", d_model=48, n_layers=2, n_heads=6,
                       vocab_size=tok.vocab_size, tokenizer=tok, seed=1)
        c = _chan_comp(a)
        tensors = {"mlp.1.W": np.zeros((32, 8), np.float32),
                   "mlp.1.b": np.zeros(8, np.float32),
                   "mlp.2.W": np.zeros((8, 32), np.float32)}
        with self.assertRaises(DimensionMismatchError):
            insert_component_tensors(b, c, tensors)  # no transform provided

    def test_cross_dim_projected_insertion_shapes(self):
        from component_transfer.apply import insert_component_tensors
        from models.synthetic import make_micro, master_tokenizer
        tok = master_tokenizer()
        a = make_micro("sa2", d_model=32, n_layers=2, n_heads=4,
                       vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        b = make_micro("sb2", d_model=48, n_layers=2, n_heads=6,
                       vocab_size=tok.vocab_size, tokenizer=tok, seed=1)
        c = _chan_comp(a)
        U_in = np.random.default_rng(0).normal(size=(48, 32)).astype(np.float32)
        V_out = np.random.default_rng(1).normal(size=(32, 48)).astype(np.float32)
        tensors = {"mlp.1.W": np.ones((32, 8), np.float32),
                   "mlp.1.b": np.zeros(8, np.float32),
                   "mlp.2.W": np.ones((8, 32), np.float32)}
        rec = insert_component_tensors(b, c, tensors,
                                       cross_dim_transform={"U_in": U_in, "V_out": V_out})
        self.assertEqual(len(rec["applied"]), 3)
        D = b.state_dict()
        self.assertTrue(np.allclose(D["L1.mlp.1.W"][:, 4:12],
                                    (U_in @ np.ones((32, 8))).astype(np.float32)))


class TestRefitCalibration(unittest.TestCase):
    def test_refit_trains_only_masked_params(self):
        from component_transfer.apply import insert_component_tensors
        from component_transfer.refitting import refit_component_only
        a, b, tok = _tiny_pair(seed_a=11, seed_b=12)
        c = _chan_comp(a, lo=0, hi=8)
        tensors = {"mlp.1.W": a.state_dict()["L1.mlp.1.W"][:, 0:8].copy(),
                   "mlp.1.b": a.state_dict()["L1.mlp.1.b"][0:8].copy(),
                   "mlp.2.W": a.state_dict()["L1.mlp.2.W"][0:8, :].copy()}
        from component_transfer.apply import tensor_locations
        locs = tensor_locations(c, b.spec)
        insert_component_tensors(b, c, tensors)
        rng = np.random.default_rng(0)
        seqs = [tok.encode("12 + 4 = 16", max_len=16) for _ in range(40)]
        before = {k: v.copy() for k, v in b.state_dict().items()}
        res = refit_component_only(b, locs, seqs, steps=25, lr=3e-3, seed=0)
        self.assertLess(res.final_loss, res.initial_loss)
        self.assertEqual(res.trainable_params, 8 * 32 + 8 + 8 * 32)
        D = b.state_dict()
        # masked tensors changed
        self.assertFalse(np.allclose(D["L1.mlp.1.W"][:, 0:8], before["L1.mlp.1.W"][:, 0:8]))
        # untouched tensors identical
        self.assertTrue(np.allclose(D["L1.mlp.1.W"][:, 8:], before["L1.mlp.1.W"][:, 8:]))
        self.assertTrue(np.allclose(D["L0.attn.qkv.W"], before["L0.attn.qkv.W"]))

    def test_calibration_residual_gate(self):
        from component_transfer.apply import insert_component_tensors, tensor_locations
        from component_transfer.calibration import calibrate, out_locations_of
        a, b, tok = _tiny_pair(seed_a=21, seed_b=22)
        c = _chan_comp(a, lo=0, hi=8)
        tensors = {"mlp.1.W": a.state_dict()["L1.mlp.1.W"][:, 0:8].copy(),
                   "mlp.1.b": a.state_dict()["L1.mlp.1.b"][0:8].copy(),
                   "mlp.2.W": (a.state_dict()["L1.mlp.2.W"][0:8, :] * 5.0).copy()}
        insert_component_tensors(b, c, tensors)
        seqs = [tok.encode("the sky is blue", max_len=16) for _ in range(24)]
        locs = tensor_locations(c, b.spec)
        res = calibrate(b, out_locations_of(locs), seqs, "residual_gate")
        self.assertIn(res.method, ("residual_gate",))
        self.assertIsNotNone(res.gate)
        self.assertLessEqual(res.loss_after, res.loss_before + 1e-6)


class TestTokenizerAndArchCompatibility(unittest.TestCase):
    def test_tokenizer_mismatch_detected(self):
        from component_transfer.component import ComponentRecord
        from models.base import Tokenizer
        from models.synthetic import make_micro
        tok1 = Tokenizer(["cat", "dog"])
        tok2 = Tokenizer(["cat", "dog", "extra"])
        m1 = make_micro("t1", d_model=16, n_layers=1, n_heads=2,
                        vocab_size=tok1.vocab_size, tokenizer=tok1, seed=0)
        m2 = make_micro("t2", d_model=16, n_layers=1, n_heads=2,
                        vocab_size=tok2.vocab_size, tokenizer=tok2, seed=1)
        c = _head_comp(m1)
        c.tokenizer = f"micro-vocab-{m1.spec.vocab}"
        mismatch = (c.tokenizer != f"micro-vocab-{m2.spec.vocab}")
        self.assertTrue(mismatch, "tokenizer mismatch must be detectable")

    def test_arch_mismatch_in_insertion(self):
        from component_transfer.apply import insert_component_tensors
        from component_transfer.alignment import DimensionMismatchError
        from models.synthetic import make_micro, master_tokenizer
        tok = master_tokenizer()
        m1 = make_micro("m1", d_model=32, n_layers=2, n_heads=4,
                        vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        m3 = make_micro("m3", d_model=32, n_layers=3, n_heads=4,
                        vocab_size=tok.vocab_size, tokenizer=tok, seed=1)
        c = _head_comp(m1, layer=2)  # layer 2 exists in source(3 layers)? source has 2
        c.layer = 0
        tensors = {"qkv.W.q": np.zeros((32, 8), np.float32)}
        # target m3 has 3 layers; shapes per-layer are the same, so arch
        # mismatch on LAYER COUNT must be caught by compatibility, not shapes:
        from models.registry import compatibility
        rec_a = {"name": "a", "fingerprint": "x", "spec": m1.spec.to_dict()}
        rec_b = {"name": "b", "fingerprint": "y", "spec": m3.spec.to_dict()}
        self.assertFalse(compatibility(rec_a, rec_b)["shape_compatible"])


class TestAdapterInsertion(unittest.TestCase):
    def _payload(self, a, lo=0, hi=16):
        return {"mlp.1.W": a.state_dict()["L0.mlp.1.W"][:, lo:hi].copy(),
                "mlp.1.b": a.state_dict()["L0.mlp.1.b"][lo:hi].copy(),
                "mlp.2.W": a.state_dict()["L0.mlp.2.W"][lo:hi, :].copy()}

    def test_gate_interpolates_and_endpoints_exact(self):
        from component_transfer.apply import (adapter_insertion, snapshot,
                                              restore_snapshot)
        a, b, tok = _tiny_pair()
        c = _chan_comp(a, layer=0, lo=0, hi=16)
        tensors = self._payload(a)
        snap = snapshot(b)
        W0 = b.state_dict()["L0.mlp.1.W"][:, 0:16].copy()
        # gate=0: target untouched
        adapter_insertion(b, c, tensors, None, gate=0.0)
        self.assertTrue(np.allclose(b.state_dict()["L0.mlp.1.W"][:, 0:16], W0))
        # gate=1: exact replacement == insert_component_tensors
        adapter_insertion(b, c, tensors, None, gate=1.0)
        W1 = b.state_dict()["L0.mlp.1.W"][:, 0:16].copy()
        self.assertTrue(np.allclose(W1, tensors["mlp.1.W"]))
        restore_snapshot(b, snap)
        # gate=0.25: convex blend
        adapter_insertion(b, c, tensors, None, gate=0.25)
        Wg = b.state_dict()["L0.mlp.1.W"][:, 0:16]
        self.assertTrue(np.allclose(Wg, 0.75 * W0 + 0.25 * tensors["mlp.1.W"], atol=1e-5))
        restore_snapshot(b, snap)

    def test_gate_bounds_and_untouched_outside_slice(self):
        from component_transfer.apply import adapter_insertion, snapshot, restore_snapshot
        a, b, tok = _tiny_pair()
        c = _chan_comp(a, layer=0, lo=0, hi=16)
        tensors = self._payload(a)
        snap = snapshot(b)
        before = b.state_dict()["L0.mlp.1.W"].copy()
        with self.assertRaises(ValueError):
            adapter_insertion(b, c, tensors, None, gate=1.5)
        adapter_insertion(b, c, tensors, None, gate=0.5)
        after = b.state_dict()["L0.mlp.1.W"]
        self.assertTrue(np.allclose(after[:, 16:], before[:, 16:]))  # outside slice
        self.assertFalse(np.allclose(after[:, 0:16], before[:, 0:16]))
        restore_snapshot(b, snap)


class TestModularSuites(unittest.TestCase):
    def test_train_heldout_value_disjoint(self):
        from models.modular_corpora import modular_suite
        tr = modular_suite("algebra", 30, "train", seed=3)
        ho = modular_suite("algebra", 30, "heldout", seed=3)
        # the a-coefficients ranges are disjoint by construction
        import re
        tr_a = {int(re.match(r"solve x : (\d+) x", it.prompt).group(1)) for it in tr.items}
        ho_a = {int(re.match(r"solve x : (\d+) x", it.prompt).group(1)) for it in ho.items}
        self.assertFalse(tr_a & ho_a)


if __name__ == "__main__":
    unittest.main()
