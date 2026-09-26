"""Core correctness tests. Run:  python -m unittest tests.test_core -v"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class TestBackendEquivalence(unittest.TestCase):
    """torch backend must match the numpy reference bit-near-exactly."""

    def test_forward_match(self):
        from models.backend_torch import TORCH_OK
        if not TORCH_OK:
            self.skipTest("torch unavailable")
        from models.backend_numpy import MicroTransformer
        from models.backend_torch import TorchMicroTransformer
        from models.base import Tokenizer
        tok = Tokenizer(["the", "cat", "sat", "+", "1", "2"])
        tn = MicroTransformer("t", d_model=32, n_layers=2, n_heads=4, ctx_len=10,
                              vocab_size=tok.vocab_size, tokenizer=tok, seed=3)
        tt = TorchMicroTransformer("t", d_model=32, n_layers=2, n_heads=4, ctx_len=10,
                                   vocab_size=tok.vocab_size, tokenizer=tok, seed=3)
        tt.load_state_dict(tn.state_dict())
        ids = np.array([[2, 4, 6, 3, 5, 2]])
        ln, _ = tn.forward(ids)
        lt, _ = tt.forward(ids)
        self.assertLess(float(np.abs(ln - lt).max()), 1e-4)


class TestGradientCheck(unittest.TestCase):
    def test_numpy_backward_matches_numerical(self):
        from models.backend_numpy import MicroTransformer
        from models.base import Tokenizer
        tok = Tokenizer(["a", "b", "c"])
        m = MicroTransformer("g", d_model=24, n_layers=2, n_heads=4, ctx_len=8,
                             vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        rng = np.random.default_rng(2)
        ids = rng.integers(2, tok.vocab_size, (2, 6))

        def loss():
            logits, _ = m.forward(ids)
            lg = logits[:, :-1, :]
            P = m._softmax(lg.reshape(-1, lg.shape[-1]))
            tgt = np.zeros_like(P)
            tgt[np.arange(len(P)), ids[:, 1:].reshape(-1)] = 1
            return -float((tgt * np.log(P + 1e-9)).sum(-1).mean())

        _, G = m.loss_and_grads(ids)
        worst = 0.0
        for k in ["We", "L0.attn.qkv.W", "L1.mlp.2.W", "lnf.g"]:
            P = m.params[k]
            for fi in rng.choice(P.size, min(4, P.size), replace=False):
                idx = np.unravel_index(fi, P.shape)
                old = P[idx]; eps = 2e-3
                P[idx] = old + eps; lp = loss()
                P[idx] = old - eps; lm = loss()
                P[idx] = old
                num = (lp - lm) / (2 * eps)
                ana = G[k][idx]
                worst = max(worst, abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
        self.assertLess(worst, 2e-2, f"gradient check too imprecise: {worst}")


class TestAblation(unittest.TestCase):
    def test_layer_ablation_changes_output(self):
        from models.backend_numpy import MicroTransformer
        from models.base import Tokenizer
        tok = Tokenizer(["a", "b", "c"])
        m = MicroTransformer("a", d_model=24, n_layers=2, n_heads=4, ctx_len=8,
                             vocab_size=tok.vocab_size, tokenizer=tok, seed=1)
        ids = np.array([[2, 3, 4, 2]])
        l1, _ = m.forward(ids)
        m.skip_layers = [0]
        l2, _ = m.forward(ids)
        m.skip_layers = []
        self.assertGreater(float(np.abs(l1 - l2).max()), 1e-6)
        # head mask zeroing must also change output
        m.head_mask = np.ones((2, 4), np.float32)
        m.head_mask[1, 2] = 0
        l3, _ = m.forward(ids)
        m.head_mask = None
        self.assertGreater(float(np.abs(l1 - l3).max()), 1e-6)


class TestMerging(unittest.TestCase):
    def setUp(self):
        from models.synthetic import make_micro, master_tokenizer
        tok = master_tokenizer()
        self.models = [make_micro(f"m{i}", d_model=32, n_layers=2, n_heads=4,
                                  vocab_size=tok.vocab_size, tokenizer=tok, seed=i)
                       for i in range(3)]

    def test_all_strategies_produce_valid_sd(self):
        from merging.strategies import run_strategy
        sds = [m.state_dict() for m in self.models]
        for method in ["weighted_average", "slerp", "ties", "dare"]:
            out = run_strategy(method, sds[:2], [0.5, 0.5])
            self.assertEqual(set(out), set(sds[0]))
            for k in out:
                self.assertEqual(out[k].shape, sds[0][k].shape)
                self.assertTrue(np.isfinite(out[k]).all())
        out = run_strategy("task_arithmetic", sds[1:], [0.5, 0.5], base=sds[0])
        self.assertEqual(set(out), set(sds[0]))

    def test_incompatible_merge_rejected(self):
        from merging.strategies import run_strategy, _check
        from models.base import Tokenizer
        from models.backend_numpy import MicroTransformer
        tok = Tokenizer(["a", "b"])
        small = MicroTransformer("s", d_model=24, n_layers=2, n_heads=4, ctx_len=8,
                                 vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        big = MicroTransformer("b", d_model=48, n_layers=2, n_heads=4, ctx_len=8,
                               vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        with self.assertRaises(AssertionError):
            _check([small.state_dict(), big.state_dict()])

    def test_slerp_equal_weights_returns_finite(self):
        from merging.strategies import slerp
        a = {"w": np.random.default_rng(0).normal(size=(10, 10)).astype(np.float32)}
        b = {"w": np.random.default_rng(1).normal(size=(10, 10)).astype(np.float32)}
        out = slerp([a, b], [0.5, 0.5])
        self.assertTrue(np.isfinite(out["w"]).all())


class TestRouter(unittest.TestCase):
    def test_router_routes_mixed_queries(self):
        from router.router import LearnedRouter, router_training_data
        X, Y = router_training_data()
        r = LearnedRouter()
        r.fit(X, Y, epochs=200)
        caps, _ = r.route("Write a Python program proving this mathematical theorem.")
        self.assertTrue(set(caps) & {"coding", "math", "reasoning"})
        caps, _ = r.route("what is 12 + 14 ?")
        self.assertIn("math", caps)


class TestRegistryCompat(unittest.TestCase):
    def test_compatibility_gates_on_shapes(self):
        from models.registry import compatibility
        a = {"name": "A", "fingerprint": "x", "spec": {"arch": "micro-gpt", "hidden": 64,
             "layers": 2, "heads": 4, "vocab": 141, "ctx_len": 32, "positional": "learned",
             "norm": "layernorm", "attention": "mha", "modality": "text", "moe": False}}
        b = dict(a, name="B", fingerprint="y")
        self.assertTrue(compatibility(a, b)["shape_compatible"])
        c = dict(a, name="C", fingerprint="z")
        c["spec"] = dict(a["spec"], hidden=128)
        self.assertFalse(compatibility(a, c)["shape_compatible"])
        self.assertIn("DISTILLATION", compatibility(a, c)["recommended_fusion"])


class TestLicensing(unittest.TestCase):
    def test_closed_source_blocks_distribution(self):
        from licensing.registry import check_configuration
        out = check_configuration([{"name": "open-thing", "license": "Apache-2.0"},
                                   {"name": "claude", "license": "proprietary-closed"}])
        self.assertEqual(out["license_compatibility"], "NOT DISTRIBUTABLE")
        out2 = check_configuration([{"name": "open-thing", "license": "Apache-2.0"}])
        self.assertEqual(out2["license_compatibility"], "COMPATIBLE")


class TestExperimentTracker(unittest.TestCase):
    def test_dedupe(self):
        import tempfile
        from experiments.tracker import ExperimentTracker
        with tempfile.TemporaryDirectory() as d:
            tr = ExperimentTracker(d)
            r1 = tr.record("merge", {"a": 1}, ["x"], {"micro-math": 50})
            r2 = tr.record("merge", {"a": 1}, ["x"], {"micro-math": 50})
            self.assertEqual(r1["id"], r2["id"])
            self.assertEqual(len(tr.all()), 1)


if __name__ == "__main__":
    unittest.main()


# ==================== MILESTONE 2 REGRESSION TESTS ====================

class TestM2Regressions(unittest.TestCase):
    """Regression tests for every bug found in the M2 audit."""

    def test_supermodel_stats_no_crash(self):
        # was: AttributeError: 'list' object has no attribute 'items'
        from models.synthetic import make_micro, master_tokenizer
        from experts.specialist import Specialist, SuperModel
        from router.router import LearnedRouter
        tok = master_tokenizer()
        m = make_micro("t", d_model=32, n_layers=2, n_heads=4,
                       vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        sp = Specialist("math", m)
        st = SuperModel([sp], LearnedRouter()).stats()
        self.assertIn("math", st["specialists"])
        self.assertGreater(st["total_params"], 0)

    def test_eval_suite_single_pass_matches_double(self):
        # eval_suite was refactored to score each item once; results must match
        from models.synthetic import load_model
        from benchmarking.suites import micro_suite
        from benchmarking.harness import score_item, eval_suite
        m = load_model("math-wiz")
        suite = micro_suite("lang", n=12, seed=5)
        manual = sum(score_item(m, it.prompt, it.options) == it.answer
                     for it in suite.items) / len(suite.items)
        r = eval_suite(m, suite)
        self.assertEqual(r["accuracy"], round(manual, 4))
        self.assertIn("n_correct", r)

    def test_distillation_shape_contract_errors(self):
        # cryptic tensor errors must now be ShapeContractError with
        # EXPECTED/RECEIVED/REASON
        from distillation.distill import ShapeContractError, _check_contract
        from models.synthetic import make_micro, master_tokenizer
        from models.base import Tokenizer
        tok = master_tokenizer()
        a = make_micro("a", d_model=32, n_layers=1, n_heads=4,
                       vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        b = make_micro("b", d_model=32, n_layers=1, n_heads=4,
                       vocab_size=tok.vocab_size, tokenizer=tok, seed=1)
        arr = np.zeros((2, 8), np.int64)
        with self.assertRaises(ShapeContractError) as cm:
            _check_contract(a, b, arr, np.zeros((2, 8, 5), np.float32))
        self.assertIn("EXPECTED", str(cm.exception))
        self.assertIn("RECEIVED", str(cm.exception))
        self.assertIn("REASON", str(cm.exception))
        # mismatched vocab must raise the tokenizer-compatibility contract
        tok2 = Tokenizer(["zzz"] + tok.itos[2:])
        c = make_micro("c", d_model=32, n_layers=1, n_heads=4,
                       vocab_size=tok2.vocab_size, tokenizer=tok2, seed=2)
        with self.assertRaises(ShapeContractError):
            _check_contract(a, c, arr, np.zeros((2, 7, tok2.vocab_size), np.float32))

    def test_distill_end_to_end_no_crash(self):
        # the M1 crash path: teacher -> filtered examples -> student
        from models.synthetic import make_micro, master_tokenizer
        from distillation.distill import distill
        tok = master_tokenizer()
        teacher = make_micro("tch", d_model=32, n_layers=1, n_heads=4,
                             vocab_size=tok.vocab_size, tokenizer=tok, seed=0)
        student = make_micro("std", d_model=32, n_layers=1, n_heads=4,
                             vocab_size=tok.vocab_size, tokenizer=tok, seed=1)
        texts = ["12 + 4 = 16", "max of 9 and 3 is 9", "Ann is taller than Ben ."] * 5
        out = distill(teacher, student, texts, epochs=2, soft=True)
        self.assertIn("final_loss", out)
        self.assertLess(out["final_loss"], 50)  # sane CE scale


class TestHFAblationHooks(unittest.TestCase):
    """Real-model hooks: head/layer/module ablation must change outputs.
    Skipped when transformers/model unavailable (kept honest, never faked)."""

    def _handle(self):
        try:
            from models.backend_hf import HFHandle
            return HFHandle("EleutherAI/pythia-70m")
        except Exception:
            self.skipTest("HF model unavailable")

    def test_head_ablation_changes_output(self):
        h = self._handle()
        import numpy as np
        ids = h.tokenizer.encode("Hello world")[:8]
        base, _ = h.forward(np.array(ids)[None, :])
        h.head_mask = np.ones((h.spec.layers, h.spec.heads), np.float32)
        h.head_mask[0, 0] = 0
        masked, _ = h.forward(np.array(ids)[None, :])
        h.head_mask = None
        self.assertGreater(float(np.abs(base - masked).max()), 1e-6)

    def test_activation_autopsy_hf(self):
        h = self._handle()
        import numpy as np
        from models.autopsy import activation_autopsy
        ids = h.tokenizer.encode("Hello world")[:8]
        r = activation_autopsy(h, np.array([ids]))
        self.assertIn("resid_std", r["layers"][0])


class TestM2Transplants(unittest.TestCase):
    """Component 'cut and copy' must move exactly the right tensors."""

    def setUp(self):
        from models.synthetic import make_micro, master_tokenizer
        self.tok = master_tokenizer()
        self.src = make_micro("ts", d_model=32, n_layers=2, n_heads=4,
                              vocab_size=self.tok.vocab_size, tokenizer=self.tok, seed=1)
        self.dst = make_micro("td", d_model=32, n_layers=2, n_heads=4,
                              vocab_size=self.tok.vocab_size, tokenizer=self.tok, seed=2)

    def test_head_transplant_moves_exactly_head_slice(self):
        from component_extraction.transplant import transplant_component, Component
        before = {k: v.copy() for k, v in self.dst.state_dict().items()}
        comp = Component(source_model="ts", architecture="micro-gpt",
                         target_capability="math", region="L0.attn.head1",
                         level="HEAD", evidence_delta=-10.0, confidence="low",
                         dependencies="residual", tokenizer="t", hidden_dimension=32)
        transplant_component(self.dst, self.src, comp)
        S, D = self.src.state_dict(), self.dst.state_dict()
        d, H, e, h0 = 32, 4, 8, 8
        # qkv columns for head 1 (all three blocks)
        for off in (0, d, 2 * d):
            cols = slice(off + h0, off + h0 + e)
            self.assertTrue(np.allclose(D["L0.attn.qkv.W"][:, cols], S["L0.attn.qkv.W"][:, cols]))
            self.assertTrue(np.allclose(D["L0.attn.qkv.b"][cols], S["L0.attn.qkv.b"][cols]))
        self.assertTrue(np.allclose(D["L0.attn.o.W"][h0:h0 + e, :], S["L0.attn.o.W"][h0:h0 + e, :]))
        # untouched tensors must be unchanged
        self.assertTrue(np.allclose(D["L1.mlp.1.W"], before["L1.mlp.1.W"]))
        self.assertTrue(np.allclose(D["L0.ln1.g"], before["L0.ln1.g"]))

    def test_incompatible_transplant_raises(self):
        from component_extraction.transplant import transplant_component, Component
        from models.synthetic import make_micro
        big = make_micro("tb", d_model=48, n_layers=2, n_heads=6,
                         vocab_size=self.tok.vocab_size, tokenizer=self.tok, seed=3)
        comp = Component(source_model="ts", architecture="micro-gpt",
                         target_capability="math", region="L0", level="LAYER",
                         evidence_delta=0, confidence="low", dependencies="",
                         tokenizer="t", hidden_dimension=32)
        with self.assertRaises(ValueError):
            transplant_component(big, self.src, comp)

    def test_abc_intervention_restores_recipient(self):
        from component_extraction.transplant import transplant_component, Component, abc_intervention
        comp = Component(source_model="ts", architecture="micro-gpt",
                         target_capability="math", region="L0", level="LAYER",
                         evidence_delta=0, confidence="low", dependencies="",
                         tokenizer="t", hidden_dimension=32)
        snap = {k: v.copy() for k, v in self.dst.state_dict().items()}
        out = abc_intervention(self.src, self.dst, comp, "micro-lang",
                               max_items=8)
        after = {k: v.copy() for k, v in self.dst.state_dict().items()}
        for k in snap:
            self.assertTrue(np.allclose(snap[k], after[k]),
                            "recipient weights were not restored after A/B/C")


class TestM2Genome(unittest.TestCase):
    def test_genome_roundtrip_and_candidates(self):
        import tempfile
        from capability_genome.genome import (CapabilityGenome, GenomeRecord,
                                              ComponentRef)
        g = CapabilityGenome(model_name="m", arch="micro-gpt", fingerprint="x",
                             param_count=111)
        g.baselines = {"micro-math": 40.0}
        g.random_baseline = {"micro-math": 25.0}
        g.add(GenomeRecord(component=ComponentRef("HEAD", "L0.attn.head1", 0,
                                                  module="attn", index=1).to_dict(),
                           capability="math", suite="micro-math",
                           intervention="ablate", baseline=40.0,
                           intervened=30.0, delta=10.0, n=40, confidence="medium"))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.json")
            g.save(p)
            g2 = CapabilityGenome.load(p)
        self.assertEqual(g2.model_name, "m")
        self.assertEqual(len(g2.candidates(capability="math", min_abs_delta=5)), 1)
        self.assertEqual(g2.random_baseline["micro-math"], 25.0)

    def test_pool_extraction_and_combinations(self):
        from component_extraction.transplant import (extract_pool,
                                                     candidate_combinations,
                                                     Component)
        comps = []
        for i, (src, region, delta) in enumerate([
                ("A", "L0.attn.head0", -20), ("B", "L1", -12),
                ("A", "L0.mlp.chan[0:8]", -6), ("C", "L0", -2)]):
            comps.append(Component(source_model=src, architecture="micro-gpt",
                                   target_capability="math", region=region,
                                   level="HEAD" if "head" in region else "LAYER",
                                   evidence_delta=delta, confidence="low",
                                   dependencies="", tokenizer="t", hidden_dimension=32))
        class FakeGenome:
            def __init__(self, recs):
                self._r = recs
            def candidates(self, capability=None, min_abs_delta=0.0, levels=None):
                return [r for r in self._r if abs(r.delta) >= min_abs_delta]
        class FakeRec:
            def __init__(self, c):
                self.component = {"path": c.region, "level": c.level, "layer": 0,
                                  "module": None, "index": None, "dims": None}
                self.delta = c.evidence_delta
                self.suite = "micro-math"
                self.n = 30
                self.confidence = "low"
        recs_by_src = {}
        for c in comps:
            recs_by_src.setdefault(c.source_model, []).append(FakeRec(c))
        genomes = {src: FakeGenome(rs) for src, rs in recs_by_src.items()}
        class NS:
            def __init__(self, n):
                self.name = n
                self.spec = type("S", (), {"arch": "micro-gpt", "vocab": 141,
                                           "hidden": 32})()
        models = {n: NS(n) for n in genomes}
        pool = extract_pool(genomes, models, "math", "micro-math",
                            top_k=3, min_abs_delta=4.0)
        self.assertEqual(len(pool), 3)
        self.assertEqual(pool[0].region, "L0.attn.head0")
        self.assertEqual(pool[0].region, "L0.attn.head0")
        combos = candidate_combinations(pool, max_size=2, max_candidates=4)
        self.assertTrue(1 <= len(combos) <= 4)


class TestM2Evolution(unittest.TestCase):
    def test_evolve_runs_and_improves_or_explores(self):
        from models.synthetic import make_micro, master_tokenizer
        from capability_genome.evolution import evolve
        tok = master_tokenizer()
        models = {f"m{i}": make_micro(f"m{i}", d_model=32, n_layers=2, n_heads=4,
                                      vocab_size=tok.vocab_size, tokenizer=tok, seed=i)
                  for i in range(3)}
        base = models["m0"]
        out = evolve(base, models, [("m1", "L0"), ("m2", "L1.attn.head0")],
                     "micro-lang", ["micro-know"], generations=2, pop_size=4,
                     max_items=8)
        self.assertIn("best_individual", out)
        self.assertGreater(len(out["history"]), 0)


class TestM2Selector(unittest.TestCase):
    def test_param_estimate_sane(self):
        from models.selector import estimate_params_bytes
        gpt2_cfg = {"n_layer": 12, "n_embd": 768, "vocab_size": 50257, "n_inner": 3072}
        b = estimate_params_bytes(gpt2_cfg)
        # gpt2 has 124M params; our rough estimate should be within ~30%
        self.assertTrue(80e6 < b < 180e6, f"estimate {b/1e6:.0f}M out of range")
