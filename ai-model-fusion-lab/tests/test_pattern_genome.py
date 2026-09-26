"""Pattern Genome tests (Milestone 4): schema, registry, detectors, nulls,
interventions, minimal-unit search, circuit transfer plumbing."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tiny(seed=0, d=32):
    from models.synthetic import make_micro, master_tokenizer
    tok = master_tokenizer()
    m = make_micro("pg-tiny", d_model=d, n_layers=2, n_heads=4,
                   vocab_size=tok.vocab_size, tokenizer=tok, seed=seed)
    return m


class TestSchema(unittest.TestCase):
    def test_pattern_record_roundtrip_and_validation(self):
        from pattern_genome.schema import PatternRecord
        r = PatternRecord("P001", "Fibonacci", "P1_SEQUENCE", "d", "h",
                          measurable_features=["x"], discovery_algorithm="seq",
                          null_model="random sets", minimum_evidence=3,
                          phase="B", detector="sequence_gaps").validate()
        d = r.to_dict()
        r2 = PatternRecord.from_dict(d).validate()
        self.assertEqual(r2.pattern_id, "P001")
        bad = PatternRecord("P1", "x", "P1_SEQUENCE", "d", "h",
                            null_model="n", phase="A")
        with self.assertRaises(ValueError):
            bad.validate()
        no_null = PatternRecord("P001", "x", "P1_SEQUENCE", "d", "h",
                                phase="A", null_model="")
        with self.assertRaises(ValueError):
            no_null.validate()

    def test_evidence_ladder_mapping(self):
        from pattern_genome.schema import evidence_level, status_from_evidence
        b0 = {}
        self.assertEqual(evidence_level(b0), 0)
        b1 = {"detection": {"significant": True}}
        self.assertEqual(evidence_level(b1), 1)
        b3 = {"detection": {"significant": True},
              "task_correlation": {"significant": True},
              "ablation": {"significant": True, "effect": -12.0,
                           "effect_direction": "degrades"}}
        self.assertEqual(evidence_level(b3), 3)
        self.assertEqual(status_from_evidence(3, b3), "PROMISING")
        # CONTRADICTED: ablating IMPROVES the capability significantly
        bc = {"detection": {"significant": True},
              "ablation": {"significant": True, "effect": 9.0,
                           "effect_direction": "improves"}}
        self.assertEqual(status_from_evidence(3, bc), "CONTRADICTED")
        # transfer needs controls AND positivity
        bt = dict(b3)
        bt["transfer"] = {"beats_random": True, "beats_capacity": True,
                          "positive": True, "status": "CROSS_MODEL_TRANSFER"}
        self.assertEqual(evidence_level(bt), 6)
        bt["reproduction"] = {"reproduced": True}
        self.assertEqual(evidence_level(bt), 7)

    def test_transfer_score_components_raw(self):
        from pattern_genome.schema import transfer_score_components as tsc
        out = tsc(held_out_gain=10.0, random_control_gain=3.0,
                  capacity_control_gain=2.0, interference_penalty=0.5,
                  parameter_cost=4144)
        self.assertEqual(out["composite"], 4.5)
        self.assertEqual(out["held_out_gain"], 10.0)   # raw kept
        self.assertEqual(out["parameter_cost"], 4144)


class TestDefinitions(unittest.TestCase):
    def test_100_patterns_registered(self):
        from pattern_genome.patterns.definitions import all_records, by_phase
        recs = all_records()
        self.assertEqual(len(recs), 100)
        self.assertEqual(len(by_phase("A")), 24)
        # §21 Phase A must contain the high-value structure list
        ids = {r.pattern_id for r in by_phase("A")}
        for need in ["P021", "P022", "P024", "P025", "P037", "P038", "P040",
                     "P036", "P044", "P045", "P055", "P080", "P027", "P081"]:
            self.assertIn(need, ids)

    def test_every_pattern_has_null_and_detector(self):
        from pattern_genome.patterns.definitions import all_records
        for r in all_records():
            self.assertTrue(r.null_model, r.pattern_id)
            self.assertTrue(r.detector, r.pattern_id)
            self.assertTrue(r.failure_conditions, r.pattern_id)


class TestDetectors(unittest.TestCase):
    def test_sequence_gap_test_fib_vs_uniform(self):
        from pattern_genome.graph_analysis import sequence_gap_test
        # planted Fibonacci gaps: positions 0,1,2,4,7 (gaps 1,1,2,3)
        res = sequence_gap_test([0, 1, 2, 4, 7], K=300, seed=1, span=12)
        self.assertTrue(res["significant"])
        self.assertEqual(res["best_fit"], "fibonacci")
        # exact arithmetic spacing: detector must name 'uniform' (it IS more
        # regular than random — significance means "more regular than chance")
        res2 = sequence_gap_test([0, 2, 4, 6], K=300, seed=1, span=12)
        self.assertEqual(res2["best_fit"], "uniform")
        # irregular positions: no named sequence beats chance reliably
        res3 = sequence_gap_test([0, 1, 4, 6, 11], K=300, seed=3, span=12)
        self.assertFalse(res3["significant"])

    def test_ratio_enrichment_phi(self):
        from pattern_genome.graph_analysis import ratio_enrichment
        phi = 1.6180339887
        ratios = [phi * (1 + 0.01 * k) for k in range(-4, 5)] * 3
        res = ratio_enrichment(ratios, K=300, seed=2)
        self.assertTrue(res["significant"])
        self.assertGreater(res["phi_fraction"], 0.8)

    def test_magnitude_family_fit_powerlaw(self):
        from pattern_genome.graph_analysis import magnitude_family_fit
        m = np.array([x ** -1.5 for x in range(1, 21)]) * 100
        res = magnitude_family_fit(m)
        self.assertEqual(res["best"], "power_law")
        self.assertGreater(res["fits"]["power_law"]["r2"], 0.9)

    def test_graph_motifs_on_hand_built_diamond(self):
        from pattern_genome.graph_analysis import (Graph, PATNode, detect_fan_in,
                                                   detect_fan_out, _paths,
                                                   detect_diamonds,
                                                   detect_hub, detect_bottleneck)
        g = Graph()
        for nid, kind in [("embed", "embed"), ("attn0", "attn"), ("mlp0", "mlp"),
                          ("resid0", "resid"), ("final", "final")]:
            g.add_node(PATNode(nid, kind, layer=0 if nid.endswith("0") else -1,
                               ablatable=kind in ("attn", "mlp")))
        for (u, v), w in [(("embed", "attn0"), .3), (("embed", "mlp0"), .2),
                          (("attn0", "resid0"), .4), (("mlp0", "resid0"), .5),
                          (("resid0", "final"), .6)]:
            g.add_edge(u, v, w)
        self.assertEqual(sorted(detect_fan_in(g, 2).get("resid0", [])),
                         ["attn0", "mlp0"])
        self.assertGreaterEqual(len(detect_fan_out(g, 2).get("embed", [])), 2)
        self.assertEqual(len(_paths(g)), 2)
        d = detect_diamonds(g)
        self.assertTrue(any(u == "embed" and v == "resid0" for u, v, _ in d))
        self.assertEqual(detect_hub(g), "resid0")
        self.assertEqual(detect_bottleneck(g), "resid0")

    def test_subspace_and_sparsity(self):
        from pattern_genome.graph_analysis import subspace_stats, hoyer_sparsity
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 32)) @ np.diag([10, 8, 6] + [0.1] * 29)
        st = subspace_stats(x)
        self.assertLess(st["participation_ratio"], 8.0)
        v = np.zeros(64); v[:3] = 1.0
        self.assertGreater(hoyer_sparsity(v), 0.8)

    def test_p_value_extreme(self):
        from pattern_genome.controls import p_value
        self.assertAlmostEqual(p_value(10.0, [1, 2, 3, 4], "greater"), 1 / 5)


class TestInterventions(unittest.TestCase):
    def test_eval_with_members_and_battery_identity(self):
        from pattern_genome.intervention import (eval_with_members,
                                                 ablation_battery)
        m = _tiny(1)
        members = [{"level": "MODULE", "layer": 0, "module": "mlp"},
                   {"level": "MODULE", "layer": 1, "module": "mlp"}]
        base = eval_with_members(m, [], "math", 16, 400)
        full = eval_with_members(m, members, "math", 16, 400)
        bat = ablation_battery(m, members, "math", n=16, seed=400,
                               other_domains=["code"])
        self.assertAlmostEqual(bat["baseline"], base, places=6)
        self.assertAlmostEqual(bat["effect"], base - full, places=6)
        self.assertIn("code", bat["other_domain_effects"])
        self.assertEqual(set(bat["per_member_effect"].keys()),
                         {"L0.mlp", "L1.mlp"})

    def test_minimal_unit_search_structure(self):
        from pattern_genome.intervention import minimal_unit_search
        m = _tiny(2)
        members = [{"level": "MODULE", "layer": 0, "module": "attn"},
                   {"level": "MODULE", "layer": 1, "module": "mlp"},
                   {"level": "MODULE", "layer": 1, "module": "attn"}]
        mu = minimal_unit_search(m, members, "math", n=16, seed=400, rho=0.8)
        keys = {"L0.attn", "L1.mlp", "L1.attn"}
        self.assertTrue(set(mu["minimal_unit"]).issubset(keys))
        self.assertLessEqual(mu["n_members_minimal"], 3)
        self.assertIn("verified", mu)
        self.assertIsInstance(mu["minimal_effect"], (int, float))
        if mu["full_circuit_effect"] > 0:
            self.assertLessEqual(mu["minimal_effect"], mu["full_circuit_effect"] + 1e-6)

    def test_restoration_battery_runs(self):
        from pattern_genome.intervention import (restoration_battery,
                                                 eval_with_members)
        from models.synthetic import finetune_from
        from benchmarking.suites import micro_suite
        m = _tiny(3)
        texts = []
        for sd in (200, 201):
            for it in micro_suite("math", n=60, seed=sd).items:
                texts.append(it.prompt + " " + it.options[it.answer])
        finetune_from(m, texts, epochs=24, lr=2e-3, seed=1)
        members = [{"level": "MODULE", "layer": 0, "module": "mlp"},
                   {"level": "MODULE", "layer": 1, "module": "mlp"}]
        res = restoration_battery(m, members, "math", n=12, seed=400)
        self.assertIn("graded_restore_acc", res)
        self.assertEqual(set(res["graded_restore_acc"].keys()),
                         {"L0.mlp", "L1.mlp"})
        # MECHANICAL GUARANTEE: patching the removed module's clean output
        # back into the ablated run must reproduce the clean run exactly
        # (mediation identity), whatever the training state.
        self.assertAlmostEqual(res["full_restore_acc"], res["clean_acc"],
                               delta=0.01)
        # honest-None logic: recovered_pct is None iff no accuracy dependence
        depends = abs(res["clean_acc"] - res["ablated_acc"]) > 1e-6
        self.assertEqual(res.get("recovered_pct") is not None, depends)
        if res.get("recovered_pct") is not None:
            self.assertLessEqual(res["recovered_pct"], 105.0)

    def test_positive_intervention_battery(self):
        from pattern_genome.intervention import positive_intervention_battery
        m = _tiny(4)
        members = [{"level": "MODULE", "layer": 0, "module": "mlp"}]
        out = positive_intervention_battery(m, members, "math", n=16, seed=401)
        self.assertIn("x1.5", out["arms"])
        self.assertIn("x2.0", out["arms"])
        self.assertIsInstance(out["significant"], bool)


class TestCircuitPlumbing(unittest.TestCase):
    def test_member_component_roundtrip_insertion(self):
        from pattern_genome.transfer import (circuit_members_payload,
                                             insert_circuit, union_locations)
        from component_transfer.apply import snapshot, restore_snapshot
        a, b = _tiny(5), _tiny(6)
        members = [{"level": "MODULE", "layer": 0, "module": "mlp"},
                   {"level": "HEAD", "layer": 1, "module": "attn", "index": 2}]
        comps, tensors = circuit_members_payload(a, members, "math")
        self.assertEqual(set(tensors.keys()), {"L0.mlp", "L1.h2"})
        snap = snapshot(b)
        rec = insert_circuit(b, members, tensors)
        self.assertTrue(len(rec["applied"]) >= 6)
        db, da = b.state_dict(), a.state_dict()
        self.assertTrue(np.allclose(db["L0.mlp.1.W"], da["L0.mlp.1.W"]))
        restore_snapshot(b, snap)
        locs = union_locations(b, members)
        self.assertTrue(any(k.startswith("L0.mlp:") for k in locs))
        self.assertTrue(any(k.startswith("L1.h2:") for k in locs))

    def test_run_circuit_transfer_smoke(self):
        from pattern_genome.schema import CircuitRecord
        from pattern_genome.transfer import run_circuit_transfer
        from models.synthetic import build_profile_corpus
        a, b = _tiny(7), _tiny(8)
        c = CircuitRecord(circuit_id="ctest", pattern_ids=["P037"],
                          model=a.name, capability="math",
                          members=[{"level": "MODULE", "layer": 0, "module": "mlp"},
                                   {"level": "MODULE", "layer": 1, "module": "mlp"}],
                          kind="bottleneck")
        train = build_profile_corpus("nomath-base", 40, seed=31)
        calib = build_profile_corpus("nomath-base", 20, seed=32)
        probe = build_profile_corpus("nomath-base", 20, seed=33)
        R = run_circuit_transfer(a, b, c, "math", "math",
                                 ["code", "reason"], train, calib, probe,
                                 cfg={"n_items": 12, "eval_seeds": [1],
                                      "refit_steps": 5})
        self.assertIn(R["verdict"]["status"], (
            "CROSS_MODEL_TRANSFER", "CROSS_MODEL_TRANSFER+INTERFERENCE",
            "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE+INTERFERENCE",
            "PARTIAL_TRANSFER", "PARTIAL_TRANSFER+INTERFERENCE",
            "NO_EVIDENCE_OF_SPECIALIZED_TRANSFER",
            "NO_EVIDENCE_OF_SPECIALIZED_TRANSFER+INTERFERENCE",
            "TRANSFER_ATTEMPT_FAILED", "TRANSFER_ATTEMPT_FAILED+INTERFERENCE"))
        self.assertIn("held_out_gain", R["transfer_score"])
        self.assertIn("composite", R["transfer_score"])


class TestRegistry(unittest.TestCase):
    def test_note_patterns_and_persistence(self):
        from pattern_genome.registry import PatternRegistry
        from pattern_genome.schema import CircuitRecord
        reg = PatternRegistry()
        c = CircuitRecord(circuit_id="c1", pattern_ids=["P024"], model="m",
                          capability="math", members=[])
        battery = {"detection": {"significant": True, "p": 0.01},
                   "task_correlation": {"significant": True, "selectivity": 9.0},
                   "ablation": {"significant": True, "effect": -14.0,
                                "effect_direction": "degrades",
                                "effect_specific": True, "specificity": 9.0}}
        reg.note_patterns(["P024"], c, battery)
        self.assertEqual(reg.results["P024"]["evidence_level"], 3)
        self.assertEqual(reg.results["P024"]["status"], "PROMISING")
        reg.set_unsupported("P024", "null not beaten")
        self.assertEqual(reg.results["P024"]["status"], "UNSUPPORTED")
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "pg.json")
            reg.save(p)
            reg2 = PatternRegistry.load(p)
            self.assertEqual(reg2.results["P024"]["status"], "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
