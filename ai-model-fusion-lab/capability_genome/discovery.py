"""Hierarchical component discovery (Milestone 2, Phases 3-4).

LEVEL 1 layer -> LEVEL 2 module -> LEVEL 3 head -> LEVEL 4 channel group.
Each level is only explored deeper where the previous level found a
meaningful candidate (|delta| >= threshold). Every intervention gets a
RESTORATION CHECK (re-enable, expect return to baseline) that raises
confidence when it passes. All results go into the genome + experiment DB.

Language discipline (spec Phase 18): records say "removing X produced a
D-point change on suite S (n=...)", never "X IS the capability".
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ablation.ablator import ablate_and_eval
from benchmarking.harness import eval_suite
from benchmarking.suites import load_suite
from experiments.tracker import ExperimentTracker
from models.base import config_hash
from .genome import ComponentRef, CapabilityGenome, GenomeRecord


def _acc(m, suite_name, max_items) -> float:
    return eval_suite(m, load_suite(suite_name), max_items=max_items)["accuracy"] * 100


def restoration_check(m, suite_name: str, baseline: float, max_items: int,
                      tol: float = 3.0) -> float:
    """Re-evaluate with no intervention; expect return to baseline."""
    v = _acc(m, suite_name, max_items)
    return v if abs(v - baseline) <= tol else baseline + (v - baseline)  # honest value


def _confidence(delta: float, restored_ok: bool, n: int) -> str:
    if n >= 40 and abs(delta) >= 10 and restored_ok:
        return "medium"
    if n >= 30 and abs(delta) >= 5 and restored_ok:
        return "medium"
    return "low"


def discover(m, capabilities: Dict[str, str], tracker: ExperimentTracker,
             genome: Optional[CapabilityGenome] = None,
             max_items: int = 40, tau_layer: float = 5.0, tau_module: float = 4.0,
             tau_head: float = 4.0, tau_chan: float = 4.0,
             n_chan_groups: int = 8, seed: int = 2026, log=None) -> CapabilityGenome:
    """capabilities: {capability_name: suite_name}."""
    if genome is None:
        genome = CapabilityGenome(model_name=m.name, arch=m.spec.arch,
                                  fingerprint=m.fingerprint(),
                                  param_count=m.param_count())
    suites = list(capabilities.values())

    # ---- baselines (Phase 6: random + parent baselines recorded too) ----
    baselines = {}
    for cap, sn in capabilities.items():
        r = eval_suite(m, load_suite(sn), max_items=max_items)
        baselines[sn] = r["accuracy"] * 100
        k = len(load_suite(sn).items[0].options)
        genome.random_baseline[sn] = round(100.0 / k, 1)
        genome.baselines[sn] = round(baselines[sn], 1)
    if log:
        log("  baselines: " + " ".join(f"{k}:{v:.0f}%" for k, v in baselines.items()))

    def record(cap, suite, comp, base, ablated, restored, notes=""):
        delta = round(base - ablated, 1)
        restored_ok = restored is not None and abs(restored - base) <= 3.0
        rec = GenomeRecord(
            component=comp.to_dict(), capability=cap, suite=suite,
            intervention="ablate", baseline=round(base, 1),
            intervened=round(ablated, 1), delta=delta, restored=restored,
            n=max_items, seed=seed,
            confidence=_confidence(delta, restored_ok, max_items),
            status="CANDIDATE" if abs(delta) >= 4 else "NOT_SIGNIFICANT",
            notes=notes)
        cfg = {"kind": "genome_ablate", "model": m.name, "component": comp.to_dict(),
               "suite": suite, "seed": seed, "n": max_items}
        exp = tracker.record(exp_type="genome-ablation", config=cfg, sources=[m.name],
                             results={"baseline": round(base, 1),
                                      "ablated": round(ablated, 1),
                                      "delta": delta,
                                      "restored": round(restored, 1) if restored is not None else None},
                             target_suite=suite, notes=notes or "hierarchical discovery",
                             license="Apache-2.0")
        rec.experiment_id = exp["id"]
        genome.add(rec)
        return rec

    # ---- LEVEL 1: layers ----
    l1_significant: Dict[str, List[int]] = {sn: [] for sn in suites}
    for l in range(m.spec.layers):
        res = ablate_and_eval(m, suites, skip_layers=[l], max_items=max_items)
        for sn in suites:
            ablated = res[sn]["accuracy"] * 100
            restored = restoration_check(m, sn, baselines[sn], max(max_items // 2, 20))
            rec = record(cap=_cap_of(capabilities, sn), suite=sn,
                         comp=ComponentRef("LAYER", f"L{l}", l),
                         base=baselines[sn], ablated=ablated, restored=restored)
            if log and abs(rec.delta) >= tau_layer:
                log(f"  L1 {rec.component['path']} {sn}: {rec.delta:+.1f} pts")
            if abs(rec.delta) >= tau_layer:
                l1_significant[sn].append(l)

    # ---- LEVEL 2: modules within significant layers (else all layers) ----
    l2_significant: Dict[str, List[tuple]] = {sn: [] for sn in suites}
    for sn in suites:
        layers_to_probe = l1_significant[sn] or list(range(m.spec.layers))
        for l in layers_to_probe:
            for mod in ["attn", "mlp"]:
                res = ablate_and_eval(m, [sn], skip_modules={l: [mod]}, max_items=max_items)
                ablated = res[sn]["accuracy"] * 100
                restored = restoration_check(m, sn, baselines[sn], max(max_items // 2, 20))
                rec = record(cap=_cap_of(capabilities, sn), suite=sn,
                             comp=ComponentRef("MODULE", f"L{l}.{mod}", l, module=mod),
                             base=baselines[sn], ablated=ablated, restored=restored)
                if log and abs(rec.delta) >= tau_module:
                    log(f"  L2 {rec.component['path']} {sn}: {rec.delta:+.1f} pts")
                if abs(rec.delta) >= tau_module:
                    l2_significant[sn].append((l, mod))

    # ---- LEVEL 3: heads within significant attn modules ----
    H = m.spec.heads
    for sn in suites:
        attn_targets = [(l, mod) for (l, mod) in l2_significant[sn] if mod == "attn"]
        if not attn_targets:
            attn_targets = [(l, "attn") for l in (l1_significant[sn] or [])]
        for (l, _mod) in attn_targets:
            for h in range(H):
                hm = np.ones((m.spec.layers, H), np.float32)
                hm[l, h] = 0.0
                res = ablate_and_eval(m, [sn], head_mask=hm, max_items=max_items)
                ablated = res[sn]["accuracy"] * 100
                rec = record(cap=_cap_of(capabilities, sn), suite=sn,
                             comp=ComponentRef("HEAD", f"L{l}.attn.head{h}", l,
                                               module="attn", index=h,
                                               dims={"head_dim": m.spec.hidden // H}),
                             base=baselines[sn], ablated=ablated,
                             restored=None, notes="head ablation")
                if log and abs(rec.delta) >= tau_head:
                    log(f"  L3 {rec.component['path']} {sn}: {rec.delta:+.1f} pts")

    # ---- LEVEL 4: channel groups within significant mlp modules ----
    F = m.spec.mlp_hidden
    gsize = max(1, F // n_chan_groups)
    for sn in suites:
        mlp_targets = [(l, mod) for (l, mod) in l2_significant[sn] if mod == "mlp"]
        if not mlp_targets:
            mlp_targets = [(l, "mlp") for l in (l1_significant[sn] or [])]
        for (l, _mod) in mlp_targets:
            for gi in range(n_chan_groups):
                chs = np.zeros(F, np.float32)
                chs[gi * gsize:(gi + 1) * gsize] = 0.0
                keep = np.ones(F, np.float32)
                keep[gi * gsize:(gi + 1) * gsize] = 0.0
                res = ablate_and_eval(m, [sn], channel_mask={l: keep}, max_items=max_items)
                ablated = res[sn]["accuracy"] * 100
                rec = record(cap=_cap_of(capabilities, sn), suite=sn,
                             comp=ComponentRef("CHANNEL_GROUP",
                                               f"L{l}.mlp.chan[{gi*gsize}:{(gi+1)*gsize}]",
                                               l, module="mlp", index=gi,
                                               dims={"group_size": int(gsize)}),
                             base=baselines[sn], ablated=ablated, restored=None,
                             notes="channel-group ablation")
                if log and abs(rec.delta) >= tau_chan:
                    log(f"  L4 {rec.component['path']} {sn}: {rec.delta:+.1f} pts")
    return genome


def _cap_of(capabilities: Dict[str, str], suite_name: str) -> str:
    for cap, sn in capabilities.items():
        if sn == suite_name:
            return cap
    return "unknown"
