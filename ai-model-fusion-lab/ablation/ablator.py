"""[WORKING] Ablation experiments + capability localization.

Mechanics: mask layers / modules / heads, re-run held-out suites, record
deltas. FINDINGS ARE EVIDENCE, NOT TRUTH (spec §23): we report "removing X
changed suite Y by Z points (n=..)", never "X IS the capability".
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List

import numpy as np

from benchmarking.harness import eval_suite
from benchmarking.suites import load_suite


def ablate_and_eval(m, suite_names: List[str], skip_layers=(), skip_modules=None,
                    head_mask=None, channel_mask=None, max_items: int = 50) -> dict:
    m.skip_layers = list(skip_layers)
    m.skip_modules = skip_modules or {}
    m.head_mask = head_mask
    m.mlp_channel_mask = channel_mask
    try:
        res = {}
        for sn in suite_names:
            res[sn] = eval_suite(m, load_suite(sn), max_items=max_items)
        return res
    finally:
        m.skip_layers, m.skip_modules, m.head_mask = [], {}, None
        m.mlp_channel_mask = None


def layer_ablation_scan(m, suite_names: List[str], baseline: Dict[str, float],
                        max_items: int = 50, log=None) -> List[dict]:
    """Ablate each layer in turn; delta vs baseline per suite."""
    rows = []
    for l in range(m.spec.layers):
        res = ablate_and_eval(m, suite_names, skip_layers=[l], max_items=max_items)
        row = {"layer": l, "deltas": {}}
        for sn in suite_names:
            base = baseline[sn]
            abl = res[sn]["accuracy"]
            row["deltas"][sn] = round((base - abl) * 100, 1)  # positive = layer mattered
        rows.append(row)
        if log:
            log(f"  ablate L{l}: " + "  ".join(f"{sn}:{row['deltas'][sn]:+.1f}" for sn in suite_names))
    return rows


def module_ablation_scan(m, suite_names: List[str], baseline: Dict[str, float],
                         max_items: int = 50, log=None) -> List[dict]:
    rows = []
    for mod in ["attn", "mlp"]:
        per_layer = {l: [mod] for l in range(m.spec.layers)}
        res = ablate_and_eval(m, suite_names, skip_modules=per_layer, max_items=max_items)
        row = {"module": mod, "deltas": {}}
        for sn in suite_names:
            row["deltas"][sn] = round((baseline[sn] - res[sn]["accuracy"]) * 100, 1)
        rows.append(row)
        if log:
            log(f"  ablate all-{mod}: " + "  ".join(f"{sn}:{row['deltas'][sn]:+.1f}" for sn in suite_names))
    return rows


def head_ablation_scan(m, suite_name: str, baseline: float, max_items: int = 50,
                       log=None) -> List[dict]:
    """Ablate each attention head (L x H grid) on one suite."""
    L, H = m.spec.layers, m.spec.heads
    rows = []
    for l in range(L):
        for h in range(H):
            hm = np.ones((L, H), np.float32)
            hm[l, h] = 0.0
            res = ablate_and_eval(m, [suite_name], head_mask=hm, max_items=max_items)
            delta = round((baseline - res[suite_name]["accuracy"]) * 100, 1)
            rows.append({"layer": l, "head": h, "delta": delta})
            if log and abs(delta) > 0:
                log(f"  head L{l}H{h}: {delta:+.1f}")
    return rows


def localize_capabilities(m, suite_names: List[str], max_items: int = 50, log=None) -> dict:
    """Full capability-contribution map for one model (spec section 5)."""
    baseline = {}
    for sn in suite_names:
        baseline[sn] = eval_suite(m, load_suite(sn), max_items=max_items)["accuracy"]
    if log:
        log(f"  baseline: " + "  ".join(f"{sn}:{baseline[sn]*100:.0f}" for sn in suite_names))
    return {
        "model": m.name,
        "baseline": {sn: round(b * 100, 1) for sn, b in baseline.items()},
        "layer_scan": layer_ablation_scan(m, suite_names, baseline, max_items, log),
        "module_scan": module_ablation_scan(m, suite_names, baseline, max_items, log),
    }


def contribution_report(local: dict) -> str:
    """Evidence-phrased summary (never 'IS the coding layer')."""
    lines = [f"CAPABILITY CONTRIBUTION MAP — {local['model']} "
             f"(deltas = accuracy points lost when region removed; n per suite in raw log)"]
    for sn, base in local["baseline"].items():
        lines.append(f"\n  {sn} (baseline {base:.0f}%):")
        for row in local["layer_scan"]:
            d = row["deltas"].get(sn, 0.0)
            if d > 0:
                strength = "very high" if d >= 15 else "high" if d >= 7 else "modest"
                lines.append(f"    removing layer {row['layer']} -> -{d} pts  [{strength} contribution]")
        for row in local["module_scan"]:
            d = row["deltas"].get(sn, 0.0)
            lines.append(f"    removing all-{row['module']} blocks -> -{d} pts")
    return "\n".join(lines)
