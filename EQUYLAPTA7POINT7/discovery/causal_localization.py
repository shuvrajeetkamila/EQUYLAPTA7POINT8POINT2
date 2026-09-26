"""discovery/causal_localization.py

Implements native causal localization, ablation, restoration, and specificity analysis
for donor models in EQUYLAPTA 7.7.
"""
from __future__ import annotations

import copy
import sys
import os
from typing import Dict, Any, List, Tuple
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import load_model, MicroTransformer
from benchmarking.suites import micro_suite
from benchmarking.harness import eval_suite


def evaluate_native_localization(model_name: str, domain: str, target_head: Tuple[int, int],
                                 discovery_seed: int = 777) -> Dict[str, Any]:
    """Evaluates native performance, candidate component ablation, restoration,

    random head controls, and specificity ratios.
    """
    model = load_model(model_name)
    suite = micro_suite(domain, n=20, seed=discovery_seed)

    # 1. Intact model baseline
    model.head_mask = None
    res_intact = eval_suite(model, suite, max_items=20)
    acc_intact = float(res_intact["accuracy"]) * 100.0

    # 2. Candidate head zero-ablation
    l_tgt, h_tgt = target_head
    model.head_mask = np.ones((model.spec.layers, model.spec.heads), dtype=np.float32)
    model.head_mask[l_tgt, h_tgt] = 0.0
    res_abl = eval_suite(model, suite, max_items=20)
    acc_abl = float(res_abl["accuracy"]) * 100.0
    causal_drop = acc_intact - acc_abl

    # 3. Restoration
    model.head_mask[l_tgt, h_tgt] = 1.0
    res_rest = eval_suite(model, suite, max_items=20)
    acc_rest = float(res_rest["accuracy"]) * 100.0
    restoration_error = abs(acc_intact - acc_rest)

    # 4. Layer-matched and random head controls
    other_head_drops = []
    layer_matched_drop = 0.0
    for l in range(model.spec.layers):
        for h in range(model.spec.heads):
            if l == l_tgt and h == h_tgt:
                continue
            model.head_mask = np.ones((model.spec.layers, model.spec.heads), dtype=np.float32)
            model.head_mask[l, h] = 0.0
            r = eval_suite(model, suite, max_items=20)
            drop = acc_intact - float(r["accuracy"]) * 100.0
            other_head_drops.append(drop)
            if l == l_tgt and h == (h_tgt + 1) % model.spec.heads:
                layer_matched_drop = drop

    mean_random_drop = float(np.mean(other_head_drops)) if other_head_drops else 0.0
    specificity_ratio = float(causal_drop / mean_random_drop) if mean_random_drop > 0 else 1.0

    return {
        "model_name": model_name,
        "domain": domain,
        "target_head": f"L{l_tgt}_head_{h_tgt}",
        "intact_accuracy": round(acc_intact, 2),
        "ablated_accuracy": round(acc_abl, 2),
        "causal_drop_pp": round(causal_drop, 2),
        "restored_accuracy": round(acc_rest, 2),
        "restoration_error_pp": round(restoration_error, 2),
        "layer_matched_control_drop_pp": round(layer_matched_drop, 2),
        "mean_other_heads_drop_pp": round(mean_random_drop, 2),
        "specificity_ratio": round(specificity_ratio, 2),
        "status": "LOCALIZED" if causal_drop >= 20.0 and specificity_ratio >= 1.2 else "WEAK_OR_DIFFUSE"
    }


if __name__ == "__main__":
    res = evaluate_native_localization("e4-math-4L", "math", (0, 2))
    print("Native Causal Localization:", res)
