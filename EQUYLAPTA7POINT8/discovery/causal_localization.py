"""discovery/causal_localization.py

Implements native causal localization and baseline characterization for donor models in EQUYLAPTA 7.8.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, Tuple
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import load_model, MicroTransformer
from benchmarking.suites import micro_suite
from benchmarking.harness import eval_suite


def evaluate_native_localization(model_name: str = "e4-math-4L", domain: str = "math",
                                 target_head: Tuple[int, int] = (0, 2),
                                 discovery_seed: int = 777) -> Dict[str, Any]:
    """Evaluates intact capability, candidate head ablation, restoration, and specificity."""
    model = load_model(model_name)
    suite = micro_suite(domain, n=20, seed=discovery_seed)

    # 1. Intact baseline
    model.head_mask = None
    res_intact = eval_suite(model, suite, max_items=20)
    acc_intact = float(res_intact["accuracy"]) * 100.0

    # 2. Candidate head ablation
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
    rest_err = abs(acc_intact - acc_rest)

    # 4. Specificity over non-specialist heads
    other_drops = []
    for l in range(model.spec.layers):
        for h in range(model.spec.heads):
            if l == l_tgt and h == h_tgt:
                continue
            model.head_mask = np.ones((model.spec.layers, model.spec.heads), dtype=np.float32)
            model.head_mask[l, h] = 0.0
            r = eval_suite(model, suite, max_items=20)
            other_drops.append(acc_intact - float(r["accuracy"]) * 100.0)

    mean_other = float(np.mean(other_drops)) if other_drops else 0.0
    specificity = float(causal_drop / mean_other) if mean_other > 0 else 1.0

    return {
        "model_name": model_name,
        "domain": domain,
        "target_head": f"L{l_tgt}_head_{h_tgt}",
        "intact_accuracy": round(acc_intact, 2),
        "ablated_accuracy": round(acc_abl, 2),
        "causal_drop_pp": round(causal_drop, 2),
        "restored_accuracy": round(acc_rest, 2),
        "restoration_error_pp": round(rest_err, 2),
        "mean_other_heads_drop_pp": round(mean_other, 2),
        "specificity_ratio": round(specificity, 2),
        "status": "LOCALIZED" if causal_drop >= 20.0 and specificity >= 1.2 else "DIFFUSE"
    }
