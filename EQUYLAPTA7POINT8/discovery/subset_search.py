"""discovery/subset_search.py

Performs exhaustive 16-subset evaluation across candidate nodes {A, B, C, D}
to test global circuit minimality in donor e4-math-4L.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import load_model, MicroTransformer
from benchmarking.suites import micro_suite
from benchmarking.harness import eval_suite


NODE_NAMES = ["L0_head_2", "L0_mlp", "L1_head_1", "L2_head_3"]
NODE_PARAMS = {
    "L0_head_2": 1024,
    "L0_mlp": 32768,
    "L1_head_1": 1024,
    "L2_head_3": 1024
}


def evaluate_subset(model: MicroTransformer, active_nodes: List[str], suite: Any) -> float:
    """Evaluates model accuracy when ONLY active_nodes are enabled among candidate nodes."""
    model.head_mask = np.ones((model.spec.layers, model.spec.heads), dtype=np.float32)
    model.skip_modules = {}

    if "L0_head_2" not in active_nodes:
        model.head_mask[0, 2] = 0.0
    if "L1_head_1" not in active_nodes:
        model.head_mask[1, 1] = 0.0
    if "L2_head_3" not in active_nodes:
        model.head_mask[2, 3] = 0.0

    if "L0_mlp" not in active_nodes:
        model.skip_modules = {0: ["mlp"]}

    res = eval_suite(model, suite, max_items=20)
    model.head_mask = None
    model.skip_modules = {}
    return float(res["accuracy"]) * 100.0


def run_exhaustive_subset_search(model_name: str = "e4-math-4L", domain: str = "math",
                                  val_seed: int = 888, heldout_seed: int = 999) -> Dict[str, Any]:
    model = load_model(model_name)
    suite_val = micro_suite(domain, n=20, seed=val_seed)
    suite_held = micro_suite(domain, n=20, seed=heldout_seed)

    # Null circuit (all 4 candidate nodes ablated)
    acc_null_val = evaluate_subset(model, [], suite_val)
    acc_null_held = evaluate_subset(model, [], suite_held)

    # Intact baseline (all active)
    model.head_mask = None
    model.skip_modules = {}
    acc_intact_val = float(eval_suite(model, suite_val, max_items=20)["accuracy"]) * 100.0
    acc_intact_held = float(eval_suite(model, suite_held, max_items=20)["accuracy"]) * 100.0

    subset_records = []
    total_recovery_span = acc_intact_val - acc_null_val

    for k in range(len(NODE_NAMES) + 1):
        for combo in itertools.combinations(NODE_NAMES, k):
            active = list(combo)
            val_acc = evaluate_subset(model, active, suite_val)
            held_acc = evaluate_subset(model, active, suite_held)

            causal_rec = val_acc - acc_null_val
            pct_rec = (causal_rec / total_recovery_span * 100.0) if total_recovery_span > 0 else 0.0
            p_count = sum(NODE_PARAMS[n] for n in active)

            subset_records.append({
                "subset_id": "+".join(active) if active else "EMPTY_CIRCUIT",
                "size": len(active),
                "nodes": active,
                "parameter_count": p_count,
                "validation_accuracy": round(val_acc, 2),
                "heldout_accuracy": round(held_acc, 2),
                "causal_recovery_pp": round(causal_rec, 2),
                "percent_capability_recovered": round(pct_rec, 2)
            })

    # Sort subsets by size then by validation accuracy
    subset_records.sort(key=lambda x: (x["size"], -x["validation_accuracy"]))

    # Identify minimal candidate circuit that recovers >= 50% capability
    min_cand_50 = next((s for s in subset_records if s["percent_capability_recovered"] >= 50.0), subset_records[-1])
    # Identify minimal candidate circuit that recovers >= 80% capability
    min_cand_80 = next((s for s in subset_records if s["percent_capability_recovered"] >= 80.0), subset_records[-1])

    return {
        "model_name": model_name,
        "domain": domain,
        "intact_val_accuracy": round(acc_intact_val, 2),
        "intact_heldout_accuracy": round(acc_intact_held, 2),
        "null_val_accuracy": round(acc_null_val, 2),
        "null_heldout_accuracy": round(acc_null_held, 2),
        "total_recovery_span_pp": round(total_recovery_span, 2),
        "total_subsets_evaluated": len(subset_records),
        "subsets": subset_records,
        "minimal_nucleus_50pct": min_cand_50,
        "minimal_subcircuit_80pct": min_cand_80,
        "full_circuit_100pct": subset_records[-1]
    }


if __name__ == "__main__":
    res = run_exhaustive_subset_search()
    print("Exhaustive Subset Search Complete. Total subsets:", res["total_subsets_evaluated"])
    print("Minimal 50% Nucleus:", res["minimal_nucleus_50pct"]["subset_id"])
    print("Minimal 80% Subcircuit:", res["minimal_subcircuit_80pct"]["subset_id"])
