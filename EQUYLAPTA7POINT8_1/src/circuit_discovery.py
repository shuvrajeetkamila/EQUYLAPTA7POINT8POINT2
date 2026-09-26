"""src/circuit_discovery.py

Implements native causal localization, exhaustive 16-subset search (distinguishing
validation-set optimum from held-out optimum), and independent recipient causal discovery.
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


def evaluate_donor_subset(model: MicroTransformer, active_nodes: List[str], suite: Any) -> float:
    """Evaluates donor accuracy when ONLY active_nodes are enabled among candidate nodes."""
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

    # Null circuit (all 4 ablated)
    acc_null_val = evaluate_donor_subset(model, [], suite_val)
    acc_null_held = evaluate_donor_subset(model, [], suite_held)

    # Intact baseline
    model.head_mask = None
    model.skip_modules = {}
    acc_intact_val = float(eval_suite(model, suite_val, max_items=20)["accuracy"]) * 100.0
    acc_intact_held = float(eval_suite(model, suite_held, max_items=20)["accuracy"]) * 100.0

    subset_records = []
    total_val_span = acc_intact_val - acc_null_val
    total_held_span = acc_intact_held - acc_null_held

    for k in range(len(NODE_NAMES) + 1):
        for combo in itertools.combinations(NODE_NAMES, k):
            active = list(combo)
            val_acc = evaluate_donor_subset(model, active, suite_val)
            held_acc = evaluate_donor_subset(model, active, suite_held)

            causal_rec_val = val_acc - acc_null_val
            pct_rec_val = (causal_rec_val / total_val_span * 100.0) if total_val_span > 0 else 0.0

            causal_rec_held = held_acc - acc_null_held
            pct_rec_held = (causal_rec_held / total_held_span * 100.0) if total_held_span > 0 else 0.0

            p_count = sum(NODE_PARAMS[n] for n in active)

            subset_records.append({
                "subset_id": "+".join(active) if active else "EMPTY_CIRCUIT",
                "size": len(active),
                "nodes": active,
                "parameter_count": p_count,
                "validation_accuracy": round(val_acc, 2),
                "heldout_accuracy": round(held_acc, 2),
                "val_causal_recovery_pp": round(causal_rec_val, 2),
                "heldout_causal_recovery_pp": round(causal_rec_held, 2),
                "val_percent_recovered": round(pct_rec_val, 2),
                "heldout_percent_recovered": round(pct_rec_held, 2)
            })

    subset_records.sort(key=lambda x: (x["size"], -x["validation_accuracy"]))

    # Distinguish validation-set optimum from held-out optimum (Section 23 & 24)
    val_optimum_2node = max([s for s in subset_records if s["size"] == 2], key=lambda x: x["validation_accuracy"])
    held_optimum_2node = max([s for s in subset_records if s["size"] == 2], key=lambda x: x["heldout_accuracy"])

    return {
        "model_name": model_name,
        "total_subsets": len(subset_records),
        "intact_val_accuracy": round(acc_intact_val, 2),
        "intact_heldout_accuracy": round(acc_intact_held, 2),
        "null_val_accuracy": round(acc_null_val, 2),
        "null_heldout_accuracy": round(acc_null_held, 2),
        "subsets": subset_records,
        "val_optimum_2node": val_optimum_2node,
        "heldout_optimum_2node": held_optimum_2node,
        "minimal_nucleus": val_optimum_2node,
        "full_circuit": subset_records[-1]
    }


def discover_recipient_pathway(recipient_model: MicroTransformer, suite: Any) -> Dict[str, Any]:
    """Independently probes recipient architecture to find which native modules carry transferred signal."""
    res_intact = eval_suite(recipient_model, suite, max_items=20)
    acc_active = float(res_intact["accuracy"]) * 100.0

    # MLP drops
    mlp_drops = {}
    for l in range(recipient_model.spec.layers):
        recipient_model.head_mask = None
        recipient_model.skip_modules = {l: ["mlp"]}
        acc_m = float(eval_suite(recipient_model, suite, max_items=20)["accuracy"]) * 100.0
        mlp_drops[f"Recipient_L{l}_mlp"] = round(acc_active - acc_m, 2)
    recipient_model.skip_modules = {}

    # Head drops
    head_drops = {}
    for l in range(1, recipient_model.spec.layers):
        for h in range(recipient_model.spec.heads):
            recipient_model.head_mask = np.ones((recipient_model.spec.layers, recipient_model.spec.heads), dtype=np.float32)
            recipient_model.head_mask[l, h] = 0.0
            acc_h = float(eval_suite(recipient_model, suite, max_items=20)["accuracy"]) * 100.0
            drop = round(acc_active - acc_h, 2)
            if drop > 0:
                head_drops[f"Recipient_L{l}_head_{h}"] = drop
    recipient_model.head_mask = None

    primary_mediator = max(mlp_drops.items(), key=lambda x: x[1])

    return {
        "recipient_active_accuracy": round(acc_active, 2),
        "recipient_mlp_drops": mlp_drops,
        "recipient_head_drops": head_drops,
        "primary_recipient_mediator": primary_mediator[0],
        "mediator_drop_pp": primary_mediator[1]
    }
