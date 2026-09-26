"""src/evaluation.py

Comprehensive evaluation suite implementing:
- 8 Success Criteria evaluation
- 90% functional-equivalence threshold check
- 5-seed statistical aggregation (mean, std, median, min, max, 95% CI)
- Multi-scale depth sweep (2L, 4L, 6L, 8L)
- Circuit-size sweep (1, 2, 4, 8 nodes)
- Mathematical surgicality scoring
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import load_model, MicroTransformer
from benchmarking.suites import micro_suite
from benchmarking.harness import eval_suite, score_item
from src.transfer import ArchitectureTranslator


def compute_statistics(values: List[float]) -> Dict[str, Any]:
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    se = std / np.sqrt(n) if n > 0 else 0.0
    ci95 = (float(mean - 2.776 * se), float(mean + 2.776 * se))
    pos_count = int(np.sum(arr > 0))
    neg_count = int(np.sum(arr < 0))
    zero_count = int(np.sum(arr == 0))
    
    if neg_count == 0 and pos_count > 0 and mean >= 5.0:
        status = "UNIFORM_POSITIVE"
    elif neg_count > 0:
        status = "MIXED_SIGN_VARIABILITY"
    else:
        status = "NON_REPLICATED"

    return {
        "seed_values": [round(float(x), 2) for x in values],
        "mean": round(mean, 2),
        "std": round(std, 2),
        "median": round(median, 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "ci_95_lower": round(ci95[0], 2),
        "ci_95_upper": round(ci95[1], 2),
        "ci95": [round(ci95[0], 2), round(ci95[1], 2)],
        "positive_seed_count": pos_count,
        "negative_seed_count": neg_count,
        "zero_seed_count": zero_count,
        "sign_consistency_status": status,
        "raw": [round(float(x), 2) for x in values]
    }


def evaluate_heldout_condition(translator: ArchitectureTranslator, donor_model: MicroTransformer,
                               heldout_seed: int = 999,
                               threshold_pct: float = 90.0) -> Dict[str, Any]:
    recipient = translator.recipient_model
    suite = micro_suite("math", n=20, seed=heldout_seed)

    # 1. Active evaluation
    translator.apply_to_recipient()
    res_act = eval_suite(recipient, suite, max_items=20)
    acc_act = float(res_act["accuracy"]) * 100.0

    # 2. Ablated evaluation
    translator.ablate_circuit()
    res_abl = eval_suite(recipient, suite, max_items=20)
    acc_abl = float(res_abl["accuracy"]) * 100.0

    # 3. Restored evaluation
    translator.restore_circuit()
    res_rest = eval_suite(recipient, suite, max_items=20)
    acc_rest = float(res_rest["accuracy"]) * 100.0

    # 4. Donor decision agreement
    agreements = 0
    for it in suite.items:
        p_src = score_item(donor_model, it.prompt, it.options)
        p_tgt = score_item(recipient, it.prompt, it.options)
        if p_src == p_tgt:
            agreements += 1
    agreement_pct = (agreements / len(suite.items)) * 100.0

    causal_drop = acc_act - acc_abl
    rest_err = abs(acc_act - acc_rest)
    threshold_met = agreement_pct >= threshold_pct

    return {
        "condition": translator.condition_name,
        "active_accuracy": round(acc_act, 2),
        "ablated_accuracy": round(acc_abl, 2),
        "causal_drop_pp": round(causal_drop, 2),
        "restored_accuracy": round(acc_rest, 2),
        "restoration_error_pp": round(rest_err, 2),
        "donor_agreement_pct": round(agreement_pct, 2),
        "predeclared_threshold_pct": threshold_pct,
        "threshold_met": threshold_met
    }


def compute_surgicality_score(target_gain_pp: float,
                              collateral_change_pp: float,
                              causal_specificity: float,
                              transferred_params: int,
                              total_recipient_params: int = 479069) -> float:
    """Computes mathematical surgicality score:
    S = (target_gain + causal_specificity) / (1 + collateral_change + normalized_parameter_footprint)
    """
    param_footprint = transferred_params / total_recipient_params
    numerator = max(0.0, target_gain_pp) + causal_specificity
    denominator = 1.0 + max(0.0, collateral_change_pp / 100.0) + param_footprint
    return round(float(numerator / denominator), 4)


def run_depth_sweep(depths: List[str] = ["2L", "4L", "6L", "8L"], seed: int = 42) -> Dict[str, Any]:
    suite = micro_suite("math", n=20, seed=888)
    results = {}
    for d in depths:
        m_src = load_model(f"e4-math-{d}")
        m_tgt = load_model(f"e6-base-{d}")
        acc_base = float(eval_suite(m_tgt, suite, max_items=20)["accuracy"]) * 100.0
        acc_donor = float(eval_suite(m_src, suite, max_items=20)["accuracy"]) * 100.0

        t = ArchitectureTranslator(m_src, m_tgt, ["L0_head_2", "L0_mlp"], f"Depth_{d}", seed=seed)
        t.apply_to_recipient()
        acc_act = float(eval_suite(m_tgt, suite, max_items=20)["accuracy"]) * 100.0
        t.ablate_circuit()
        acc_abl = float(eval_suite(m_tgt, suite, max_items=20)["accuracy"]) * 100.0
        t.restore_circuit()

        results[d] = {
            "donor_model": f"e4-math-{d}",
            "recipient_model": f"e6-base-{d}",
            "donor_accuracy": round(acc_donor, 2),
            "recipient_baseline": round(acc_base, 2),
            "recipient_active": round(acc_act, 2),
            "causal_drop_pp": round(acc_act - acc_abl, 2)
        }
    return results


def run_circuit_size_sweep(seed: int = 42) -> List[Dict[str, Any]]:
    m_src = load_model("e4-math-4L")
    m_tgt = load_model("e6-base-4L")
    suite_val = micro_suite("math", n=20, seed=888)
    suite_held = micro_suite("math", n=20, seed=999)

    cfgs = [
        {"size": 1, "name": "1_node", "nodes": ["L0_head_2"], "params": 1024},
        {"size": 2, "name": "2_node_minimal", "nodes": ["L0_head_2", "L0_mlp"], "params": 33792},
        {"size": 3, "name": "3_node_subcircuit", "nodes": ["L0_head_2", "L0_mlp", "L2_head_3"], "params": 34816},
        {"size": 4, "name": "4_node_exact_fdc", "nodes": ["L0_head_2", "L0_mlp", "L1_head_1", "L2_head_3"], "params": 35840},
        {"size": 8, "name": "8_node_oversized", "nodes": ["L0_head_2", "L0_mlp", "L1_head_1", "L2_head_3", "L0_head_3", "L1_mlp", "L2_mlp", "L3_mlp"], "params": 135168}
    ]

    curve = []
    for c in cfgs:
        t = ArchitectureTranslator(m_src, m_tgt, c["nodes"], c["name"], seed=seed)
        t.apply_to_recipient()
        val_act = float(eval_suite(m_tgt, suite_val, max_items=20)["accuracy"]) * 100.0
        held_act = float(eval_suite(m_tgt, suite_held, max_items=20)["accuracy"]) * 100.0
        t.ablate_circuit()
        val_abl = float(eval_suite(m_tgt, suite_val, max_items=20)["accuracy"]) * 100.0
        held_abl = float(eval_suite(m_tgt, suite_held, max_items=20)["accuracy"]) * 100.0
        t.restore_circuit()

        curve.append({
            "size": c["size"],
            "name": c["name"],
            "parameter_count": c["params"],
            "val_active_acc": round(val_act, 2),
            "val_causal_drop_pp": round(val_act - val_abl, 2),
            "heldout_active_acc": round(held_act, 2),
            "heldout_causal_drop_pp": round(held_act - held_abl, 2)
        })
    return curve
