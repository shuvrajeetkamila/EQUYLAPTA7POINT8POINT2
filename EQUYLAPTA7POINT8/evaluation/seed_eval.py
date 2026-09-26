"""evaluation/seed_eval.py

Runs 5-seed replication (9001, 9002, 9003, 9004, 9005) across all conditions.
Computes comprehensive statistical distributions (mean, std, 95% CI, min, max, raw).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from benchmarking.suites import micro_suite
from benchmarking.harness import eval_suite, score_item
from transfer.architecture_translator import ArchitectureTranslator


def compute_statistics(values: List[float]) -> Dict[str, Any]:
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    # 95% CI using t-multiplier ~2.776 for df=4
    ci95 = (float(mean - 2.776 * se), float(mean + 2.776 * se))
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "ci95": [round(ci95[0], 2), round(ci95[1], 2)],
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "raw": [round(x, 2) for x in values]
    }


def evaluate_condition_across_seeds(translator: ArchitectureTranslator, donor_model: Any,
                                    seeds: List[int] = [9001, 9002, 9003, 9004, 9005]) -> Dict[str, Any]:
    recipient = translator.recipient_model

    active_accs = []
    ablated_accs = []
    causal_drops = []
    agreements = []

    for s in seeds:
        suite = micro_suite("math", n=20, seed=s)

        # Active
        translator.apply_to_recipient()
        res_act = eval_suite(recipient, suite, max_items=20)
        acc_act = float(res_act["accuracy"]) * 100.0
        active_accs.append(acc_act)

        # Ablated
        translator.ablate_circuit()
        res_abl = eval_suite(recipient, suite, max_items=20)
        acc_abl = float(res_abl["accuracy"]) * 100.0
        ablated_accs.append(acc_abl)

        # Restore
        translator.restore_circuit()

        causal_drops.append(acc_act - acc_abl)

        # Donor agreement
        agree_count = 0
        for it in suite.items:
            if score_item(donor_model, it.prompt, it.options) == score_item(recipient, it.prompt, it.options):
                agree_count += 1
        agreements.append((agree_count / len(suite.items)) * 100.0)

    return {
        "condition": translator.condition_name,
        "seeds": seeds,
        "active_accuracy": compute_statistics(active_accs),
        "ablated_accuracy": compute_statistics(ablated_accs),
        "causal_drop_pp": compute_statistics(causal_drops),
        "donor_agreement_pct": compute_statistics(agreements)
    }
