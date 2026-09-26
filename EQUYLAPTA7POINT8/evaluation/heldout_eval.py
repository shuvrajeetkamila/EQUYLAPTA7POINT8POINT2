"""evaluation/heldout_eval.py

Evaluates condition accuracy, causal drop, and exact token agreement on strictly held-out test set (seed=999).
Enforces predeclared 90.0% threshold without shifting criteria.
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


def evaluate_heldout_suite(translator: ArchitectureTranslator, donor_model: Any,
                           heldout_seed: int = 999,
                           threshold_pct: float = 90.0) -> Dict[str, Any]:
    recipient = translator.recipient_model
    suite = micro_suite("math", n=20, seed=heldout_seed)

    # 1. Active evaluation
    translator.apply_to_recipient()
    res_active = eval_suite(recipient, suite, max_items=20)
    acc_active = float(res_active["accuracy"]) * 100.0

    # 2. Ablated evaluation
    translator.ablate_circuit()
    res_ablated = eval_suite(recipient, suite, max_items=20)
    acc_ablated = float(res_ablated["accuracy"]) * 100.0

    # 3. Restored evaluation
    translator.restore_circuit()
    res_restored = eval_suite(recipient, suite, max_items=20)
    acc_restored = float(res_restored["accuracy"]) * 100.0

    # 4. Exact Token / Decision Agreement with Donor on Held-Out Split
    donor_agreements = 0
    for it in suite.items:
        pred_donor = score_item(donor_model, it.prompt, it.options)
        pred_recipient = score_item(recipient, it.prompt, it.options)
        if pred_donor == pred_recipient:
            donor_agreements += 1

    agreement_pct = (donor_agreements / len(suite.items)) * 100.0

    # 5. Causal drop
    causal_drop = acc_active - acc_ablated
    rest_err = abs(acc_active - acc_restored)

    threshold_met = (agreement_pct >= threshold_pct)

    return {
        "condition": translator.condition_name,
        "n_items": len(suite.items),
        "heldout_seed": heldout_seed,
        "active_accuracy": round(acc_active, 2),
        "ablated_accuracy": round(acc_ablated, 2),
        "causal_drop_pp": round(causal_drop, 2),
        "restored_accuracy": round(acc_restored, 2),
        "restoration_error_pp": round(rest_err, 2),
        "exact_donor_agreement_pct": round(agreement_pct, 2),
        "predeclared_threshold_pct": threshold_pct,
        "predeclared_threshold_met": threshold_met,
        "classification": "FUNCTIONAL_CAPABILITY_TRANSFER" if threshold_met and causal_drop > 0 else "REPRESENTATIONAL_ALIGNMENT_ONLY"
    }
