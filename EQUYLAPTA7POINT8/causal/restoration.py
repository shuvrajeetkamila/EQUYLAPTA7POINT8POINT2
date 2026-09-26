"""causal/restoration.py

Validates circuit restoration on held-out inputs and computes recovery error.
"""
from __future__ import annotations

from typing import Dict, Any
from benchmarking.harness import eval_suite
from transfer.architecture_translator import ArchitectureTranslator


def evaluate_recipient_restoration(translator: ArchitectureTranslator, suite_held: Any) -> Dict[str, Any]:
    recipient = translator.recipient_model

    # Initial active
    translator.apply_to_recipient()
    acc_active = float(eval_suite(recipient, suite_held, max_items=20)["accuracy"]) * 100.0

    # Ablate
    translator.ablate_circuit()
    acc_ablated = float(eval_suite(recipient, suite_held, max_items=20)["accuracy"]) * 100.0

    # Restore
    translator.restore_circuit()
    acc_restored = float(eval_suite(recipient, suite_held, max_items=20)["accuracy"]) * 100.0

    rest_error = abs(acc_active - acc_restored)

    return {
        "heldout_active_accuracy": round(acc_active, 2),
        "heldout_ablated_accuracy": round(acc_ablated, 2),
        "heldout_causal_drop_pp": round(acc_active - acc_ablated, 2),
        "heldout_restored_accuracy": round(acc_restored, 2),
        "restoration_error_pp": round(rest_error, 2),
        "restoration_perfect": rest_error == 0.0
    }
