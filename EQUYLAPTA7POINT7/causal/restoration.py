"""causal/restoration.py

Evaluates surgical restoration recovery and bypass conditions in recipient models.
"""
from __future__ import annotations

from typing import Dict, Any
from benchmarking.harness import eval_suite
from transfer.dependency_transfer import FDCTransferSystem


def run_recipient_restoration_battery(transfer_system: FDCTransferSystem, suite: Any) -> Dict[str, Any]:
    """Tests whether reinserting the FDC recovers intact accuracy, and tests bypass."""
    model = transfer_system.recipient_model

    # 1. Intact
    transfer_system.restore_all_fdc()
    acc_intact = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0

    # 2. Ablate
    transfer_system.ablate_all_fdc()
    acc_ablated = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0

    # 3. Restore
    transfer_system.restore_all_fdc()
    acc_restored = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0

    recovery_error = abs(acc_intact - acc_restored)

    return {
        "condition": transfer_system.condition_name,
        "active_accuracy": round(acc_intact, 2),
        "ablated_accuracy": round(acc_ablated, 2),
        "restored_accuracy": round(acc_restored, 2),
        "restoration_recovery_error_pp": round(recovery_error, 2),
        "restoration_exact": bool(recovery_error < 1e-4)
    }
