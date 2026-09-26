"""causal/ablation.py

Implements full-circuit and node-specific surgical ablation routines for recipient models.
"""
from __future__ import annotations

from typing import Dict, Any, List
from benchmarking.harness import eval_suite
from transfer.dependency_transfer import FDCTransferSystem


def run_recipient_ablation_battery(transfer_system: FDCTransferSystem, suite: Any) -> Dict[str, Any]:
    """Runs intact vs ablated evaluations and node-specific ablations in the recipient."""
    model = transfer_system.recipient_model

    # 1. Intact recipient
    transfer_system.restore_all_fdc()
    acc_intact = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0

    # 2. Complete FDC ablated
    transfer_system.ablate_all_fdc()
    acc_ablated = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0
    transfer_system.restore_all_fdc()

    full_circuit_drop = acc_intact - acc_ablated

    # 3. Node-specific ablations
    node_drops = {}
    for node_id in transfer_system.active_elements:
        transfer_system.restore_all_fdc()
        transfer_system.ablate_node(node_id)
        acc_node_abl = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0
        drop = acc_intact - acc_node_abl
        node_drops[node_id] = {
            "active_accuracy": round(acc_intact, 2),
            "ablated_accuracy": round(acc_node_abl, 2),
            "causal_drop_pp": round(drop, 2)
        }
    transfer_system.restore_all_fdc()

    return {
        "condition": transfer_system.condition_name,
        "active_accuracy": round(acc_intact, 2),
        "full_circuit_ablated_accuracy": round(acc_ablated, 2),
        "full_circuit_causal_drop_pp": round(full_circuit_drop, 2),
        "node_ablations": node_drops,
        "causal_necessity_demonstrated": bool(full_circuit_drop >= 10.0)
    }
