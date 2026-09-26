"""causal/ablation.py

Surgical ablation battery for recipient model across circuit nodes.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from benchmarking.harness import eval_suite
from transfer.architecture_translator import ArchitectureTranslator


def evaluate_recipient_ablations(translator: ArchitectureTranslator, suite: Any) -> Dict[str, Any]:
    recipient = translator.recipient_model

    # 1. Active intact performance
    translator.apply_to_recipient()
    res_active = eval_suite(recipient, suite, max_items=20)
    acc_active = float(res_active["accuracy"]) * 100.0

    # 2. Entire circuit ablation
    translator.ablate_circuit()
    res_full_abl = eval_suite(recipient, suite, max_items=20)
    acc_full_abl = float(res_full_abl["accuracy"]) * 100.0
    translator.restore_circuit()

    # 3. Node-by-node ablations
    node_drops = {}
    # L0_head_0 (slot for L0_head_2)
    recipient.head_mask = np.ones((recipient.spec.layers, recipient.spec.heads), dtype=np.float32)
    recipient.head_mask[0, 0] = 0.0
    r_h0 = eval_suite(recipient, suite, max_items=20)
    node_drops["slot_L0_head_0"] = round(acc_active - float(r_h0["accuracy"]) * 100.0, 2)
    recipient.head_mask = None

    # L0_mlp
    recipient.skip_modules = {0: ["mlp"]}
    r_mlp0 = eval_suite(recipient, suite, max_items=20)
    node_drops["slot_L0_mlp"] = round(acc_active - float(r_mlp0["accuracy"]) * 100.0, 2)
    recipient.skip_modules = {}

    # Downstream Layer 1 head 0
    recipient.head_mask = np.ones((recipient.spec.layers, recipient.spec.heads), dtype=np.float32)
    recipient.head_mask[1, 0] = 0.0
    r_h1 = eval_suite(recipient, suite, max_items=20)
    node_drops["slot_L1_head_0"] = round(acc_active - float(r_h1["accuracy"]) * 100.0, 2)
    recipient.head_mask = None

    # Downstream Layer 2 head 0
    recipient.head_mask = np.ones((recipient.spec.layers, recipient.spec.heads), dtype=np.float32)
    recipient.head_mask[2, 0] = 0.0
    r_h2 = eval_suite(recipient, suite, max_items=20)
    node_drops["slot_L2_head_0"] = round(acc_active - float(r_h2["accuracy"]) * 100.0, 2)
    recipient.head_mask = None

    return {
        "active_accuracy": round(acc_active, 2),
        "ablated_accuracy": round(acc_full_abl, 2),
        "circuit_causal_drop_pp": round(acc_active - acc_full_abl, 2),
        "node_drops": node_drops
    }
