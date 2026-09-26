"""transfer/architecture_translator.py

Defines cross-architecture functional mappings and slot allocations
between Donor Family A (Micro-GPT) and Recipient Family B (LLaMA-Style).
"""
from __future__ import annotations

from typing import Dict, Any, List
from circuits.circuit_graph import FDCGraph


class ArchitectureTranslator:
    def __init__(self, donor_spec: Any, recipient_spec: Any):
        self.donor_spec = donor_spec
        self.recipient_spec = recipient_spec

        # Slot allocations in recipient model
        self.slot_mapping = {
            "L0_head_2": {"layer": 0, "head": 0, "type": "attention_slot", "role": "math_specialist"},
            "L0_mlp":    {"layer": 0, "slot": "mlp_aux", "type": "mlp_branch", "role": "math_mlp_dependency"},
            "L1_head_1": {"layer": 1, "head": 0, "type": "attention_slot", "role": "routing_dependency"},
            "L2_head_3": {"layer": 2, "head": 0, "type": "attention_slot", "role": "execution_dependency"},
        }

    def get_slot_info(self, node_id: str) -> Dict[str, Any]:
        return self.slot_mapping.get(node_id, {})

    def total_bridge_parameters(self) -> Dict[str, int]:
        d_src = self.donor_spec.hidden
        d_tgt = self.recipient_spec.hidden
        h_dim_src = self.donor_spec.hidden // self.donor_spec.heads
        h_dim_tgt = self.recipient_spec.hidden // self.recipient_spec.heads

        head_bridge = (h_dim_tgt * h_dim_src) + (d_src * d_tgt)  # 16*16 + 64*96 = 256 + 6144 = 6400
        mlp_bridge = (d_tgt * d_src) + (d_src * d_tgt)           # 96*64 + 64*96 = 6144 + 6144 = 12288

        return {
            "head_slot_params": head_bridge,
            "mlp_slot_params": mlp_bridge,
            "minimal_fdc_bridge_params": 3 * head_bridge + mlp_bridge, # 3*6400 + 12288 = 31488
            "recipient_total_params": 479069,
            "bridge_capacity_pct": round((3 * head_bridge + mlp_bridge) / 479069 * 100, 2)
        }
