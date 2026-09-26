"""transfer/fdc_transfer.py

Condition B & E: Functional Dependency Circuit (FDC) Transfer.
Condition D: Minimal Candidate Circuit.
Condition F: Oversized Circuit (8 nodes).
"""
from __future__ import annotations

from typing import Dict, Any, List
from transfer.architecture_translator import ArchitectureTranslator


def setup_fdc_transfer(donor_model: Any, recipient_model: Any,
                       condition_name: str = "Condition_B_Donor_FDC_7_7",
                       circuit_type: str = "exact_fdc", seed: int = 42) -> ArchitectureTranslator:
    if circuit_type == "minimal_candidate":
        active_elements = ["L0_head_2", "L0_mlp"]
    elif circuit_type == "exact_fdc":
        active_elements = ["L0_head_2", "L0_mlp", "L1_head_1", "L2_head_3"]
    elif circuit_type == "oversized":
        active_elements = [
            "L0_head_2", "L0_mlp", "L1_head_1", "L2_head_3",
            "L0_head_3", "L1_mlp", "L2_mlp", "L3_mlp"
        ]
    else:
        active_elements = ["L0_head_2", "L0_mlp", "L1_head_1", "L2_head_3"]

    return ArchitectureTranslator(
        donor_model=donor_model,
        recipient_model=recipient_model,
        active_elements=active_elements,
        condition_name=condition_name,
        seed=seed
    )
