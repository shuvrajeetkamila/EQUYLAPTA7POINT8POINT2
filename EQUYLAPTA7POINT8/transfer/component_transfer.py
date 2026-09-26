"""transfer/component_transfer.py

Condition A: Component-Only (7.6 style, single specialist head at Layer 0).
"""
from __future__ import annotations

from typing import Dict, Any, List
from transfer.architecture_translator import ArchitectureTranslator


def setup_component_only_transfer(donor_model: Any, recipient_model: Any, seed: int = 42) -> ArchitectureTranslator:
    return ArchitectureTranslator(
        donor_model=donor_model,
        recipient_model=recipient_model,
        active_elements=["L0_head_2"],
        condition_name="Condition_A_Component_Only_7_6",
        seed=seed
    )
