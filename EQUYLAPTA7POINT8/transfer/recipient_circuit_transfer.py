"""transfer/recipient_circuit_transfer.py

Condition C: Recipient-Reconstructed Circuit.
Transfers the functional nucleus into the recipient while adapting the recipient-native
pathway (Layer 0 MLP and residual stream) discovered through empirical recipient discovery.
"""
from __future__ import annotations

from typing import Dict, Any, List
from transfer.architecture_translator import ArchitectureTranslator


def setup_recipient_reconstructed_transfer(donor_model: Any, recipient_model: Any, seed: int = 42) -> ArchitectureTranslator:
    return ArchitectureTranslator(
        donor_model=donor_model,
        recipient_model=recipient_model,
        active_elements=["L0_head_2", "L0_mlp"],
        condition_name="Condition_C_Recipient_Reconstructed_Circuit",
        seed=seed
    )
