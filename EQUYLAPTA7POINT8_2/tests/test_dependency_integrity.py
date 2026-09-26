"""tests/test_dependency_integrity.py

Verifies functional dependency circuit integrity, parameter counts, frozen weight invariants,
and surgical ablation isolation.
"""
import json
import os
import sys
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.synthetic import load_model
from src.transfer import ArchitectureTranslator


def test_dependency_integrity():
    donor = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e4-math-4L"))
    recip = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e6-base-4L"))

    translator = ArchitectureTranslator(
        donor, recip,
        active_elements=["L0_head_2", "L0_mlp"],
        condition_name="Condition_E_Recipient_Reconstructed_FDC",
        seed=42
    )

    # 1. Parameter counts
    head_b = translator.head_bridges["L0_head_2"]
    mlp_b = translator.mlp_bridges["L0_mlp"]

    assert head_b.W_in.size == 256, f"Expected 256 W_in params for head, got {head_b.W_in.size}"
    assert head_b.W_out.size == 6144, f"Expected 6144 W_out params for head, got {head_b.W_out.size}"
    assert mlp_b.W_in.size == 6144, f"Expected 6144 W_in params for mlp, got {mlp_b.W_in.size}"
    assert mlp_b.W_out.size == 6144, f"Expected 6144 W_out params for mlp, got {mlp_b.W_out.size}"

    total_trainable = head_b.param_count() + mlp_b.param_count()
    assert total_trainable == 18688, f"Expected 18688 trainable parameters, got {total_trainable}"

    # 2. Frozen invariant verification
    assert translator.verify_all_frozen() is True, "Donor frozen invariants failed!"

    # 3. Surgical ablation check
    translator.apply_to_recipient()
    # Check that slot has non-zero weights
    assert np.linalg.norm(recip.params["L0.attn.o.W"][:16, :]) > 0
    # Now ablate
    translator.ablate_circuit()
    assert np.linalg.norm(recip.params["L0.attn.o.W"][:16, :]) == 0.0, "Head slot not ablated to 0.0"


if __name__ == "__main__":
    test_dependency_integrity()
    print("test_dependency_integrity passed successfully!")
