"""tests/test_objective.py

Verifies mathematical properties of the two-branch functional causal objective:
1. Exact scalar non-negativity
2. Loss is 0 when recipient delta matches target delta
3. Backward pass returns non-zero gradient when discrepancies exist
4. Shape correctness across multi-branch backpropagation
"""
import os
import sys
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.synthetic import load_model, gen_suite
from src.corrected_functional_objective import (
    compute_donor_causal_signature,
    compute_functional_loss,
    compute_multi_branch_gradients
)


def test_functional_objective_properties():
    donor = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e4-math-4L"))
    recip = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e6-base-4L"))

    item = gen_suite("math", n=1, seed=888)[0]
    p_ids = recip.tokenizer.encode(item.prompt)
    ids = np.array([p_ids], dtype=np.int64)

    delta_target = compute_donor_causal_signature(donor, ids, source_head=(0, 2))

    # Test 1: Non-negativity
    loss, diff, l_int, l_abl = compute_functional_loss(recip, ids, delta_target)
    assert loss >= 0.0, "Loss must be non-negative!"

    # Test 2: Zero loss on matched delta
    delta_recip = (l_int[0, -1, :] - l_abl[0, -1, :]).astype(np.float32)
    loss_zero, diff_zero, _, _ = compute_functional_loss(recip, ids, delta_recip)
    assert abs(loss_zero) < 1e-12, "Loss must be zero when recipient delta matches target delta!"

    # Test 3: Gradient non-zero when discrepancies exist
    g_head, g_mlp1, g_mlp2 = compute_multi_branch_gradients(recip, ids, diff, l_int, l_abl)
    assert np.linalg.norm(g_head) > 1e-6, "Head effective gradient should be non-zero!"
    assert np.linalg.norm(g_mlp1) > 1e-6, "MLP1 effective gradient should be non-zero!"
    assert np.linalg.norm(g_mlp2) > 1e-6, "MLP2 effective gradient should be non-zero!"

    # Test 4: Shape correctness
    h_dim = recip.spec.hidden // recip.spec.heads
    assert g_head.shape == (h_dim, recip.spec.hidden)
    assert g_mlp1.shape == (recip.spec.hidden, 256)
    assert g_mlp2.shape == (256, recip.spec.hidden)


if __name__ == "__main__":
    test_functional_objective_properties()
    print("test_objective passed successfully!")
