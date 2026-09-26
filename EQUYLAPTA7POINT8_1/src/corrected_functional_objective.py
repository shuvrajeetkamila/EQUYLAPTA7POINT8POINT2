"""src/corrected_functional_objective.py

Implements the true differentiable two-branch functional causal objective:
    L_functional = 0.5 * mean( ((Y_intact(x, theta) - Y_ablated(x, theta)) - Delta_target(x))^2 )
and its exact analytical backpropagation gradients across bidirectional bridge parameters.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import MicroTransformer


def compute_donor_causal_signature(donor_model: MicroTransformer, ids: np.ndarray,
                                   source_head: Tuple[int, int] = (0, 2)) -> np.ndarray:
    """Computes Delta_target(x) = Y_donor_intact(x) - Y_donor_ablated(x) for prompt ids."""
    donor_model.head_mask = None
    donor_model.skip_modules = {}
    l_int, _ = donor_model.forward(ids)

    donor_model.head_mask = np.ones((donor_model.spec.layers, donor_model.spec.heads), dtype=np.float32)
    donor_model.head_mask[source_head[0], source_head[1]] = 0.0
    l_abl, _ = donor_model.forward(ids)

    donor_model.head_mask = None
    delta_target = (l_int[0, -1, :] - l_abl[0, -1, :]).astype(np.float32)
    return delta_target


def compute_functional_loss(recipient_model: MicroTransformer, ids: np.ndarray,
                            delta_target: np.ndarray,
                            ablate_slot_fn: Any,
                            restore_slot_fn: Any) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the scalar functional loss and the error residual vector."""
    # 1. Intact forward pass
    restore_slot_fn()
    l_int, _ = recipient_model.forward(ids)

    # 2. Ablated forward pass
    ablate_slot_fn()
    l_abl, _ = recipient_model.forward(ids)
    restore_slot_fn()

    V = recipient_model.spec.vocab
    delta_recip = (l_int[0, -1, :] - l_abl[0, -1, :]).astype(np.float32)
    diff = delta_recip - delta_target
    loss = 0.5 * float(np.mean(diff ** 2))
    return loss, diff, l_int, l_abl


def compute_two_branch_gradients(recipient_model: MicroTransformer, ids: np.ndarray,
                                 diff: np.ndarray,
                                 l_int: np.ndarray,
                                 l_abl: np.ndarray,
                                 ablate_slot_fn: Any,
                                 restore_slot_fn: Any,
                                 head_slot: int = 0) -> np.ndarray:
    """Executes the exact two-branch backward pass through the recipient model."""
    V = recipient_model.spec.vocab
    spec = recipient_model.spec
    h_dim = spec.hidden // spec.heads

    # Branch 1: Intact branch (+diff / V)
    restore_slot_fn()
    recipient_model.head_mask = None
    dlogits_int = np.zeros_like(l_int)
    dlogits_int[0, -1, :] = (diff / V).astype(np.float32)
    g_int = recipient_model.backward(ids, dlogits_int)

    # Branch 2: Ablated branch (-diff / V)
    ablate_slot_fn()
    recipient_model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
    recipient_model.head_mask[0, head_slot] = 0.0
    dlogits_abl = np.zeros_like(l_abl)
    dlogits_abl[0, -1, :] = (-diff / V).astype(np.float32)
    g_abl = recipient_model.backward(ids, dlogits_abl)

    # Restore clean state
    restore_slot_fn()
    recipient_model.head_mask = None

    # Total effective slot gradient
    g_eff = (g_int["L0.attn.o.W"][head_slot * h_dim : (head_slot + 1) * h_dim, :] +
             g_abl["L0.attn.o.W"][head_slot * h_dim : (head_slot + 1) * h_dim, :])
    return g_eff
