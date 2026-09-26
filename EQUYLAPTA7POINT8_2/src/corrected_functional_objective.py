"""src/corrected_functional_objective.py

EQUYLAPTA 7.8.2: True differentiable multi-branch functional causal objective:
    L_functional = 0.5 * mean( ((Y_intact(x, theta) - Y_ablated(x, theta)) - Delta_target(x))^2 )
and exact analytical backpropagation gradients across head and MLP adapters.
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
                            delta_target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the scalar functional loss and the error residual vector."""
    # 1. Intact forward pass
    recipient_model.head_mask = None
    l_int, _ = recipient_model.forward(ids)

    # 2. Ablated forward pass
    recipient_model.head_mask = np.ones((recipient_model.spec.layers, recipient_model.spec.heads), dtype=np.float32)
    recipient_model.head_mask[0, 0] = 0.0
    l_abl, _ = recipient_model.forward(ids)
    recipient_model.head_mask = None

    delta_recip = (l_int[0, -1, :] - l_abl[0, -1, :]).astype(np.float32)
    diff = delta_recip - delta_target
    loss = 0.5 * float(np.mean(diff ** 2))
    return loss, diff, l_int, l_abl


def compute_multi_branch_gradients(recipient_model: MicroTransformer, ids: np.ndarray,
                                   diff: np.ndarray,
                                   l_int: np.ndarray,
                                   l_abl: np.ndarray,
                                   head_slot: int = 0,
                                   mlp_d_sub: int = 256,
                                   alpha_mlp: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executes exact analytical two-branch backward pass for both head and MLP recipient parameters."""
    V = recipient_model.spec.vocab
    spec = recipient_model.spec
    h_dim = spec.hidden // spec.heads

    # Branch 1: Intact branch (+diff / V)
    recipient_model.head_mask = None
    dlogits_int = np.zeros_like(l_int)
    dlogits_int[0, -1, :] = (diff / V).astype(np.float32)
    g_int = recipient_model.backward(ids, dlogits_int)

    # Branch 2: Ablated branch (-diff / V)
    recipient_model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
    recipient_model.head_mask[0, head_slot] = 0.0
    dlogits_abl = np.zeros_like(l_abl)
    dlogits_abl[0, -1, :] = (-diff / V).astype(np.float32)
    g_abl = recipient_model.backward(ids, dlogits_abl)
    recipient_model.head_mask = None

    # Head effective gradient
    g_eff_head = (g_int["L0.attn.o.W"][head_slot * h_dim : (head_slot + 1) * h_dim, :] +
                  g_abl["L0.attn.o.W"][head_slot * h_dim : (head_slot + 1) * h_dim, :])

    # MLP effective gradients
    g_eff_mlp1 = alpha_mlp * (g_int["L0.mlp.1.W"][:, :mlp_d_sub] + g_abl["L0.mlp.1.W"][:, :mlp_d_sub])
    g_eff_mlp2 = alpha_mlp * (g_int["L0.mlp.2.W"][:mlp_d_sub, :] + g_abl["L0.mlp.2.W"][:mlp_d_sub, :])

    return g_eff_head, g_eff_mlp1, g_eff_mlp2


def compute_head_adapter_gradients(g_eff_head: np.ndarray, head_bridge: Any,
                                   lambda_reg: float = 0.0001) -> Tuple[np.ndarray, np.ndarray]:
    """Computes exact analytical gradients for HeadSlotBridge adapters."""
    dW_in = g_eff_head @ (head_bridge.W_comp @ head_bridge.W_out).T + lambda_reg * (head_bridge.W_in - head_bridge.W_in_0)
    dW_out = (head_bridge.W_in @ head_bridge.W_comp).T @ g_eff_head + lambda_reg * (head_bridge.W_out - head_bridge.W_out_0)
    return dW_in, dW_out


def compute_mlp_adapter_gradients(g_eff_mlp1: np.ndarray, g_eff_mlp2: np.ndarray,
                                  mlp_bridge: Any, lambda_reg: float = 0.0001) -> Tuple[np.ndarray, np.ndarray]:
    """Computes exact analytical gradients for MLPSlotBridge adapters."""
    dW_in = g_eff_mlp1 @ mlp_bridge.W_mlp1.T + lambda_reg * (mlp_bridge.W_in - mlp_bridge.W_in_0)
    dW_out = mlp_bridge.W_mlp2.T @ g_eff_mlp2 + lambda_reg * (mlp_bridge.W_out - mlp_bridge.W_out_0)
    return dW_in, dW_out
