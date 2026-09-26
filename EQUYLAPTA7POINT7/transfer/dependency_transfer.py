"""transfer/dependency_transfer.py

Implements cross-architecture Functional Dependency Circuit (FDC) transfer,
two-branch functional-effect training using Adam optimizer, and surgical intervention handles.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from transfer.interface_adapter import HeadSlotBridge, MLPSlotBridge


class FDCTransferSystem:
    """Manages cross-architecture transfer of an FDC into a recipient model."""

    def __init__(self, donor_model: Any, recipient_model: Any,
                 active_elements: List[str], condition_name: str, seed: int = 42):
        self.donor_model = donor_model
        self.recipient_model = recipient_model
        self.active_elements = active_elements
        self.condition_name = condition_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.d_src = donor_model.spec.hidden
        self.d_tgt = recipient_model.spec.hidden
        self.h_dim_src = donor_model.spec.hidden // donor_model.spec.heads
        self.h_dim_tgt = recipient_model.spec.hidden // recipient_model.spec.heads

        # Store pristine backup of recipient weights to allow clean restorations
        self.recipient_backup = {k: v.copy() for k, v in recipient_model.params.items()}

        # Build bridges for active components
        self.head_bridges: Dict[str, HeadSlotBridge] = {}
        self.mlp_bridges: Dict[str, MLPSlotBridge] = {}

        self._init_bridges()
        self.apply_to_recipient()

    def _init_bridges(self):
        # 1. L0_head_2 (Core Specialist)
        if "L0_head_2" in self.active_elements:
            W_src = self.donor_model.params["L0.attn.o.W"][2*self.h_dim_src : 3*self.h_dim_src, :]
            if self.condition_name == "Condition_F_Random_Matched_Circuit":
                W_src = self.rng.standard_normal(W_src.shape).astype(np.float32) * 0.1
            self.head_bridges["L0_head_2"] = HeadSlotBridge(
                self.h_dim_src, self.d_src, self.h_dim_tgt, self.d_tgt, W_src, seed=self.seed
            )

        # 2. L0_mlp (Immediate Downstream Dependency)
        if "L0_mlp" in self.active_elements:
            W1 = self.donor_model.params["L0.mlp.1.W"]
            b1 = self.donor_model.params["L0.mlp.1.b"]
            W2 = self.donor_model.params["L0.mlp.2.W"]
            b2 = self.donor_model.params["L0.mlp.2.b"]
            if self.condition_name == "Condition_F_Random_Matched_Circuit":
                W1 = self.rng.standard_normal(W1.shape).astype(np.float32) * 0.1
                W2 = self.rng.standard_normal(W2.shape).astype(np.float32) * 0.1
            self.mlp_bridges["L0_mlp"] = MLPSlotBridge(
                self.d_tgt, self.d_src, W1, b1, W2, b2, seed=self.seed
            )

        # 3. L1_head_1 (Routing Node)
        if "L1_head_1" in self.active_elements:
            src_idx = 1
            if self.condition_name == "Condition_G_Shuffled_Dependency_Circuit":
                src_idx = 3  # Scramble dependency connection
            W_src = self.donor_model.params["L1.attn.o.W"][src_idx*self.h_dim_src : (src_idx+1)*self.h_dim_src, :]
            if self.condition_name == "Condition_F_Random_Matched_Circuit":
                W_src = self.rng.standard_normal(W_src.shape).astype(np.float32) * 0.1
            self.head_bridges["L1_head_1"] = HeadSlotBridge(
                self.h_dim_src, self.d_src, self.h_dim_tgt, self.d_tgt, W_src, seed=self.seed + 1
            )

        # 4. L2_head_3 (Execution Node)
        if "L2_head_3" in self.active_elements:
            src_idx = 3
            if self.condition_name == "Condition_G_Shuffled_Dependency_Circuit":
                src_idx = 0  # Scramble dependency connection
            W_src = self.donor_model.params["L2.attn.o.W"][src_idx*self.h_dim_src : (src_idx+1)*self.h_dim_src, :]
            if self.condition_name == "Condition_F_Random_Matched_Circuit":
                W_src = self.rng.standard_normal(W_src.shape).astype(np.float32) * 0.1
            self.head_bridges["L2_head_3"] = HeadSlotBridge(
                self.h_dim_src, self.d_src, self.h_dim_tgt, self.d_tgt, W_src, seed=self.seed + 2
            )

    def apply_to_recipient(self):
        """Applies effective bridge weights to recipient slots."""
        # Reset to base parameters first
        for k in ["L0.attn.o.W", "L1.attn.o.W", "L2.attn.o.W", "L0.mlp.1.W", "L0.mlp.2.W"]:
            if k in self.recipient_model.params:
                self.recipient_model.params[k] = self.recipient_backup[k].copy()

        # Apply Head 0 of Layer 0
        if "L0_head_2" in self.head_bridges:
            eff = self.head_bridges["L0_head_2"].get_effective_weights()
            self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = eff

        # Apply MLP in Layer 0
        if "L0_mlp" in self.mlp_bridges:
            br = self.mlp_bridges["L0_mlp"]
            eff_1 = br.W_in @ br.W_mlp1
            eff_2 = br.W_mlp2 @ br.W_out
            self.recipient_model.params["L0.mlp.1.W"][:, :256] = eff_1
            self.recipient_model.params["L0.mlp.2.W"][:256, :] = eff_2

        # Apply Head 0 of Layer 1
        if "L1_head_1" in self.head_bridges:
            eff = self.head_bridges["L1_head_1"].get_effective_weights()
            self.recipient_model.params["L1.attn.o.W"][:self.h_dim_tgt, :] = eff

        # Apply Head 0 of Layer 2
        if "L2_head_3" in self.head_bridges:
            eff = self.head_bridges["L2_head_3"].get_effective_weights()
            self.recipient_model.params["L2.attn.o.W"][:self.h_dim_tgt, :] = eff

    def ablate_all_fdc(self):
        """Surgically zeroes out all transplanted FDC slots in the recipient."""
        if "L0_head_2" in self.head_bridges:
            self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = 0.0
        if "L0_mlp" in self.mlp_bridges:
            self.recipient_model.params["L0.mlp.1.W"][:, :256] = 0.0
            self.recipient_model.params["L0.mlp.2.W"][:256, :] = 0.0
        if "L1_head_1" in self.head_bridges:
            self.recipient_model.params["L1.attn.o.W"][:self.h_dim_tgt, :] = 0.0
        if "L2_head_3" in self.head_bridges:
            self.recipient_model.params["L2.attn.o.W"][:self.h_dim_tgt, :] = 0.0

    def restore_all_fdc(self):
        """Restores the FDC slots to their active states."""
        self.apply_to_recipient()

    def ablate_node(self, node_id: str):
        """Surgically zeroes out a specific single node of the transferred FDC."""
        if node_id == "L0_head_2":
            self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = 0.0
        elif node_id == "L0_mlp":
            self.recipient_model.params["L0.mlp.1.W"][:, :256] = 0.0
            self.recipient_model.params["L0.mlp.2.W"][:256, :] = 0.0
        elif node_id == "L1_head_1":
            self.recipient_model.params["L1.attn.o.W"][:self.h_dim_tgt, :] = 0.0
        elif node_id == "L2_head_3":
            self.recipient_model.params["L2.attn.o.W"][:self.h_dim_tgt, :] = 0.0

    def restore_node(self, node_id: str):
        """Restores a specific single node of the transferred FDC."""
        if node_id == "L0_head_2" and "L0_head_2" in self.head_bridges:
            self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = self.head_bridges["L0_head_2"].get_effective_weights()
        elif node_id == "L0_mlp" and "L0_mlp" in self.mlp_bridges:
            br = self.mlp_bridges["L0_mlp"]
            self.recipient_model.params["L0.mlp.1.W"][:, :256] = br.W_in @ br.W_mlp1
            self.recipient_model.params["L0.mlp.2.W"][:256, :] = br.W_mlp2 @ br.W_out
        elif node_id == "L1_head_1" and "L1_head_1" in self.head_bridges:
            self.recipient_model.params["L1.attn.o.W"][:self.h_dim_tgt, :] = self.head_bridges["L1_head_1"].get_effective_weights()
        elif node_id == "L2_head_3" and "L2_head_3" in self.head_bridges:
            self.recipient_model.params["L2.attn.o.W"][:self.h_dim_tgt, :] = self.head_bridges["L2_head_3"].get_effective_weights()

    def train_adapters(self, train_texts: List[str], epochs: int = 15, lr: float = 3e-3) -> Dict[str, Any]:
        """Optimizes the linear interface adapters using Adam and the two-branch functional effect objective."""
        tok = self.recipient_model.tokenizer
        enc_tr = [tok.encode(t, max_len=16) for t in train_texts if len(tok.encode(t, max_len=16)) >= 4]

        # Precompute donor target functional effects
        donor_effects = []
        for t in train_texts[:len(enc_tr)]:
            s_ids = np.array([self.donor_model.tokenizer.encode(t, max_len=16)], dtype=np.int64)
            self.donor_model.head_mask = None
            l_int, _ = self.donor_model.forward(s_ids)
            self.donor_model.head_mask = np.ones((self.donor_model.spec.layers, self.donor_model.spec.heads), dtype=np.float32)
            self.donor_model.head_mask[0, 2] = 0.0
            l_abl, _ = self.donor_model.forward(s_ids)
            self.donor_model.head_mask = None
            donor_effects.append((l_int[0, -1, :] - l_abl[0, -1, :]).astype(np.float32))

        # Adam state
        opt_state = {}
        for nid, br in self.head_bridges.items():
            opt_state[f"{nid}_in"] = {"m": np.zeros_like(br.W_in), "v": np.zeros_like(br.W_in)}
            opt_state[f"{nid}_out"] = {"m": np.zeros_like(br.W_out), "v": np.zeros_like(br.W_out)}
        if "L0_mlp" in self.mlp_bridges:
            br_mlp = self.mlp_bridges["L0_mlp"]
            opt_state["mlp_in"] = {"m": np.zeros_like(br_mlp.W_in), "v": np.zeros_like(br_mlp.W_in)}
            opt_state["mlp_out"] = {"m": np.zeros_like(br_mlp.W_out), "v": np.zeros_like(br_mlp.W_out)}

        t_step = 0
        loss_history = []
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for ep in range(epochs):
            ep_losses = []
            for idx, s in enumerate(enc_tr):
                t_step += 1
                arr = np.array([s], dtype=np.int64)
                target_F = donor_effects[idx]

                # Intact pass
                self.restore_all_fdc()
                logits_int, _ = self.recipient_model.forward(arr)
                y_int = logits_int[0, -1, :]

                # Ablated pass (FDC disabled)
                self.ablate_all_fdc()
                logits_abl, _ = self.recipient_model.forward(arr)
                y_abl = logits_abl[0, -1, :]

                # Restore
                self.restore_all_fdc()

                effect_pred = y_int - y_abl
                diff = effect_pred - target_F
                loss = 0.5 * float(np.mean(diff ** 2))
                ep_losses.append(loss)

                # Two-branch gradient update on adapters
                dlogits = (diff / self.recipient_model.spec.vocab).astype(np.float32)
                # Intact backward
                dint = np.zeros_like(logits_int); dint[0, -1, :] = dlogits
                g_int = self.recipient_model.backward(arr, dint)
                # Ablated backward
                dabl = np.zeros_like(logits_abl); dabl[0, -1, :] = -dlogits
                self.ablate_all_fdc()
                g_abl = self.recipient_model.backward(arr, dabl)
                self.restore_all_fdc()

                # Update head bridges via Adam
                for node_id, br in self.head_bridges.items():
                    l_idx = 0 if node_id == "L0_head_2" else (1 if node_id == "L1_head_1" else 2)
                    key = f"L{l_idx}.attn.o.W"
                    g_eff = g_int[key][:self.h_dim_tgt, :] + g_abl[key][:self.h_dim_tgt, :]
                    dW_in = np.clip(g_eff @ (br.W_comp @ br.W_out).T, -5.0, 5.0)
                    dW_out = np.clip((br.W_in @ br.W_comp).T @ g_eff, -5.0, 5.0)

                    # Adam update W_in
                    st_in = opt_state[f"{node_id}_in"]
                    st_in["m"] = beta1 * st_in["m"] + (1 - beta1) * dW_in
                    st_in["v"] = beta2 * st_in["v"] + (1 - beta2) * (dW_in ** 2)
                    m_hat = st_in["m"] / (1.0 - beta1 ** t_step)
                    v_hat = st_in["v"] / (1.0 - beta2 ** t_step)
                    br.W_in -= lr * m_hat / (np.sqrt(v_hat) + eps)

                    # Adam update W_out
                    st_out = opt_state[f"{node_id}_out"]
                    st_out["m"] = beta1 * st_out["m"] + (1 - beta1) * dW_out
                    st_out["v"] = beta2 * st_out["v"] + (1 - beta2) * (dW_out ** 2)
                    m_hat = st_out["m"] / (1.0 - beta1 ** t_step)
                    v_hat = st_out["v"] / (1.0 - beta2 ** t_step)
                    br.W_out -= lr * m_hat / (np.sqrt(v_hat) + eps)

                # Update MLP bridge via Adam
                if "L0_mlp" in self.mlp_bridges:
                    br_mlp = self.mlp_bridges["L0_mlp"]
                    g_mlp1 = np.clip(g_int["L0.mlp.1.W"][:, :256] + g_abl["L0.mlp.1.W"][:, :256], -5.0, 5.0)
                    g_mlp2 = np.clip(g_int["L0.mlp.2.W"][:256, :] + g_abl["L0.mlp.2.W"][:256, :], -5.0, 5.0)
                    dW_in_mlp = g_mlp1 @ br_mlp.W_mlp1.T
                    dW_out_mlp = br_mlp.W_mlp2.T @ g_mlp2

                    st_m_in = opt_state["mlp_in"]
                    st_m_in["m"] = beta1 * st_m_in["m"] + (1 - beta1) * dW_in_mlp
                    st_m_in["v"] = beta2 * st_m_in["v"] + (1 - beta2) * (dW_in_mlp ** 2)
                    m_hat = st_m_in["m"] / (1.0 - beta1 ** t_step)
                    v_hat = st_m_in["v"] / (1.0 - beta2 ** t_step)
                    br_mlp.W_in -= lr * m_hat / (np.sqrt(v_hat) + eps)

                    st_m_out = opt_state["mlp_out"]
                    st_m_out["m"] = beta1 * st_m_out["m"] + (1 - beta1) * dW_out_mlp
                    st_m_out["v"] = beta2 * st_m_out["v"] + (1 - beta2) * (dW_out_mlp ** 2)
                    m_hat = st_m_out["m"] / (1.0 - beta1 ** t_step)
                    v_hat = st_m_out["v"] / (1.0 - beta2 ** t_step)
                    br_mlp.W_out -= lr * m_hat / (np.sqrt(v_hat) + eps)

                self.apply_to_recipient()

            loss_history.append(float(np.mean(ep_losses)))

        # Verify frozen invariants after training
        all_frozen = True
        for br in self.head_bridges.values():
            if not br.verify_frozen_invariant():
                all_frozen = False
        for br in self.mlp_bridges.values():
            if not br.verify_frozen_invariant():
                all_frozen = False

        return {
            "initial_loss": round(loss_history[0], 4) if loss_history else 0.0,
            "final_loss": round(loss_history[-1], 4) if loss_history else 0.0,
            "loss_drop": round(loss_history[0] - loss_history[-1], 4) if loss_history else 0.0,
            "all_donor_weights_frozen": all_frozen
        }

    def total_adapter_parameters(self) -> int:
        h_params = sum(br.param_count() for br in self.head_bridges.values())
        m_params = sum(br.param_count() for br in self.mlp_bridges.values())
        return h_params + m_params
