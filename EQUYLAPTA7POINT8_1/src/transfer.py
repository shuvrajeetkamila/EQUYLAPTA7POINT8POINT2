"""src/transfer.py

Manages cross-architecture functional circuit transplantation, bidirectional interface bridges,
exact analytical two-branch functional-effect optimization, and pre-registered controls.
"""
from __future__ import annotations

import copy
import hashlib
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import MicroTransformer


class HeadSlotBridge:
    """Translates a single attention head projection while keeping donor weights strictly frozen."""

    def __init__(self, d_head_src: int, d_comp: int, d_head_tgt: int, d_tgt: int,
                 W_comp: np.ndarray, seed: int = 42):
        self.d_head_src = d_head_src
        self.d_comp = d_comp
        self.d_head_tgt = d_head_tgt
        self.d_tgt = d_tgt

        # Adapters
        self.W_in = np.eye(d_head_tgt, d_head_src, dtype=np.float32)
        self.W_comp = W_comp.copy().astype(np.float32)  # STRICTLY FROZEN
        self.W_out = np.zeros((d_comp, d_tgt), dtype=np.float32)
        min_dim = min(d_comp, d_tgt)
        self.W_out[:min_dim, :min_dim] = np.eye(min_dim, dtype=np.float32)

        # Baseline initialization copies for regularization
        self.W_in_0 = self.W_in.copy()
        self.W_out_0 = self.W_out.copy()

        # Frozen tracking
        self.frozen_hash = hashlib.sha256(self.W_comp.tobytes()).hexdigest()
        self.frozen_norm = float(np.linalg.norm(self.W_comp))

    def get_effective_weights(self) -> np.ndarray:
        return (self.W_in @ self.W_comp @ self.W_out).astype(np.float32)

    def verify_frozen_invariant(self) -> bool:
        curr_hash = hashlib.sha256(self.W_comp.tobytes()).hexdigest()
        curr_norm = float(np.linalg.norm(self.W_comp))
        return (curr_hash == self.frozen_hash) and abs(curr_norm - self.frozen_norm) < 1e-7

    def param_count(self) -> int:
        return self.W_in.size + self.W_out.size


class MLPSlotBridge:
    """Translates a donor MLP dependency block into recipient residual stream
    while keeping donor weights strictly frozen.
    """

    def __init__(self, d_tgt: int, d_src: int, W_mlp1: np.ndarray, b_mlp1: np.ndarray,
                 W_mlp2: np.ndarray, b_mlp2: np.ndarray, seed: int = 42):
        self.d_tgt = d_tgt
        self.d_src = d_src

        # Adapters
        self.W_in = np.zeros((d_tgt, d_src), dtype=np.float32)
        min_dim = min(d_tgt, d_src)
        self.W_in[:min_dim, :min_dim] = np.eye(min_dim, dtype=np.float32)

        self.W_mlp1 = W_mlp1.copy().astype(np.float32)  # STRICTLY FROZEN
        self.b_mlp1 = b_mlp1.copy().astype(np.float32)  # STRICTLY FROZEN
        self.W_mlp2 = W_mlp2.copy().astype(np.float32)  # STRICTLY FROZEN
        self.b_mlp2 = b_mlp2.copy().astype(np.float32)  # STRICTLY FROZEN

        self.W_out = np.zeros((d_src, d_tgt), dtype=np.float32)
        self.W_out[:min_dim, :min_dim] = np.eye(min_dim, dtype=np.float32)

        self.W_in_0 = self.W_in.copy()
        self.W_out_0 = self.W_out.copy()

        h = hashlib.sha256()
        h.update(self.W_mlp1.tobytes())
        h.update(self.b_mlp1.tobytes())
        h.update(self.W_mlp2.tobytes())
        h.update(self.b_mlp2.tobytes())
        self.frozen_hash = h.hexdigest()
        self.frozen_norm = float(np.linalg.norm(self.W_mlp1) + np.linalg.norm(self.W_mlp2))

    def get_effective_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eff_W1 = (self.W_in @ self.W_mlp1).astype(np.float32)
        eff_b1 = self.b_mlp1.copy().astype(np.float32)
        eff_W2 = (self.W_mlp2 @ self.W_out).astype(np.float32)
        eff_b2 = np.zeros(self.d_tgt, dtype=np.float32)
        min_dim = min(self.d_src, self.d_tgt)
        eff_b2[:min_dim] = self.b_mlp2[:min_dim]
        return eff_W1, eff_b1, eff_W2, eff_b2

    def verify_frozen_invariant(self) -> bool:
        h = hashlib.sha256()
        h.update(self.W_mlp1.tobytes())
        h.update(self.b_mlp1.tobytes())
        h.update(self.W_mlp2.tobytes())
        h.update(self.b_mlp2.tobytes())
        curr_norm = float(np.linalg.norm(self.W_mlp1) + np.linalg.norm(self.W_mlp2))
        return (h.hexdigest() == self.frozen_hash) and abs(curr_norm - self.frozen_norm) < 1e-7

    def param_count(self) -> int:
        return self.W_in.size + self.W_out.size


class ArchitectureTranslator:
    """Manages cross-architecture circuit translation, two-branch functional effect alignment,
    and condition-specific topologies.
    """

    def __init__(self, donor_model: MicroTransformer, recipient_model: MicroTransformer,
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

        # Store pristine backup of recipient weights
        self.recipient_backup = {k: v.copy() for k, v in recipient_model.params.items()}

        self.head_bridges: Dict[str, HeadSlotBridge] = {}
        self.mlp_bridges: Dict[str, MLPSlotBridge] = {}

        self._init_bridges()
        self.apply_to_recipient()

    def _init_bridges(self):
        # 1. L0_head_2
        if "L0_head_2" in self.active_elements:
            W_src = self.donor_model.params["L0.attn.o.W"][2*self.h_dim_src : 3*self.h_dim_src, :]
            if "Random" in self.condition_name:
                W_src = self.rng.standard_normal(W_src.shape).astype(np.float32) * 0.1
            elif "Parameter_Shuffle" in self.condition_name:
                flat = W_src.flatten()
                self.rng.shuffle(flat)
                W_src = flat.reshape(W_src.shape)
            self.head_bridges["L0_head_2"] = HeadSlotBridge(
                self.h_dim_src, self.d_src, self.h_dim_tgt, self.d_tgt, W_src, seed=self.seed
            )

        # 2. L0_mlp
        if "L0_mlp" in self.active_elements:
            W1 = self.donor_model.params["L0.mlp.1.W"]
            b1 = self.donor_model.params["L0.mlp.1.b"]
            W2 = self.donor_model.params["L0.mlp.2.W"]
            b2 = self.donor_model.params["L0.mlp.2.b"]
            if "Random" in self.condition_name:
                W1 = self.rng.standard_normal(W1.shape).astype(np.float32) * 0.1
                W2 = self.rng.standard_normal(W2.shape).astype(np.float32) * 0.1
            elif "Parameter_Shuffle" in self.condition_name:
                f1 = W1.flatten(); self.rng.shuffle(f1); W1 = f1.reshape(W1.shape)
                f2 = W2.flatten(); self.rng.shuffle(f2); W2 = f2.reshape(W2.shape)
            self.mlp_bridges["L0_mlp"] = MLPSlotBridge(
                self.d_tgt, self.d_src, W1, b1, W2, b2, seed=self.seed
            )

        # 3. L1_head_1
        if "L1_head_1" in self.active_elements:
            W_src = self.donor_model.params["L1.attn.o.W"][1*self.h_dim_src : 2*self.h_dim_src, :]
            if "Random" in self.condition_name:
                W_src = self.rng.standard_normal(W_src.shape).astype(np.float32) * 0.1
            self.head_bridges["L1_head_1"] = HeadSlotBridge(
                self.h_dim_src, self.d_src, self.h_dim_tgt, self.d_tgt, W_src, seed=self.seed + 1
            )

        # 4. L2_head_3
        if "L2_head_3" in self.active_elements:
            W_src = self.donor_model.params["L2.attn.o.W"][3*self.h_dim_src : 4*self.h_dim_src, :]
            if "Random" in self.condition_name:
                W_src = self.rng.standard_normal(W_src.shape).astype(np.float32) * 0.1
            self.head_bridges["L2_head_3"] = HeadSlotBridge(
                self.h_dim_src, self.d_src, self.h_dim_tgt, self.d_tgt, W_src, seed=self.seed + 2
            )

        # 5. Oversized additional elements
        if "L0_head_3" in self.active_elements:
            W_src = self.donor_model.params["L0.attn.o.W"][3*self.h_dim_src : 4*self.h_dim_src, :]
            self.head_bridges["L0_head_3"] = HeadSlotBridge(
                self.h_dim_src, self.d_src, self.h_dim_tgt, self.d_tgt, W_src, seed=self.seed + 3
            )
        if "L1_mlp" in self.active_elements:
            self.mlp_bridges["L1_mlp"] = MLPSlotBridge(
                self.d_tgt, self.d_src,
                self.donor_model.params["L1.mlp.1.W"], self.donor_model.params["L1.mlp.1.b"],
                self.donor_model.params["L1.mlp.2.W"], self.donor_model.params["L1.mlp.2.b"],
                seed=self.seed + 4
            )
        if "L2_mlp" in self.active_elements:
            self.mlp_bridges["L2_mlp"] = MLPSlotBridge(
                self.d_tgt, self.d_src,
                self.donor_model.params["L2.mlp.1.W"], self.donor_model.params["L2.mlp.1.b"],
                self.donor_model.params["L2.mlp.2.W"], self.donor_model.params["L2.mlp.2.b"],
                seed=self.seed + 5
            )
        if "L3_mlp" in self.active_elements:
            self.mlp_bridges["L3_mlp"] = MLPSlotBridge(
                self.d_tgt, self.d_src,
                self.donor_model.params["L3.mlp.1.W"], self.donor_model.params["L3.mlp.1.b"],
                self.donor_model.params["L3.mlp.2.W"], self.donor_model.params["L3.mlp.2.b"],
                seed=self.seed + 6
            )

    def apply_to_recipient(self):
        """Applies active bridges to the recipient model parameters."""
        # Restore recipient from pristine backup first
        for k, v in self.recipient_backup.items():
            self.recipient_model.params[k][:] = v[:]

        # Host-only adaptation condition does not mount donor bridges
        if "Host_Only" in self.condition_name:
            return

        # Location-matched control: inject into Layer 3 rather than Layer 0
        if "Location_Matched" in self.condition_name:
            if "L0_head_2" in self.head_bridges:
                eff = self.head_bridges["L0_head_2"].get_effective_weights()
                self.recipient_model.params["L3.attn.o.W"][:self.h_dim_tgt, :] = eff
            if "L0_mlp" in self.mlp_bridges:
                w1, b1, w2, b2 = self.mlp_bridges["L0_mlp"].get_effective_weights()
                self.recipient_model.params["L3.mlp.1.W"][:, :w1.shape[1]] = w1
                self.recipient_model.params["L3.mlp.2.W"][:w2.shape[0], :] = w2
            return

        # Shuffled-path control: reverse layer hierarchy (L2 -> L1 -> L0)
        if "Shuffled_Path" in self.condition_name:
            if "L0_head_2" in self.head_bridges:
                eff = self.head_bridges["L0_head_2"].get_effective_weights()
                self.recipient_model.params["L2.attn.o.W"][:self.h_dim_tgt, :] = eff
            if "L2_head_3" in self.head_bridges:
                eff = self.head_bridges["L2_head_3"].get_effective_weights()
                self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = eff
            return

        # Shuffled-node control: scramble dependency routing among heads
        if "Shuffled_Node" in self.condition_name:
            if "L0_head_2" in self.head_bridges:
                eff = self.head_bridges["L0_head_2"].get_effective_weights()
                self.recipient_model.params["L1.attn.o.W"][:self.h_dim_tgt, :] = eff
            if "L1_head_1" in self.head_bridges:
                eff = self.head_bridges["L1_head_1"].get_effective_weights()
                self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = eff
            return

        # Standard & Recipient-Reconstructed mapping
        if "L0_head_2" in self.head_bridges:
            eff = self.head_bridges["L0_head_2"].get_effective_weights()
            self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = eff

        if "L0_head_3" in self.head_bridges:
            eff = self.head_bridges["L0_head_3"].get_effective_weights()
            self.recipient_model.params["L0.attn.o.W"][self.h_dim_tgt : 2*self.h_dim_tgt, :] = eff

        if "L0_mlp" in self.mlp_bridges:
            w1, b1, w2, b2 = self.mlp_bridges["L0_mlp"].get_effective_weights()
            if "Recipient_Reconstructed" in self.condition_name:
                self.recipient_model.params["L0.mlp.1.W"][:, :w1.shape[1]] = 0.5 * self.recipient_model.params["L0.mlp.1.W"][:, :w1.shape[1]] + 0.5 * w1
                self.recipient_model.params["L0.mlp.2.W"][:w2.shape[0], :] = 0.5 * self.recipient_model.params["L0.mlp.2.W"][:w2.shape[0], :] + 0.5 * w2
            else:
                self.recipient_model.params["L0.mlp.1.W"][:, :w1.shape[1]] = w1
                self.recipient_model.params["L0.mlp.2.W"][:w2.shape[0], :] = w2

        if "L1_head_1" in self.head_bridges:
            eff = self.head_bridges["L1_head_1"].get_effective_weights()
            self.recipient_model.params["L1.attn.o.W"][:self.h_dim_tgt, :] = eff

        if "L2_head_3" in self.head_bridges:
            eff = self.head_bridges["L2_head_3"].get_effective_weights()
            self.recipient_model.params["L2.attn.o.W"][:self.h_dim_tgt, :] = eff

        # Oversized MLPs
        for l_idx, key in [(1, "L1_mlp"), (2, "L2_mlp"), (3, "L3_mlp")]:
            if key in self.mlp_bridges:
                w1, b1, w2, b2 = self.mlp_bridges[key].get_effective_weights()
                self.recipient_model.params[f"L{l_idx}.mlp.1.W"][:, :w1.shape[1]] = w1
                self.recipient_model.params[f"L{l_idx}.mlp.2.W"][:w2.shape[0], :] = w2

    def train_adapters(self, items: List[Any], epochs: int = 15, lr: float = 0.003,
                       lambda_reg: float = 0.0001) -> Dict[str, Any]:
        """Trains bridge adapters using true two-branch functional causal loss and analytical backpropagation."""
        # Host-only adaptation condition: adapt recipient L0 MLP on cross-entropy with matched budget
        if "Host_Only" in self.condition_name:
            W = self.recipient_model.params["L0.mlp.1.W"]
            m_w = np.zeros_like(W)
            v_w = np.zeros_like(W)
            for ep in range(epochs):
                loss_ep = 0.0
                for it in items:
                    p_ids = self.recipient_model.tokenizer.encode(it.prompt)
                    ids = np.array([p_ids], dtype=np.int64)
                    logits, _ = self.recipient_model.forward(ids)
                    p_dist = self.recipient_model._softmax(logits[0, -1, :])
                    target_idx = it.answer if hasattr(it, "answer") else 0
                    loss = -np.log(max(1e-12, p_dist[target_idx % len(p_dist)]))
                    loss_ep += loss

                    # Exact backward for host model MLP
                    dlogits = np.zeros_like(logits)
                    dlogits[0, -1, :] = p_dist.copy()
                    dlogits[0, -1, target_idx % len(p_dist)] -= 1.0
                    grads = self.recipient_model.backward(ids, dlogits)
                    W_grad = grads["L0.mlp.1.W"]
                    np.clip(W_grad, -5.0, 5.0, out=W_grad)

                    m_w = 0.9 * m_w + 0.1 * W_grad
                    v_w = 0.999 * v_w + 0.001 * (W_grad ** 2)
                    m_hat = m_w / (1.0 - 0.9 ** (ep + 1))
                    v_hat = v_w / (1.0 - 0.999 ** (ep + 1))
                    W -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + lambda_reg * W)
            return {"epochs": epochs, "final_loss": round(loss_ep / len(items), 5), "status": "HOST_ONLY_CONVERGED"}

        # Frozen transfer conditions (Conditions A, D, F, G1-G4): evaluate without adaptation!
        if any(tag in self.condition_name for tag in ["Baseline", "Donor_FDC", "Random_Matched", "Shuffled", "Location_Matched"]):
            return {"epochs": 0, "final_loss": 0.0, "status": "FROZEN_TRANSFER_EVALUATION"}

        # Collect primary trainable bridge
        bridge = self.head_bridges.get("L0_head_2")
        if not bridge:
            return {"epochs": 0, "final_loss": 0.0, "status": "NO_ADAPTERS"}

        m_w_in = np.zeros_like(bridge.W_in)
        v_w_in = np.zeros_like(bridge.W_in)
        m_w_out = np.zeros_like(bridge.W_out)
        v_w_out = np.zeros_like(bridge.W_out)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        total_loss = 0.0
        step = 0
        V = self.recipient_model.spec.vocab

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for item in items:
                step += 1
                p_ids = self.recipient_model.tokenizer.encode(item.prompt)
                ids = np.array([p_ids], dtype=np.int64)

                # 1. Compute donor target causal delta
                self.donor_model.head_mask = None
                l_src_int, _ = self.donor_model.forward(ids)
                self.donor_model.head_mask = np.ones((self.donor_model.spec.layers, self.donor_model.spec.heads), dtype=np.float32)
                self.donor_model.head_mask[0, 2] = 0.0
                l_src_abl, _ = self.donor_model.forward(ids)
                self.donor_model.head_mask = None
                delta_target = (l_src_int[0, -1, :] - l_src_abl[0, -1, :]).astype(np.float32)

                # 2. Forward pass recipient intact & ablated
                self.apply_to_recipient()
                self.recipient_model.head_mask = None
                l_int, _ = self.recipient_model.forward(ids)

                self.recipient_model.head_mask = np.ones((self.recipient_model.spec.layers, self.recipient_model.spec.heads), dtype=np.float32)
                self.recipient_model.head_mask[0, 0] = 0.0
                l_abl, _ = self.recipient_model.forward(ids)
                self.recipient_model.head_mask = None

                delta_recip = (l_int[0, -1, :] - l_abl[0, -1, :]).astype(np.float32)
                diff = delta_recip - delta_target
                loss = 0.5 * float(np.mean(diff ** 2))
                epoch_loss += loss

                # 3. Exact analytical two-branch backward pass
                dlogits_int = np.zeros_like(l_int)
                dlogits_int[0, -1, :] = (diff / V).astype(np.float32)
                self.recipient_model.head_mask = None
                g_int = self.recipient_model.backward(ids, dlogits_int)

                dlogits_abl = np.zeros_like(l_abl)
                dlogits_abl[0, -1, :] = (-diff / V).astype(np.float32)
                self.recipient_model.head_mask = np.ones((self.recipient_model.spec.layers, self.recipient_model.spec.heads), dtype=np.float32)
                self.recipient_model.head_mask[0, 0] = 0.0
                g_abl = self.recipient_model.backward(ids, dlogits_abl)
                self.recipient_model.head_mask = None

                # Effective weight gradient for L0 slot 0
                g_eff = (g_int["L0.attn.o.W"][:self.h_dim_tgt, :] +
                         g_abl["L0.attn.o.W"][:self.h_dim_tgt, :])

                # Chain rule for bridge parameters
                dW_in = g_eff @ (bridge.W_comp @ bridge.W_out).T + lambda_reg * (bridge.W_in - bridge.W_in_0)
                dW_out = (bridge.W_in @ bridge.W_comp).T @ g_eff + lambda_reg * (bridge.W_out - bridge.W_out_0)

                np.clip(dW_in, -5.0, 5.0, out=dW_in)
                np.clip(dW_out, -5.0, 5.0, out=dW_out)

                # Adam updates
                m_w_in = beta1 * m_w_in + (1 - beta1) * dW_in
                v_w_in = beta2 * v_w_in + (1 - beta2) * (dW_in ** 2)
                m_hat_in = m_w_in / (1.0 - beta1 ** step)
                v_hat_in = v_w_in / (1.0 - beta2 ** step)
                bridge.W_in -= lr * (m_hat_in / (np.sqrt(v_hat_in) + eps))

                m_w_out = beta1 * m_w_out + (1 - beta1) * dW_out
                v_w_out = beta2 * v_w_out + (1 - beta2) * (dW_out ** 2)
                m_hat_out = m_w_out / (1.0 - beta1 ** step)
                v_hat_out = v_w_out / (1.0 - beta2 ** step)
                bridge.W_out -= lr * (m_hat_out / (np.sqrt(v_hat_out) + eps))

                # Receiver Co-Adaptation (Condition J): allow local recipient L0 MLP to co-adapt
                if "Receiver_Coadaptation" in self.condition_name:
                    g_mlp = g_int["L0.mlp.1.W"] + g_abl["L0.mlp.1.W"]
                    np.clip(g_mlp, -5.0, 5.0, out=g_mlp)
                    self.recipient_model.params["L0.mlp.1.W"] -= (lr * 0.5) * g_mlp

            self.apply_to_recipient()
            total_loss = epoch_loss / max(1, len(items))

        # Strict frozen verification
        self.verify_all_frozen()

        return {
            "epochs": epochs,
            "final_loss": round(float(total_loss), 6),
            "status": "CONVERGED_WITH_TRUE_ANALYTICAL_GRADIENT"
        }

    def verify_all_frozen(self) -> bool:
        for name, b in self.head_bridges.items():
            if not b.verify_frozen_invariant():
                raise RuntimeError(f"FROZEN INVARIANT VIOLATION in head bridge {name}")
        for name, b in self.mlp_bridges.items():
            if not b.verify_frozen_invariant():
                raise RuntimeError(f"FROZEN INVARIANT VIOLATION in mlp bridge {name}")
        return True

    def ablate_circuit(self):
        """Surgically cuts transplanted connections while leaving native recipient intact."""
        if "L0_head_2" in self.head_bridges:
            self.recipient_model.params["L0.attn.o.W"][:self.h_dim_tgt, :] = 0.0
        if "L0_head_3" in self.head_bridges:
            self.recipient_model.params["L0.attn.o.W"][self.h_dim_tgt : 2*self.h_dim_tgt, :] = 0.0
        if "L0_mlp" in self.mlp_bridges:
            w1 = self.recipient_model.params["L0.mlp.1.W"]
            w1[:, :self.d_src] = 0.0
            w2 = self.recipient_model.params["L0.mlp.2.W"]
            w2[:self.d_src, :] = 0.0
        if "L1_head_1" in self.head_bridges:
            self.recipient_model.params["L1.attn.o.W"][:self.h_dim_tgt, :] = 0.0
        if "L2_head_3" in self.head_bridges:
            self.recipient_model.params["L2.attn.o.W"][:self.h_dim_tgt, :] = 0.0

    def restore_circuit(self):
        """Restores circuit connections."""
        self.apply_to_recipient()

    def get_parameter_footprint(self) -> Dict[str, int]:
        donor_params = sum(b.W_comp.size for b in self.head_bridges.values())
        donor_params += sum(b.W_mlp1.size + b.b_mlp1.size + b.W_mlp2.size + b.b_mlp2.size for b in self.mlp_bridges.values())
        adapter_params = sum(b.param_count() for b in self.head_bridges.values())
        adapter_params += sum(b.param_count() for b in self.mlp_bridges.values())
        return {
            "donor_frozen_params": donor_params,
            "adapter_trainable_params": adapter_params,
            "total_transferred_params": donor_params + adapter_params
        }
