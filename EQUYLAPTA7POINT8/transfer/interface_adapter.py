"""transfer/interface_adapter.py

Implements capacity-constrained bidirectional bridges for Attention Heads and MLP blocks
ensuring strict frozen invariants for all donor computation.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Any, Tuple
import numpy as np


class HeadSlotBridge:
    """Translates a single attention head projection while keeping donor weights frozen."""

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

        # Frozen invariant tracking
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

        # Frozen invariant tracking
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
