"""Low-rank alignment (Phase 9): ridge fit + SVD rank truncation.

Reports BEFORE (trivial mean-predictor) vs AFTER (ridge, low-rank) alignment
quality. Used by the synthesis engine as an honest gate: transplants across
weakly-aligned representations are escalated to distillation/routing instead.
"""
from __future__ import annotations

import numpy as np


def fit_projection_lowrank(src_h: np.ndarray, dst_h: np.ndarray, rank: int = 8,
                           ridge: float = 1e-2) -> dict:
    X = src_h.astype(np.float64)
    Y = dst_h.astype(np.float64)
    d_in, d_out = X.shape[1], Y.shape[1]
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + ridge * np.eye(d_in + 1)
    sol = np.linalg.solve(A, Xb.T @ Y)
    W, b = sol[:-1], sol[-1]
    # rank truncation via SVD on the weight matrix
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    k = max(1, min(rank, len(S)))
    Wk = (U[:, :k] * S[:k]) @ Vt[:k]
    energy = float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-9))
    pred = X @ Wk + b
    cos = float(np.mean([np.dot(pred[i], Y[i]) /
                         (np.linalg.norm(pred[i]) * np.linalg.norm(Y[i]) + 1e-9)
                         for i in range(len(Y))]))
    r2 = float(1 - ((pred - Y) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9))
    return {"W": Wk.astype(np.float32), "b": b.astype(np.float32),
            "rank": int(k), "variance_energy": round(energy, 3),
            "cosine_alignment": round(cos, 3), "r2": round(r2, 3),
            "d_in": d_in, "d_out": d_out}
