"""[WORKING micro] Adapter / projection layer + representation alignment.

When weights cannot merge (different dims/arch), align REPRESENTATIONS:
fit a ridge-regression projection from model A hidden states to model B
hidden states on paired inputs, report alignment quality (R^2 / cosine).
This is the honest micro version of learned projection adapters; a trained
nonlinear adapter [REQUIRES GPU] plugs into the same interface at scale.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def collect_hidden(m, texts, layer: Optional[int] = None) -> np.ndarray:
    """Mean-pooled hidden states per text at `layer` (default: last)."""
    from models.base import np as _  # noqa
    outs = []
    L = m.spec.layers - 1 if layer is None else layer
    for t in texts:
        ids = m.tokenizer.encode(t, max_len=m.spec.ctx_len)
        if len(ids) < 2:
            continue
        arr = ids[None, :]
        _, cache = m.forward(arr, collect=True)
        h = cache["hiddens"][L][0]           # [T, d]
        outs.append(h.mean(0))
    return np.stack(outs)


def fit_projection(src_h: np.ndarray, dst_h: np.ndarray, ridge: float = 1e-2):
    """Learn W,b minimizing ||src @ W + b - dst||^2 + ridge||W||^2."""
    X = src_h.astype(np.float64)
    Y = dst_h.astype(np.float64)
    d_in, d_out = X.shape[1], Y.shape[1]
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + ridge * np.eye(d_in + 1)
    B = Xb.T @ Y
    sol = np.linalg.solve(A, B)
    W, b = sol[:-1], sol[-1]
    pred = Xb @ sol
    # alignment metrics
    cos = np.mean([np.dot(pred[i], Y[i]) /
                   (np.linalg.norm(pred[i]) * np.linalg.norm(Y[i]) + 1e-9)
                   for i in range(len(Y))])
    r2 = 1 - ((pred - Y) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)
    return {"W": W.astype(np.float32), "b": b.astype(np.float32),
            "cosine_alignment": round(float(cos), 3), "r2": round(float(r2), 3),
            "d_in": d_in, "d_out": d_out}


def align_models(model_a, model_b, texts) -> dict:
    """Full pipeline: paired hidden collection -> projection -> report."""
    ha = collect_hidden(model_a, texts)
    hb = collect_hidden(model_b, texts)
    n = min(len(ha), len(hb))
    return fit_projection(ha[:n], hb[:n])
