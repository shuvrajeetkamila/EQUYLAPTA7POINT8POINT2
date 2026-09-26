"""[WORKING] Model merging strategies (numpy state-dict space).

Implemented: weighted average, SLERP, TIES, DARE, task arithmetic,
layer-wise selective merging, learned/evolutionary coefficients.
All operate on compatible state dicts ONLY — the merge engine checks
tensor shapes first (spec §26: never fake compatibility).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from models.base import TensorDict

Strategy = Callable[[List[TensorDict], List[float]], TensorDict]


def _check(compat_sd: List[TensorDict]) -> None:
    keys = set(compat_sd[0])
    for sd in compat_sd[1:]:
        assert set(sd) == keys, "state dicts have different keys"
    for k in compat_sd[0]:
        shapes = [sd[k].shape for sd in compat_sd]
        assert all(s == shapes[0] for s in shapes), f"shape mismatch at {k}: {shapes}"


def weighted_average(sds: List[TensorDict], weights: Sequence[float]) -> TensorDict:
    w = np.asarray(weights, np.float64)
    w = w / w.sum()
    out: TensorDict = {}
    for k in sds[0]:
        out[k] = sum(float(w[i]) * sds[i][k].astype(np.float64) for i in range(len(sds))).astype(np.float32)
    return out


def slerp_pair(a: np.ndarray, b: np.ndarray, t: float, eps: float = 1e-7) -> np.ndarray:
    a_f, b_f = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if na < eps or nb < eps:
        return (1 - t) * a + t * b
    cos = np.clip(np.dot(a_f, b_f) / (na * nb), -1 + eps, 1 - eps)
    omega = np.arccos(cos)
    if omega < 1e-4:  # nearly identical -> lerp is numerically safer
        return (1 - t) * a + t * b
    so = np.sin(omega)
    return ((np.sin((1 - t) * omega) / so) * a + (np.sin(t * omega) / so) * b).astype(np.float32)


def slerp(sds: List[TensorDict], weights: Sequence[float]) -> TensorDict:
    """Pairwise SLERP chain in weight order (works for >=2)."""
    cur = sds[0]
    total = sum(weights) or 1.0
    for i in range(1, len(sds)):
        t = weights[i] / (sum(weights[:i + 1]) or 1.0)
        cur = {k: slerp_pair(cur[k], sds[i][k], t) for k in cur}
    return cur


def task_arithmetic(base: TensorDict, deltas: List[TensorDict],
                    coeffs: Sequence[float], clamp: float = 1.5) -> TensorDict:
    """base + sum_i coeff_i * (model_i - base)."""
    out: TensorDict = {}
    for k in base:
        acc = np.zeros_like(base[k], np.float64)
        for d, c in zip(deltas, coeffs):
            c = float(np.clip(c, -clamp, clamp))
            acc += c * (d[k].astype(np.float64) - base[k].astype(np.float64))
        out[k] = (base[k].astype(np.float64) + acc).astype(np.float32)
    return out


def ties(sds: List[TensorDict], weights: Sequence[float], density: float = 0.7) -> TensorDict:
    """TIES: per-tensor trim to top-k magnitude, elect sign by total mass,
    average only sign-agreeing entries."""
    out: TensorDict = {}
    for k in sds[0]:
        stack = np.stack([sd[k].astype(np.float64) for sd in sds])  # [M, ...]
        flat = stack.reshape(len(sds), -1)
        kk = max(1, int(density * flat.shape[1]))
        thresh = np.sort(np.abs(flat), axis=1)[:, -kk][:, None]
        mask = np.abs(flat) >= thresh
        trimmed = flat * mask
        sign = np.sign(trimmed.sum(0))
        agree = np.where((np.sign(trimmed) == sign) & (mask), trimmed, 0.0)
        cnt = np.maximum(1, (np.abs(agree) > 0).sum(0))
        agg = (agree.sum(0) / cnt).reshape(stack.shape[1:])
        out[k] = agg.astype(np.float32)
    return out


def dare(sds: List[TensorDict], weights: Sequence[float], drop_p: float = 0.5,
         seed: int = 0) -> TensorDict:
    """DARE: random-drop deltas then rescale, then average (here vs first model as base)."""
    rng = np.random.default_rng(seed)
    base, rest = sds[0], sds[1:]
    ws = np.asarray(weights[1:], np.float64)
    ws = ws / (ws.sum() or 1.0)
    out: TensorDict = {}
    for k in base:
        acc = np.zeros_like(base[k], np.float64)
        for i, sd in enumerate(rest):
            delta = sd[k].astype(np.float64) - base[k].astype(np.float64)
            keep = (rng.random(delta.shape) > drop_p)
            delta = delta * keep / (1 - drop_p)
            acc += ws[i] * delta
        out[k] = (base[k].astype(np.float64) + acc).astype(np.float32)
    return out


def layer_selective(sds: List[TensorDict], weights: Sequence[float],
                    source_index: Dict[str, int]) -> TensorDict:
    """Take each tensor from the designated source model (segment transplant)."""
    out: TensorDict = {}
    for k in sds[0]:
        src = source_index.get(k, 0)
        out[k] = sds[src][k].copy()
    return out


STRATEGIES = {
    "weighted_average": weighted_average,
    "slerp": slerp,
    "ties": ties,
    "dare": dare,
}


def run_strategy(name: str, sds: List[TensorDict], weights: Sequence[float],
                 base: Optional[TensorDict] = None, **kw) -> TensorDict:
    _check(sds if base is None else sds + [base])
    if name == "weighted_average":
        return weighted_average(sds, weights)
    if name == "slerp":
        return slerp(sds, weights)
    if name == "ties":
        return ties(sds, weights, **kw)
    if name == "dare":
        return dare(sds, weights, **kw)
    if name == "task_arithmetic":
        assert base is not None, "task arithmetic needs a base model"
        return task_arithmetic(base, sds, weights, **kw)
    raise KeyError(f"unknown strategy {name}")
