"""Representation alignment subsystem (M3, spec §§6-7).

Maps SOURCE representations into TARGET space. Handles dimension mismatch
EXPLICITLY: identity alignment across mismatched dims raises
DimensionMismatchError — silent reshape/truncate/pad is forbidden.

Methods: identity | linear (ridge) | low_rank (ridge + SVD truncation) |
learned (gradient-trained linear) | activation_stats (mean/std matching).

Every fit records train/validation/HELD-OUT losses on the held-out split,
baseline (pre-alignment) similarity, and a SATURATION flag when the sample
count is too small relative to representation dimension (never report
"perfect alignment" from a saturated fit). Checkpoints are stored OUTSIDE
the project tree.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple

import numpy as np

ALIGNMENT_METHODS = ["identity", "linear", "low_rank", "learned", "activation_stats"]


class DimensionMismatchError(Exception):
    def __init__(self, what, src_shape, dst_shape, reason):
        super().__init__(
            f"ALIGNMENT DIMENSION MISMATCH — {what}\n"
            f"  SOURCE shape: {src_shape}\n  TARGET shape: {dst_shape}\n"
            f"  REASON: {reason}")


@dataclass
class AlignmentResult:
    alignment_id: str
    method: str
    d_in: int
    d_out: int
    n_train: int
    n_val: int
    n_heldout: int
    train_loss: float
    val_loss: float
    heldout_loss: float
    baseline_similarity: Optional[float]      # pre-alignment (None if dims differ)
    aligned_similarity: float                 # held-out cosine after alignment
    heldout_r2: float
    saturated: bool
    saturation_note: str
    seed: int
    checkpoint: str = ""                      # EXTERNAL npz path
    rank: Optional[int] = None
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        return (f"alignment[{self.method}] {self.d_in}->{self.d_out} "
                f"n={self.n_train}/{self.n_val}/{self.n_heldout} "
                f"loss {self.train_loss:.4f}/{self.val_loss:.4f}/{self.heldout_loss:.4f} "
                f"heldout-cos {self.aligned_similarity:.3f} r2 {self.heldout_r2:.3f} "
                f"{'SATURATED: ' + self.saturation_note if self.saturated else ''}")


def _split(n: int, holdout_frac: float = 0.25, val_frac: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_ho = max(1, int(n * holdout_frac))
    n_va = max(1, int(n * val_frac))
    return idx[:-n_ho - n_va], idx[-n_ho - n_va:-n_ho], idx[-n_ho:]


def _cos_r2(P: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    cos = float(np.mean([np.dot(P[i], Y[i]) /
                         (np.linalg.norm(P[i]) * np.linalg.norm(Y[i]) + 1e-9)
                         for i in range(len(Y))]))
    r2 = float(1 - ((P - Y) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9))
    return cos, r2


def fit_alignment(source_acts: np.ndarray, target_acts: np.ndarray,
                  method: str, seed: int = 0, rank: Optional[int] = None,
                  ridge: float = 1e-2, learned_steps: int = 400) -> Tuple[Dict, AlignmentResult]:
    """Fit source->target mapping on paired activations.
    Returns (transform dict, AlignmentResult). transform keys:
    {method, W, b, mean_s, std_s, mean_t, std_t}"""
    if method not in ALIGNMENT_METHODS:
        raise ValueError(f"unknown alignment method {method}")
    X, Y = np.asarray(source_acts, np.float64), np.asarray(target_acts, np.float64)
    if len(X) != len(Y):
        raise DimensionMismatchError("paired activation count", (len(X),), (len(Y),),
                                     "activations must be collected on matched inputs")
    d_in, d_out = X.shape[1], Y.shape[1]
    if method == "identity" and d_in != d_out:
        raise DimensionMismatchError(
            "identity alignment", f"d_in={d_in}", f"d_out={d_out}",
            "identity requires equal dimensions; use linear/low_rank/learned")

    tr, va, ho = _split(len(X), seed=seed)
    Xtr, Ytr = X[tr], Y[tr]
    Xva, Yva = X[va], Y[va]
    Xho, Yho = X[ho], Y[ho]

    transform: Dict = {"method": method, "d_in": int(d_in), "d_out": int(d_out), "seed": seed}
    mean_s = Xtr.mean(0); std_s = Xtr.std(0) + 1e-6
    mean_t = Ytr.mean(0); std_t = Ytr.std(0) + 1e-6

    def apply_T(Wb, Xa):
        W, b = Wb
        return Xa @ W + b

    if method == "identity":
        W = np.eye(d_in); b = np.zeros(d_in)
        Wb = (W, b)
        transform.update({"W": W, "b": b})
    elif method in ("linear", "low_rank"):
        Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
        A = Xb.T @ Xb + ridge * np.eye(d_in + 1)
        sol = np.linalg.solve(A, Xb.T @ Ytr)
        W, b = sol[:-1], sol[-1]
        if method == "low_rank":
            r = rank or max(2, min(d_in, d_out) // 4)
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            k = max(1, min(r, len(S)))
            W = (U[:, :k] * S[:k]) @ Vt[:k]
            transform["rank"] = int(k)
        Wb = (W, b)
        transform.update({"W": W, "b": b})
    elif method == "learned":
        try:
            import torch
            Wt = torch.zeros(d_in, d_out, requires_grad=True)
            torch.manual_seed(seed)
            with torch.no_grad():
                Wt += torch.randn(d_in, d_out) * 0.02
            bt = torch.zeros(d_out, requires_grad=True)
            opt = torch.optim.Adam([Wt, bt], lr=5e-3)
            Xs = torch.tensor((Xtr - mean_s) / std_s, dtype=torch.float32)
            Ys = torch.tensor(Ytr, dtype=torch.float32)
            for _ in range(learned_steps):
                opt.zero_grad()
                loss = ((Xs @ Wt + bt - Ys) ** 2).mean()
                loss.backward(); opt.step()
            W = Wt.detach().numpy() * (1.0 / std_s)[:, None]
            b = bt.detach().numpy() - (mean_s / std_s) @ W
            Wb = (W, b)
            transform.update({"W": W, "b": b})
        except ImportError:
            # fall back to ridge if torch unavailable (recorded honestly)
            Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
            sol = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(d_in + 1), Xb.T @ Ytr)
            W, b = sol[:-1], sol[-1]
            Wb = (W, b)
            transform.update({"W": W, "b": b, "note": "torch unavailable: ridge fallback"})
    elif method == "activation_stats":
        # pure normalization matching: standardized source -> target stats
        Wb = (np.diag(std_t / (std_s.mean() * np.ones(d_in) if d_in != d_out
                              else std_s)) @ np.eye(d_in) if d_in == d_out
              else np.zeros((d_out, d_in)))
        if d_in != d_out:
            raise DimensionMismatchError(
                "activation_stats alignment", f"d_in={d_in}", f"d_out={d_out}",
                "statistics matching preserves dimensions; pair it with linear "
                "for mismatched dims")
        transform.update({"mean_s": mean_s, "std_s": std_s,
                          "mean_t": mean_t, "std_t": std_t})
        # define apply via standardization
        def T(Xa):
            Z = (Xa - mean_s) / std_s
            return Z * std_t + mean_t
        Pva, Pho = T(Xva), T(Xho)
        Ptr = T(Xtr)
        base_cos = None
        cos_ho, r2_ho = _cos_r2(Pho, Yho)
        sat, note = _saturation(len(Xtr), max(d_in, d_out))
        aid = "align-" + uuid.uuid4().hex[:8]
        res = AlignmentResult(aid, method, int(d_in), int(d_out), len(tr), len(va), len(ho),
                              float(((Ptr - Ytr) ** 2).mean()), float(((Pva - Yva) ** 2).mean()),
                              float(((Pho - Yho) ** 2).mean()), base_cos, cos_ho, r2_ho,
                              sat, note, seed)
        _save_checkpoint(aid, transform, res)
        return transform, res

    Ptr, Pva, Pho = apply_T(Wb, Xtr), apply_T(Wb, Xva), apply_T(Wb, Xho)
    base_cos = None
    if d_in == d_out:
        base_cos = float(np.mean([np.dot(Xho[i], Yho[i]) /
                                  (np.linalg.norm(Xho[i]) * np.linalg.norm(Yho[i]) + 1e-9)
                                  for i in range(len(Yho))]))
    cos_ho, r2_ho = _cos_r2(Pho, Yho)
    sat, note = _saturation(len(Xtr), max(d_in, d_out))
    aid = "align-" + uuid.uuid4().hex[:8]
    res = AlignmentResult(aid, method, int(d_in), int(d_out), len(tr), len(va), len(ho),
                          float(((Ptr - Ytr) ** 2).mean()), float(((Pva - Yva) ** 2).mean()),
                          float(((Pho - Yho) ** 2).mean()), base_cos, cos_ho, r2_ho,
                          sat, note, seed, rank=transform.get("rank"))
    _save_checkpoint(aid, transform, res)
    return transform, res


def _saturation(n_train: int, d: int) -> Tuple[bool, str]:
    if n_train < d:
        return True, f"n_train={n_train} < dimension d={d}: fit is underdetermined"
    if n_train < 10 * d:
        return True, f"n_train={n_train} < 10*d={10*d}: treat alignment metrics cautiously"
    return False, ""


def _save_checkpoint(aid: str, transform: Dict, res: AlignmentResult):
    import numpy as np
    d = os.environ.get("FUSIONLAB_DATA") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "fusionlab_data")
    os.makedirs(os.path.join(d, "alignments"), exist_ok=True)
    rel = os.path.join("alignments", aid + ".npz")
    np.savez_compressed(os.path.join(d, rel), **{k: np.asarray(v)
                        for k, v in transform.items() if isinstance(v, np.ndarray)})
    res.checkpoint = rel
    json.dump(res.to_dict(), open(os.path.join(d, "alignments", aid + ".json"), "w"),
              indent=1)


def apply_alignment(transform: Dict, X: np.ndarray) -> np.ndarray:
    """Apply a fitted transform. Never reshapes silently: validates dims."""
    W = transform.get("W")
    Xa = np.asarray(X, np.float64)
    if W is not None:
        if Xa.shape[1] != W.shape[0]:
            raise DimensionMismatchError("apply_alignment", Xa.shape, W.shape,
                                         "activation dim must match fitted d_in")
        b = transform.get("b", np.zeros(W.shape[1]))
        return Xa @ W + b
    if transform["method"] == "activation_stats":
        Z = (Xa - transform["mean_s"]) / transform["std_s"]
        return Z * transform["std_t"] + transform["mean_t"]
    if transform["method"] == "identity":
        return Xa
    raise DimensionMismatchError("apply_alignment", Xa.shape, "unknown transform",
                                 "transform dict has neither W nor stats")
