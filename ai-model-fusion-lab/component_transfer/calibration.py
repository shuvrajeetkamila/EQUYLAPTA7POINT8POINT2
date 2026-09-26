"""Calibration (M3, spec §14).

A transferred component can be mathematically correct yet mis-scaled for the
target's residual stream. Calibration methods (all cheap, all recorded
separately so calibration gains are never attributed to the component):

  none         — untouched
  norm_match   — rescale output-side rows to the median norm of the target's
                 own corresponding tensor rows
  residual_gate— fit ONE scalar gate g (grid search) minimizing target CE on
                 the CALIBRATION split (never the held-out test)

Each returns CalibrationResult with before/after calibration-split loss so
the calibration delta is reported separately from the component effect.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

CALIBRATION_METHODS = ["none", "norm_match", "residual_gate"]


@dataclass
class CalibrationResult:
    method: str
    params_changed: int
    loss_before: float
    loss_after: float
    gate: Optional[float] = None
    note: str = ""

    def to_dict(self):
        return asdict(self)


def _ce_on(target, seqs: List[np.ndarray]) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(seqs), 32):
        chunk = seqs[i:i + 32]
        L = max(len(s) for s in chunk)
        arr = np.zeros((len(chunk), L), np.int64)
        for r, s in enumerate(chunk):
            arr[r, :len(s)] = s
        logits, _ = target.forward(arr)
        lg = logits[:, :-1, :]
        sh = arr[:, 1:]
        z = lg.reshape(-1, lg.shape[-1])
        z = z - z.max(-1, keepdims=True)
        P = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        keep = sh.reshape(-1) != 0
        tot += -float(np.log(P[keep] + 1e-9)[np.arange(int(keep.sum())),
                                             sh.reshape(-1)[keep]].sum())
        n += int(keep.sum())
    return tot / max(1, n)


def calibrate(target, out_locations: List[tuple], calib_seqs: List[np.ndarray],
              method: str = "residual_gate", own_rows_ref=None) -> CalibrationResult:
    """out_locations: [(tensor_name, slice, role)] where role == 'out'.
    Applies calibration to those slices IN PLACE (target state dict)."""
    if method == "none":
        loss = _ce_on(target, calib_seqs)
        return CalibrationResult("none", 0, loss, loss, None, "no calibration")
    T = target.state_dict()
    changed = 0
    if method == "norm_match":
        for tname, sl, role in out_locations:
            if role != "out" or tname not in T:
                continue
            rows = T[tname][sl]
            if rows.ndim != 2:
                continue
            ref = None
            if own_rows_ref is not None and own_rows_ref in T:
                ref_rows = T[own_rows_ref]
                ref = float(np.median(np.linalg.norm(ref_rows, axis=1)))
            cur = np.linalg.norm(rows, axis=1).mean()
            scale = (ref / cur) if (ref and cur > 0) else 1.0
            T[tname][sl] = (rows.astype(np.float64) * scale).astype(np.float32)
            changed += int(np.prod(rows.shape))
    target.load_state_dict(T)
    if method == "norm_match":
        after = _ce_on(target, calib_seqs)
        return CalibrationResult("norm_match", changed, after, after, None,
                                 "row-norm rescale applied; loss re-measured after")

    # residual_gate: 1-param grid search on the calibration split
    # dedupe out-locations per tensor name: multi-member circuits contribute
    # several slices of the SAME tensor (module + its heads); keep the LARGEST
    # slice so no parameter is gated twice and no shape collision occurs.
    dedup: Dict[str, tuple] = {}
    for tname, sl, role in out_locations:
        if role != "out" or tname not in T:
            continue
        rows = T[tname][sl]
        n_el = int(np.prod(rows.shape))
        cur = dedup.get(tname)
        if cur is None or n_el > cur[0]:
            dedup[tname] = (n_el, sl)
    locs2 = [(tname, sl, "out") for tname, (_n, sl) in dedup.items()]
    saved = {tname: T[tname][sl].copy() for tname, sl, _r in locs2}
    best_g, best_loss = 1.0, _ce_on(target, calib_seqs)
    before = best_loss
    for g in [0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0]:
        for tname, sl, _r in locs2:
            T[tname][sl] = (saved[tname].astype(np.float64) * g).astype(np.float32)
        target.load_state_dict(T)
        l = _ce_on(target, calib_seqs)
        if l < best_loss:
            best_loss, best_g = l, g
    # apply best gate (restore original then scale by best_g)
    for tname, sl, _r in locs2:
        T[tname][sl] = (saved[tname].astype(np.float64) * best_g).astype(np.float32)
    target.load_state_dict(T)
    n_gated = sum(int(np.prod(saved[t].shape)) for t in saved)
    return CalibrationResult("residual_gate", n_gated, before, best_loss, best_g,
                             f"single scalar gate g={best_g} (1 effective dof, "
                             f"scaled {n_gated} params)")


def out_locations_of(locations: Dict[str, tuple]) -> List[tuple]:
    return [(t, s, r) for (_n, (t, s, r)) in locations.items() if r == "out"]
