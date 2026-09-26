"""Component refitting (M3, spec §8).

METHOD A direct insertion      -> apply.insert_component_tensors (no training)
METHOD B projected insertion   -> apply.insert_component_tensors(cross_dim_transform)
METHOD C low-rank adapter      -> SVD-truncated insertion (module params recorded)
METHOD D activation adapter    -> activation-space alignment before insertion
METHOD E small learned refit   -> train ONLY the inserted slices (masked
                                  gradients) on the train split; everything
                                  else frozen. Trainable params counted
                                  EXACTLY as the slice numel (not the
                                  enclosing tensor).

Typical inserted module: 1-6k params vs a 111k-param target (recorded).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .alignment import DimensionMismatchError


@dataclass
class RefitResult:
    method: str
    trainable_params: int
    steps: int
    lr: float
    seed: int
    initial_loss: float
    final_loss: float
    loss_curve: List[float] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""

    def to_dict(self):
        return asdict(self)


def lowrank_factorize(tensors: Dict[str, np.ndarray], rank: int, seed: int = 0) -> tuple:
    """SVD-truncate 2-D component tensors to rank r (METHOD C).
    Returns (tensors_lowrank, module_params, energy_kept)."""
    out: Dict[str, np.ndarray] = {}
    params = 0
    energies = []
    for k, v in tensors.items():
        if v.ndim == 2 and min(v.shape) > rank:
            U, S, Vt = np.linalg.svd(v.astype(np.float64), full_matrices=False)
            r = min(rank, len(S))
            out[k] = ((U[:, :r] * S[:r]) @ Vt[:r]).astype(np.float32)
            params += int(v.shape[0] * r + r * v.shape[1])
            energies.append(float((S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-9)))
        else:
            out[k] = v.copy()
            params += int(v.size)
    return out, params, float(np.mean(energies)) if energies else 1.0


def build_masks(target, locations: Dict[str, tuple]) -> Tuple[Dict[str, any], int]:
    """locations: {name: (tensor_name, slice, role)} from apply.tensor_locations.
    Returns ({full_tensor_name: 0/1 mask}, total_masked_params)."""
    if target.backend == "torch-synthetic":
        import torch
        masks: Dict[str, "torch.Tensor"] = {}
        for name, (tname, sl, _role) in locations.items():
            if tname not in target.t:
                raise KeyError(f"target lacks tensor {tname} for component part {name}")
            m = masks.setdefault(tname, torch.zeros_like(target.t[tname]))
            m[sl] = 1.0
        total = int(sum(float(m.sum()) for m in masks.values()))
        return masks, total
    else:
        masks: Dict[str, np.ndarray] = {}
        for name, (tname, sl, _role) in locations.items():
            if tname not in target.params:
                raise KeyError(f"target lacks tensor {tname} for component part {name}")
            m = masks.setdefault(tname, np.zeros_like(target.params[tname]))
            m[sl] = 1.0
        total = int(sum(float(m.sum()) for m in masks.values()))
        return masks, total


def refit_component_only(target, locations: Dict[str, tuple],
                         train_seqs: List[np.ndarray], steps: int = 150,
                         lr: float = 2e-3, batch: int = 64, seed: int = 0,
                         log=None) -> RefitResult:
    """METHOD E: masked-gradient training of ONLY the inserted slices.
    Freezes everything else; counts exactly the slice params."""
    if target.backend != "torch-synthetic":
        masks, trainable_params = build_masks(target, locations)
        rng = np.random.default_rng(seed)
        curve: List[float] = []
        initial = None
        t0 = time.time()
        for st in range(steps):
            idx = rng.choice(len(train_seqs), size=min(batch, len(train_seqs)), replace=False)
            L = max(len(train_seqs[i]) for i in idx)
            arr = np.zeros((len(idx), L), np.int64)
            for r, i in enumerate(idx):
                arr[r, :len(train_seqs[i])] = train_seqs[i]
            loss, grads = target.loss_and_grads(arr)
            masked_grads = {}
            for tname, m in masks.items():
                if tname in grads:
                    masked_grads[tname] = (grads[tname] * m).astype(np.float32)
            target.adam_step(masked_grads, lr=lr)
            v = float(loss)
            if initial is None:
                initial = v
            curve.append(v)
            if log and st % 30 == 0:
                log(f"      refit step {st}/{steps} loss={v:.3f}")
        return RefitResult("METHOD E (numpy)", trainable_params, steps, lr, seed,
                           float(initial or 0.0), float(curve[-1] if curve else 0.0),
                           curve, time.time() - t0, "numpy synthetic refit")

    import torch
    masks, trainable_params = build_masks(target, locations)
    frozen = [k for k in target.t if k not in masks]
    saved_rg = {k: bool(target.t[k].requires_grad) for k in target.t}
    for k in frozen:
        target.t[k].requires_grad_(False)
    for k in masks:
        target.t[k].requires_grad_(True)

    opt = torch.optim.Adam([target.t[k] for k in masks], lr=lr)
    rng = np.random.default_rng(seed)
    curve: List[float] = []
    initial = None
    t0 = time.time()
    for st in range(steps):
        idx = rng.choice(len(train_seqs), size=min(batch, len(train_seqs)),
                         replace=False)
        L = max(len(train_seqs[i]) for i in idx)
        arr = np.zeros((len(idx), L), np.int64)
        for r, i in enumerate(idx):
            arr[r, :len(train_seqs[i])] = train_seqs[i]
        x = torch.tensor(arr)
        logits = target.forward_torch(x)
        lg = logits[:, :-1, :]
        sh = x[:, 1:]
        logP = torch.log_softmax(lg.reshape(-1, lg.shape[-1]), -1)
        keep = (sh.reshape(-1) != 0)
        tgt = sh.reshape(-1).clamp(min=0)
        loss = -logP[torch.arange(len(tgt)), tgt][keep].mean()
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for k, m in masks.items():
                if target.t[k].grad is not None:
                    target.t[k].grad *= m          # masked gradients only
        torch.nn.utils.clip_grad_norm_([target.t[k] for k in masks], 1.0)
        opt.step()
        v = float(loss.detach())
        if initial is None:
            initial = v
        curve.append(v)
        if log and st % 30 == 0:
            log(f"      refit step {st}/{steps} loss={v:.3f}")
    for k, rg in saved_rg.items():
        target.t[k].requires_grad_(rg)
    return RefitResult("component_only", trainable_params, steps, lr, seed,
                       float(initial), float(curve[-1]),
                       [round(float(c), 4) for c in curve[:: max(1, steps // 10)]],
                       round(time.time() - t0, 1),
                       note="masked-gradient training: only inserted slices trainable")
