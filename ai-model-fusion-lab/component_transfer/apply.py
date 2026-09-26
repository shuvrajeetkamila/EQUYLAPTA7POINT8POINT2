"""Component tensor extraction + insertion (M3).

Strict shape discipline: insertion validates target tensor shapes against the
component's expected shapes; mismatch WITHOUT a fitted transform raises
DimensionMismatchError (never silent reshape/truncate/pad). Cross-dimension
insertion is supported for MLP channel components via activation-fitted
projections (projected insertion, spec §8 METHOD B) and recorded as such.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .alignment import DimensionMismatchError
from .component import ComponentRecord


# ---------------------------------------------------------------- slicing
def _head_slice(spec, head_ids):
    d, H = spec.hidden, spec.heads
    e = d // H
    out = {}
    for h in head_ids:
        h0 = h * e
        for off, tag in ((0, "q"), (d, "k"), (2 * d, "v")):
            cols = slice(off + h0, off + h0 + e)
            out[f"qkv.W.{tag}"] = ("L{l}.attn.qkv.W", (slice(None), cols))
            out[f"qkv.b.{tag}"] = ("L{l}.attn.qkv.b", (cols,))
        out["o.W"] = ("L{l}.attn.o.W", (slice(h0, h0 + e), slice(None)))
    return out


def tensor_locations(comp: ComponentRecord, spec) -> Dict[str, tuple]:
    """component_type -> list of (tensor_name_template, slice, role)."""
    l = comp.layer
    d, H, F = spec.hidden, spec.heads, spec.mlp_hidden
    e = d // H
    locs: Dict[str, tuple] = {}
    t = comp.component_type
    if t == "HEAD" and comp.head_ids:
        h = comp.head_ids[0]
        h0 = h * e
        for off, tag in ((0, "q"), (d, "k"), (2 * d, "v")):
            cols = slice(off + h0, off + h0 + e)
            locs[f"qkv.W.{tag}"] = (f"L{l}.attn.qkv.W", (slice(None), cols), "io")
            locs[f"qkv.b.{tag}"] = (f"L{l}.attn.qkv.b", (cols,), "bias")
        locs["o.W"] = (f"L{l}.attn.o.W", (slice(h0, h0 + e), slice(None)), "out")
    elif t in ("Q_PROJ", "K_PROJ", "V_PROJ") and comp.head_ids:
        h = comp.head_ids[0]
        h0 = h * e
        off = {"Q_PROJ": 0, "K_PROJ": d, "V_PROJ": 2 * d}[t]
        cols = slice(off + h0, off + h0 + e)
        tag = t[0].lower()
        locs[f"qkv.W.{tag}"] = (f"L{l}.attn.qkv.W", (slice(None), cols), "io")
        locs[f"qkv.b.{tag}"] = (f"L{l}.attn.qkv.b", (cols,), "bias")
    elif t == "O_PROJ" and comp.head_ids:
        h = comp.head_ids[0]
        h0 = h * e
        locs["o.W"] = (f"L{l}.attn.o.W", (slice(h0, h0 + e), slice(None)), "out")
    elif t in ("CHANNEL_GROUP", "MLP_UP", "MLP_DOWN"):
        lo, hi = comp.channel_range
        if t in ("CHANNEL_GROUP", "MLP_UP"):
            locs["mlp.1.W"] = (f"L{l}.mlp.1.W", (slice(None), slice(lo, hi)), "io")
            locs["mlp.1.b"] = (f"L{l}.mlp.1.b", (slice(lo, hi),), "bias")
        if t in ("CHANNEL_GROUP", "MLP_DOWN"):
            locs["mlp.2.W"] = (f"L{l}.mlp.2.W", (slice(lo, hi), slice(None)), "out")
    elif t in ("MODULE_ATTN", "MODULE_MLP"):
        key = "attn" if t == "MODULE_ATTN" else "mlp"
        for k, v in _full_module_locs(l, key).items():
            locs[k] = v
    elif t == "LAYER":
        raise ValueError("LAYER transfer is handled by merging/segment transplant; "
                         "component engine targets sub-layer units")
    else:
        raise ValueError(f"unsupported component_type {t}")
    return locs


def _full_module_locs(l: int, key: str) -> Dict[str, tuple]:
    if key == "attn":
        return {"qkv.W.all": (f"L{l}.attn.qkv.W", (slice(None), slice(None)), "io"),
                "qkv.b.all": (f"L{l}.attn.qkv.b", (slice(None),), "bias"),
                "o.W.all": (f"L{l}.attn.o.W", (slice(None), slice(None)), "out"),
                "o.b.all": (f"L{l}.attn.o.b", (slice(None),), "bias")}
    return {"mlp.1.W.all": (f"L{l}.mlp.1.W", (slice(None), slice(None)), "io"),
            "mlp.1.b.all": (f"L{l}.mlp.1.b", (slice(None),), "bias"),
            "mlp.2.W.all": (f"L{l}.mlp.2.W", (slice(None), slice(None)), "out"),
            "mlp.2.b.all": (f"L{l}.mlp.2.b", (slice(None),), "bias")}


def extract_component_tensors(source, comp: ComponentRecord) -> Dict[str, np.ndarray]:
    locs = tensor_locations(comp, source.spec)
    S = source.state_dict()
    out = {}
    for name, (tname, sl, _role) in locs.items():
        if tname not in S:
            raise KeyError(f"source model lacks tensor {tname}")
        out[name] = S[tname][sl].copy()
    return out


def expected_shapes(comp: ComponentRecord, target_spec) -> Dict[str, tuple]:
    locs = tensor_locations(comp, target_spec)
    T = target_spec.state_dict() if hasattr(target_spec, "state_dict") else None
    # shapes from the TARGET model directly (we validate against it)
    return locs


def insert_component_tensors(target, comp: ComponentRecord, tensors: Dict[str, np.ndarray],
                             cross_dim_transform: Optional[Dict] = None,
                             out_scale: float = 1.0) -> Dict:
    """Insert component tensors into the target model IN PLACE.

    Same-dimension: exact slice copy after shape validation.
    Cross-dimension (MLP channel comps only): projected insertion using
    cross_dim_transform={'U_in': [d_t,d_s], 'V_out': [d_s,d_t]} fitted from
    activation pairs. Returns an insertion record."""
    T = target.state_dict()
    locs = tensor_locations(comp, target.spec)
    same_dim = target.spec.hidden == comp.hidden_dimension
    if not same_dim and cross_dim_transform is None:
        raise DimensionMismatchError(
            f"insertion of {comp.component_type} {comp.region_desc() if hasattr(comp,'region_desc') else comp.component_id}",
            f"source d={comp.hidden_dimension}", f"target d={target.spec.hidden}",
            "cross-dimension insertion requires a fitted projection "
            "(fit_alignment on paired activations first)")
    applied, skipped = [], []
    for name, (tname, sl, role) in locs.items():
        if tname not in T:
            skipped.append((name, f"target lacks {tname}"))
            continue
        tgt_shape = T[tname][sl].shape
        val = tensors.get(name)
        if val is None:
            skipped.append((name, "no tensor in payload"))
            continue
        if same_dim:
            if tuple(val.shape) != tuple(tgt_shape):
                raise DimensionMismatchError(
                    f"tensor {tname}{sl}", tuple(tgt_shape), tuple(val.shape),
                    "component payload does not match target slice; refit or "
                    "re-extract — no silent reshape")
            T[tname][sl] = val.copy()
        else:
            if comp.component_type not in ("CHANNEL_GROUP", "MLP_UP", "MLP_DOWN"):
                raise DimensionMismatchError(
                    f"cross-dim insertion of {comp.component_type}",
                    tuple(tgt_shape), tuple(val.shape),
                    "cross-dimension projected insertion is only implemented for "
                    "MLP channel components; use ALIGNMENT->DISTILLATION otherwise")
            U_in = cross_dim_transform.get("U_in")
            V_out = cross_dim_transform.get("V_out")
            if name == "mlp.1.W":
                if U_in is None:
                    raise DimensionMismatchError(tname, tgt_shape, tuple(val.shape),
                                                 "missing U_in projection")
                proj = U_in.astype(np.float64) @ val.astype(np.float64)
            elif name == "mlp.1.b":
                proj = val.astype(np.float64).copy()   # channel biases transfer as-is
            elif name == "mlp.2.W":
                if V_out is None:
                    raise DimensionMismatchError(tname, tgt_shape, tuple(val.shape),
                                                 "missing V_out projection")
                proj = val.astype(np.float64) @ V_out.astype(np.float64)
            else:
                raise DimensionMismatchError(tname, tgt_shape, tuple(val.shape),
                                             f"no cross-dim rule for {name}")
            if proj.shape != tuple(tgt_shape):
                raise DimensionMismatchError(tname, tuple(tgt_shape), proj.shape,
                                             "projected insertion produced wrong shape")
            T[tname][sl] = proj.astype(np.float32)
        if role == "out" and out_scale != 1.0:
            T[tname][sl] = (T[tname][sl].astype(np.float64) * out_scale).astype(np.float32)
        applied.append((name, tname))
    target.load_state_dict(T)
    return {"applied": applied, "skipped": skipped, "same_dim": same_dim,
            "out_scale": out_scale,
            "inserted_params": int(sum(int(np.prod(tensors[k].shape)) for k in tensors))}


def adapter_insertion(target, comp: ComponentRecord,
                      tensors: Dict[str, np.ndarray],
                      cross_dim_transform: Optional[Dict] = None,
                      gate: float = 1.0) -> Dict:
    """Task-arithmetic component transfer (§16 ADAPTER_TRANSFER mechanism).

    W'[slice] = (1-gate) * W[slice] + gate * src_projected[slice]

    gate=0 leaves the target untouched, gate=1 equals full replacement — the
    gate interpolates between the target's own module function and the
    specialist's. Requires shape-compatible sites (same-dim components, or
    MLP channel comps with a fitted projection, exactly like
    insert_component_tensors). NO silent reshape: shape errors raise.
    """
    if not (0.0 <= gate <= 1.0):
        raise ValueError(f"adapter gate must be in [0,1], got {gate}")
    T = target.state_dict()
    locs = tensor_locations(comp, target.spec)
    same_dim = target.spec.hidden == comp.hidden_dimension
    if not same_dim and cross_dim_transform is None:
        raise DimensionMismatchError(
            f"adapter insertion of {comp.component_type} {comp.component_id}",
            f"source d={comp.hidden_dimension}", f"target d={target.spec.hidden}",
            "cross-dimension adapter requires a fitted projection")
    applied, skipped = [], []
    for name, (tname, sl, role) in locs.items():
        if tname not in T:
            skipped.append((name, f"target lacks {tname}"))
            continue
        cur = T[tname][sl].astype(np.float64)
        val = tensors.get(name)
        if val is None:
            skipped.append((name, "no tensor in payload"))
            continue
        if same_dim:
            if tuple(val.shape) != tuple(cur.shape):
                raise DimensionMismatchError(
                    f"tensor {tname}{sl}", tuple(cur.shape), tuple(val.shape),
                    "component payload does not match target slice; no silent reshape")
            src = val.astype(np.float64)
        else:
            U_in = cross_dim_transform.get("U_in")
            V_out = cross_dim_transform.get("V_out")
            if name == "mlp.1.W":
                if U_in is None:
                    raise DimensionMismatchError(tname, tuple(cur.shape),
                                                 tuple(val.shape), "missing U_in")
                src = U_in.astype(np.float64) @ val.astype(np.float64)
            elif name == "mlp.1.b":
                src = val.astype(np.float64)     # channel biases transfer as-is
            elif name == "mlp.2.W":
                if V_out is None:
                    raise DimensionMismatchError(tname, tuple(cur.shape),
                                                 tuple(val.shape), "missing V_out")
                src = val.astype(np.float64) @ V_out.astype(np.float64)
            else:
                raise DimensionMismatchError(tname, tuple(cur.shape),
                                             tuple(val.shape),
                                             f"no cross-dim rule for {name}")
            if src.shape != tuple(cur.shape):
                raise DimensionMismatchError(tname, tuple(cur.shape), src.shape,
                                             "projected adapter tensor wrong shape")
        blended = (1.0 - gate) * cur + gate * src
        T[tname][sl] = blended.astype(np.float32)
        applied.append((name, tname))
    target.load_state_dict(T)
    return {"applied": applied, "skipped": skipped, "same_dim": same_dim,
            "gate": gate,
            "blended_params": int(sum(int(np.prod(tensors[k].shape))
                                      for k in tensors if k in tensors))}


def restore_snapshot(target, snapshot: Dict[str, np.ndarray]):
    target.load_state_dict({k: v.copy() for k, v in snapshot.items()})


def snapshot(target) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in target.state_dict().items()}
