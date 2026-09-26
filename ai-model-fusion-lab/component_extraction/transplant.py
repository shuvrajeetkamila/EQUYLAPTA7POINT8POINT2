"""Component extraction + 'cut and copy' transplant engine (Phases 5, 7, 8).

A Component is a PORTABLE DESCRIPTION of an experimentally-implicated internal
region. Components are marked SELF_CONTAINED (weights fully describe the unit)
or DEPENDENT_ON_PARENT_REPRESENTATION (needs the surrounding residual stream).

Transplants implement the A/B/C positive-intervention protocol (Phase 5):
  A = donor score, B = recipient baseline, C = recipient after insertion.
Success requires C > B (toward A) — recorded either way.
Granularity for micro models: LAYER, MODULE, HEAD, CHANNEL_GROUP.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from benchmarking.harness import eval_suite
from benchmarking.suites import load_suite
from capability_genome.genome import ComponentRef


# --------------------------------------------------------------------------
@dataclass
class Component:
    source_model: str
    architecture: str
    target_capability: str
    region: str                  # ComponentRef.path
    level: str
    evidence_delta: float        # ablation delta that nominated this component
    confidence: str
    dependencies: str            # e.g. "layers 0-1 residual stream"
    tokenizer: str
    hidden_dimension: int
    containment: str = "DEPENDENT_ON_PARENT_REPRESENTATION"
    tensors: Dict[str, list] = field(default_factory=dict)  # name -> shape spec
    notes: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_genome_record(cls, rec, model, capability: str) -> "Component":
        c = rec.component
        return cls(
            source_model=model.name, architecture=model.spec.arch,
            target_capability=capability, region=c["path"], level=c["level"],
            evidence_delta=rec.delta, confidence=rec.confidence,
            dependencies=f"residual stream up to layer {c['layer']}",
            tokenizer=f"shared-micro-vocab-{model.spec.vocab}",
            hidden_dimension=model.spec.hidden,
            containment=("SELF_CONTAINED" if c["level"] in ("LAYER", "MODULE")
                         else "DEPENDENT_ON_PARENT_REPRESENTATION"),
            notes=f"nominated by ablation on {rec.suite} (n={rec.n})")

    def describe(self) -> str:
        return (f"COMPONENT [{self.level}] {self.region}\n"
                f"  source_model:      {self.source_model}\n"
                f"  architecture:      {self.architecture}\n"
                f"  target_capability: {self.target_capability}\n"
                f"  evidence:          ablation delta {self.evidence_delta:+.1f} pts\n"
                f"  confidence:        {self.confidence}\n"
                f"  dependencies:      {self.dependencies}\n"
                f"  containment:       {self.containment}")


# --------------------------------------------------------------------------
# tensor slicing helpers for the micro-gpt layout
def _head_slices(d: int, H: int, h: int):
    e = d // H
    return {"qkv.W.q": (slice(None), slice(h * e, (h + 1) * e)),
            "qkv.W.k": (slice(None), slice(d + h * e, d + (h + 1) * e)),
            "qkv.W.v": (slice(None), slice(2 * d + h * e, 2 * d + (h + 1) * e)),
            "qkv.b.q": (slice(h * e, (h + 1) * e),),
            "qkv.b.k": (slice(d + h * e, d + (h + 1) * e),),
            "qkv.b.v": (slice(2 * d + h * e, 2 * d + (h + 1) * e),),
            "o.W":     (slice(h * e, (h + 1) * e), slice(None))}


def transplant_component(dst, src, comp: Component) -> Dict[str, bool]:
    """Copy the component's tensors from src into dst (same arch required).
    Returns which tensors were copied. Raises on incompatible shapes (never
    fakes compatibility — spec §26)."""
    if dst.spec.arch != src.spec.arch or dst.spec.hidden != src.spec.hidden \
            or dst.spec.layers != src.spec.layers:
        raise ValueError(f"incompatible architectures: {src.spec.arch}(d{src.spec.hidden},L{src.spec.layers}) "
                         f"-> {dst.spec.arch}(d{dst.spec.hidden},L{dst.spec.layers}); "
                         "use ALIGNMENT or DISTILLATION instead")
    S, D = src.state_dict(), dst.state_dict()
    ref = ComponentRef(**comp.region and _parse_region(comp.region))
    l = ref.layer
    copied = {}
    if ref.level == "LAYER":
        for k in S:
            if k.startswith(f"L{l}."):
                D[k] = S[k].copy(); copied[k] = True
    elif ref.level == "MODULE":
        for k in S:
            if k.startswith(f"L{l}.") and f".{ref.module}" in k:
                D[k] = S[k].copy(); copied[k] = True
    elif ref.level == "HEAD":
        d, H = dst.spec.hidden, dst.spec.heads
        e = d // H
        h0 = ref.index * e
        W, b = f"L{l}.attn.qkv.W", f"L{l}.attn.qkv.b"
        for off in (0, d, 2 * d):                    # q, k, v column blocks
            cols = slice(off + h0, off + h0 + e)
            D[W][:, cols] = S[W][:, cols]
            D[b][cols] = S[b][cols]
        rows = slice(h0, h0 + e)                     # o.W rows for this head
        D[f"L{l}.attn.o.W"][rows, :] = S[f"L{l}.attn.o.W"][rows, :]
        copied = {"qkv.cols": True, "o.rows": True}
    elif ref.level == "CHANNEL_GROUP":
        d0, d1 = ref.dims["lo"], ref.dims["hi"]
        D[f"L{l}.mlp.1.W"][:, d0:d1] = S[f"L{l}.mlp.1.W"][:, d0:d1]
        D[f"L{l}.mlp.1.b"][d0:d1] = S[f"L{l}.mlp.1.b"][d0:d1]
        D[f"L{l}.mlp.2.W"][d0:d1, :] = S[f"L{l}.mlp.2.W"][d0:d1, :]
        copied["mlp_group"] = True
    else:
        raise ValueError(f"transplant of level {ref.level} not supported")
    dst.load_state_dict(D)
    return copied


def _parse_region(path: str) -> dict:
    """'L1.attn.head3' -> level/layer/module/index; 'L0.mlp.chan[16:24]' -> dims."""
    import re
    layer = int(re.match(r"L(\d+)", path).group(1))
    if ".head" in path:
        h = int(path.split(".head")[-1])
        return {"level": "HEAD", "layer": layer, "module": "attn", "index": h,
                "dims": {"head_dim": None}, "path": path}
    if ".chan[" in path:
        lo, hi = map(int, path.split(".chan[")[-1].rstrip("]").split(":"))
        return {"level": "CHANNEL_GROUP", "layer": layer, "module": "mlp",
                "index": lo, "dims": {"lo": lo, "hi": hi}, "path": path}
    if ".attn" in path or ".mlp" in path:
        return {"level": "MODULE", "layer": layer,
                "module": "attn" if ".attn" in path else "mlp",
                "index": None, "dims": None, "path": path}
    return {"level": "LAYER", "layer": layer, "module": None, "index": None,
            "dims": None, "path": path}


# --------------------------------------------------------------------------
def abc_intervention(donor, recipient, comp: Component, suite_name: str,
                     secondary_suites: List[str] = (), max_items: int = 40,
                     seed: int = 5) -> dict:
    """Phase 5 protocol. Recipient weights are RESTORED after the experiment
    (the caller may keep them by reading out['state'] first — no: we snapshot
    and restore; the caller re-applies via keep=True variant if needed)."""
    snap = {k: v.copy() for k, v in recipient.state_dict().items()}
    A = eval_suite(donor, load_suite(suite_name), max_items=max_items)["accuracy"] * 100
    B = eval_suite(recipient, load_suite(suite_name), max_items=max_items)["accuracy"] * 100
    t0 = time.time()
    copied = transplant_component(recipient, donor, comp)
    C = eval_suite(recipient, load_suite(suite_name), max_items=max_items)["accuracy"] * 100
    sec = {}
    for sn in secondary_suites:
        sec[sn] = eval_suite(recipient, load_suite(sn), max_items=max_items)["accuracy"] * 100
    recipient.load_state_dict(snap)  # restore
    B_sec = {sn: eval_suite(recipient, load_suite(sn), max_items=max_items)["accuracy"] * 100
             for sn in secondary_suites}
    return {"A_donor": round(A, 1), "B_recipient": round(B, 1), "C_transplanted": round(C, 1),
            "gain": round(C - B, 1), "toward_donor": round((C - B) / max(A - B, 1e-9), 2) if A > B else None,
            "secondary_before": B_sec, "secondary_after": sec,
            "copied_tensors": sorted(copied), "seconds": round(time.time() - t0, 1)}


def extract_pool(genomes: Dict[str, "CapabilityGenome"], models: Dict[str, object],
                 capability: str, suite: str, top_k: int = 3,
                 min_abs_delta: float = 4.0) -> List[Component]:
    """Phase 8: build the capability component pool from multiple genomes."""
    pool: List[Component] = []
    for name, g in genomes.items():
        for rec in g.candidates(capability=capability, min_abs_delta=min_abs_delta):
            if rec.suite != suite:
                continue
            comp = Component.from_genome_record(rec, models[name], capability)
            pool.append(comp)
    # dedupe by (source, region), keep strongest evidence
    seen = {}
    for c in pool:
        k = (c.source_model, c.region)
        if k not in seen or abs(c.evidence_delta) > abs(seen[k].evidence_delta):
            seen[k] = c
    pool = sorted(seen.values(), key=lambda c: -abs(c.evidence_delta))
    return pool[:top_k]


def candidate_combinations(pool: List[Component], max_size: int = 2,
                           max_candidates: int = 6) -> List[List[Component]]:
    """Phase 8: generate candidate combinations (do NOT combine everything)."""
    from itertools import combinations
    out = []
    for r in range(1, max_size + 1):
        for combo in combinations(pool, r):
            # same-source combos are trivially the source model; keep mixed first
            out.append(list(combo))
    out.sort(key=lambda c: (len({x.source_model for x in c}) == 1, -sum(abs(x.evidence_delta) for x in c)))
    return out[:max_candidates]
