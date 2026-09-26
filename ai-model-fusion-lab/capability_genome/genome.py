"""Capability Genome (Milestone 2, Phase 2).

An experimentally MEASURED map of which internal components of a model
contribute to which capability. Every record carries full provenance:
suite, n, seed, intervention, baseline/intervened/restored scores, delta,
confidence, and the experiment-DB id. Nothing in a genome is assumed —
an empty genome is the honest starting state of any model.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

LEVELS = ["LAYER", "MODULE", "HEAD", "CHANNEL_GROUP", "PARAM_REGION"]

CAPABILITIES = ["reasoning", "coding", "math", "language", "knowledge",
                "multilingual", "agent", "vision"]


@dataclass
class ComponentRef:
    level: str            # one of LEVELS
    path: str             # e.g. "L1.attn.head3", "L0.mlp.chan[16:24]"
    layer: int
    module: Optional[str] = None      # attn | mlp | norm | other
    index: Optional[int] = None       # head or channel-group index
    dims: Optional[dict] = None       # e.g. {"head_dim": 16}

    def to_dict(self):
        return asdict(self)


@dataclass
class GenomeRecord:
    component: dict                 # ComponentRef.to_dict()
    capability: str
    suite: str
    intervention: str               # "ablate" | "restore_check" | "transplant_in"
    baseline: float
    intervened: float
    delta: float                    # baseline - intervened (points, %); >0 = component mattered
    restored: Optional[float] = None
    n: int = 0
    ci95: Optional[list] = None
    seed: int = 0
    confidence: str = "low"         # low | medium | high
    status: str = "CANDIDATE"       # CANDIDATE | CONFIRMED | NOT_SIGNIFICANT | TRANSFERRED | FAILED_TRANSFER
    experiment_id: str = ""
    notes: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class CapabilityGenome:
    model_name: str
    arch: str
    fingerprint: str
    revision: str = "main"
    tokenizer_hash: str = ""
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    param_count: int = 0
    records: List[GenomeRecord] = field(default_factory=list)
    baselines: Dict[str, float] = field(default_factory=dict)   # suite -> acc%
    random_baseline: Dict[str, float] = field(default_factory=dict)  # chance level
    parent_baseline: Dict[str, float] = field(default_factory=dict)  # weak control model

    # ------------------------------------------------------------------
    def add(self, rec: GenomeRecord):
        self.records.append(rec)

    def suite_baselines(self) -> Dict[str, float]:
        return dict(self.baselines)

    def candidates(self, capability: Optional[str] = None, min_abs_delta: float = 5.0,
                   levels: Optional[List[str]] = None) -> List[GenomeRecord]:
        out = []
        for r in self.records:
            if r.intervention != "ablate":
                continue
            if capability and r.capability != capability:
                continue
            if levels and r.component["level"] not in levels:
                continue
            if abs(r.delta) >= min_abs_delta:
                out.append(r)
        out.sort(key=lambda r: -abs(r.delta))
        return out

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name, "arch": self.arch,
            "fingerprint": self.fingerprint, "revision": self.revision,
            "tokenizer_hash": self.tokenizer_hash, "created": self.created,
            "param_count": self.param_count,
            "baselines": self.baselines,
            "random_baseline": self.random_baseline,
            "parent_baseline": self.parent_baseline,
            "n_records": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "CapabilityGenome":
        d = json.load(open(path))
        g = cls(**{k: v for k, v in d.items() if k not in ("records", "n_records")})
        g.records = [GenomeRecord(**r) for r in d["records"]]
        return g

    def summary(self) -> str:
        lines = [f"GENOME {self.model_name} ({self.arch}, {self.param_count:,} params, "
                 f"{len(self.records)} records)"]
        if self.baselines:
            lines.append("  baselines:   " + "  ".join(f"{k}:{v:.0f}%" for k, v in sorted(self.baselines.items())))
        if self.random_baseline:
            lines.append("  chance:      " + "  ".join(f"{k}:{v:.0f}%" for k, v in sorted(self.random_baseline.items())))
        caps = {}
        for r in self.records:
            if r.intervention == "ablate" and abs(r.delta) >= 5:
                caps.setdefault(r.capability, []).append(r)
        for cap, recs in sorted(caps.items()):
            lines.append(f"  [{cap}]")
            for r in recs[:6]:
                c = r.component
                lines.append(f"    {c['path']:<28} removing -> {r.delta:+.1f} pts "
                             f"(n={r.n}, restored={r.restored if r.restored is not None else '—'}, "
                             f"conf={r.confidence}, {r.status})")
        return "\n".join(lines)
