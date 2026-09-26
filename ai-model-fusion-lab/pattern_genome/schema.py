"""Pattern Genome schema (Milestone 4, spec §1-2, §7, §13, §17).

100 patterns are HYPOTHESES, never facts (§1). Every claim must climb the
evidence ladder:

  0 NOT_TESTED            4 RESTORATION_EFFECT
  1 PATTERN_DETECTED      5 POSITIVE_INTERVENTION
  2 CORRELATED            6 CROSS_MODEL_TRANSFER
  3 ABLATION_EFFECT       7 REPRODUCED_CROSS_MODEL_TRANSFER

Statuses (§1): NOT_TESTED | UNSUPPORTED | CONTRADICTED | PROMISING | REPRODUCED.
Never "validated" below level 5; never decorative pattern matching (§22).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

FAMILIES = [
    "P1_SEQUENCE", "P2_RATIO_SCALE", "P3_GEOMETRY", "P4_GRAPH_NETWORK",
    "P5_FEEDBACK", "P6_DISTRIBUTION", "P7_BIOLOGICAL", "P8_SIGNAL_CIRCUIT",
    "P9_TRANSFORMATION_PIPELINE", "P10_SELECTION_EVOLUTION",
]

EVIDENCE_LEVELS = {
    0: "NOT_TESTED",
    1: "PATTERN_DETECTED",
    2: "CORRELATED_WITH_CAPABILITY",
    3: "ABLATION_EFFECT",
    4: "RESTORATION_EFFECT",
    5: "POSITIVE_INTERVENTION",
    6: "CROSS_MODEL_TRANSFER",
    7: "REPRODUCED_CROSS_MODEL_TRANSFER",
}

STATUSES = ["NOT_TESTED", "UNSUPPORTED", "CONTRADICTED", "PROMISING", "REPRODUCED"]

PHASES = ["A", "B", "C", "D"]

# §13 progressive intervention levels -> SMALLEST CAUSALLY NECESSARY UNIT search
ABLATION_LEVELS = {
    1: "NODE", 2: "COMPONENT", 3: "COMPONENT_GROUP", 4: "PATH",
    5: "SUBGRAPH", 6: "ACTIVATION_SUBSPACE", 7: "ENTIRE_CIRCUIT",
}


@dataclass
class PatternRecord:
    """One of the 100 hypotheses (spec §2). Definition only — evidence lives
    in the registry, never inside the definition."""
    pattern_id: str                     # "P001".."P100"
    name: str
    family: str                         # one of FAMILIES
    description: str
    computational_hypothesis: str
    measurable_features: List[str] = field(default_factory=list)
    discovery_algorithm: str = ""       # detector primitive (§11 A-F)
    null_model: str = ""                # §12 — mandatory
    ablation_strategy: str = ""
    restoration_strategy: str = ""
    transfer_strategy: str = ""
    minimum_evidence: int = 3           # ladder level required to be PROMISING
    failure_conditions: List[str] = field(default_factory=list)
    parameter_efficiency: str = "MEDIUM"        # expectation: HIGH/MEDIUM/LOW
    architecture_dependence: str = "MEDIUM"     # expectation: LOW/MEDIUM/HIGH
    phase: str = "B"                    # staged search §21: A/B/C/D
    detector: str = ""                  # primitive key used by detectors.py

    def validate(self):
        import re
        if not re.match(r"^P\d{3}$", self.pattern_id):
            raise ValueError(f"bad pattern_id {self.pattern_id}")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family}")
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase {self.phase}")
        if not (0 <= int(self.minimum_evidence) <= 7):
            raise ValueError("minimum_evidence must be a ladder level 0-7")
        if not self.null_model:
            raise ValueError(f"{self.pattern_id}: every pattern needs a null model (§12)")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class MemberRef:
    """One member of a functional structure — a maskable model region."""
    level: str           # MODULE | HEAD | CHANNEL_GROUP | NODE (collection point)
    layer: int
    module: Optional[str] = None     # attn | mlp
    index: Optional[int] = None      # head index or channel-group index
    dims: Optional[dict] = None      # channel group {"lo","hi"}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def key(self) -> str:
        if self.level == "HEAD":
            return f"L{self.layer}.h{self.index}"
        if self.level == "MODULE":
            return f"L{self.layer}.{self.module}"
        if self.level == "CHANNEL_GROUP":
            return f"L{self.layer}.chan[{self.dims['lo']}:{self.dims['hi']}]"
        return f"L{self.layer}.{self.module or 'node'}"


@dataclass
class CircuitRecord:
    """A candidate functional structure (spec §3 output of motif detection,
    refined by the causal battery)."""
    circuit_id: str
    pattern_ids: List[str]              # which hypotheses nominated it
    model: str
    capability: str
    members: List[dict]                 # MemberRef.to_dict()
    kind: str = "circuit"               # chain|convergence|bottleneck|...|compound
    detection: Dict = field(default_factory=dict)       # stat, p, null stats
    task_correlation: Dict = field(default_factory=dict)
    ablation: Dict = field(default_factory=dict)        # per-level effects
    restoration: Dict = field(default_factory=dict)
    positive_intervention: Dict = field(default_factory=dict)
    interaction: Dict = field(default_factory=dict)
    minimal_unit: Optional[dict] = None                 # §14 result
    transfer: Dict = field(default_factory=dict)        # §15-16 raw components
    evidence_level: int = 0
    status: str = "NOT_TESTED"
    notes: str = ""
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def member_keys(self) -> List[str]:
        return [MemberRef.from_dict(m).key() for m in self.members]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def evidence_checklist(battery: Dict) -> Dict[str, bool]:
    """Which ladder rungs the battery cleared (kept raw, §17)."""
    det = battery.get("detection") or {}
    cor = battery.get("task_correlation") or {}
    abl = battery.get("ablation") or {}
    res = battery.get("restoration") or {}
    pos = battery.get("positive_intervention") or {}
    tra = battery.get("transfer") or {}
    rep = battery.get("reproduction") or {}
    return {
        "detected_vs_null": bool(det.get("significant")),
        "correlated": bool(cor.get("significant")),
        "ablation_effect": bool(abl.get("significant")),
        "restoration_effect": bool(res.get("recovered_pct") is not None
                                   and res["recovered_pct"] >= 50.0),
        "positive_intervention": bool(pos.get("significant")),
        "cross_model_transfer": bool(tra.get("beats_random") and tra.get("beats_capacity")
                                     and (tra.get("positive") is True)),
        "reproduced": bool(rep.get("reproduced")),
    }


def evidence_level(battery: Dict) -> int:
    """Evidence ladder (§17). Levels are cumulative-ish but honestly gated:
    you cannot claim ablation effect without detection+correlation context,
    transfer without the endogenous battery, reproduction without transfer."""
    chk = evidence_checklist(battery)
    lvl = 0
    if chk["detected_vs_null"]:
        lvl = 1
    if lvl == 1 and chk["correlated"]:
        lvl = 2
    if lvl == 2 and chk["ablation_effect"]:
        lvl = 3
    if lvl == 3 and chk["restoration_effect"]:
        lvl = 4
    if lvl == 4 and chk["positive_intervention"]:
        lvl = 5
    if chk["cross_model_transfer"]:
        lvl = max(lvl, 6)
    if lvl >= 6 and chk["reproduced"]:
        lvl = 7
    return lvl


def status_from_evidence(level: int, battery: Dict) -> str:
    """§1 status vocabulary. CONTRADICTED = evidence points the WRONG way
    (ablating the structure IMPROVES the capability, or restoration hurts)."""
    abl = battery.get("ablation") or {}
    if abl.get("effect") is not None and abl.get("effect_direction") == "improves" \
            and abl.get("significant"):
        return "CONTRADICTED"
    if level == 0:
        return "NOT_TESTED"
    if level >= 7:
        return "REPRODUCED"
    if level >= 3:
        return "PROMISING"
    return "UNSUPPORTED"


def transfer_score_components(held_out_gain: Optional[float],
                              random_control_gain: Optional[float],
                              capacity_control_gain: Optional[float],
                              interference_penalty: float,
                              alignment_cost: float = 0.0,
                              parameter_cost: int = 0,
                              architecture_dependency: str = "LOW",
                              reproducibility: str = "UNTESTED") -> Dict:
    """§16 — every raw component kept; the composite is reported but never
    used alone in the scientific report."""
    comp = None
    if held_out_gain is not None:
        rc = random_control_gain or 0.0
        cc = capacity_control_gain or 0.0
        comp = round(held_out_gain - rc - cc - interference_penalty, 3)
    return {
        "held_out_gain": held_out_gain,
        "random_control_gain": random_control_gain,
        "capacity_control_gain": capacity_control_gain,
        "interference_penalty": round(interference_penalty, 3),
        "composite": comp,
        "alignment_cost": round(alignment_cost, 4),
        "parameter_cost": parameter_cost,
        "architecture_dependency": architecture_dependency,
        "reproducibility": reproducibility,
    }
