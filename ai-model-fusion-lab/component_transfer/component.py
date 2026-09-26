"""Formal component representation (M3, spec §3) + component library (§24).

A Component is NOT "a layer" — it is an experimentally-evidenced internal unit
with full provenance and typed location. Tensor payloads are stored OUTSIDE
the project tree (fusionlab_data/components/) with references in metadata,
keeping the release small (spec §2).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# typed component taxonomy (§3)
COMPONENT_TYPES = {
    "LAYER": "full transformer block",
    "MODULE_ATTN": "attention module of one layer",
    "MODULE_MLP": "MLP module of one layer",
    "HEAD": "single attention head (q,k,v,o slices)",
    "HEAD_GROUP": "group of attention heads",
    "Q_PROJ": "query projection slice for given heads",
    "K_PROJ": "key projection slice",
    "V_PROJ": "value projection slice",
    "O_PROJ": "output projection rows for given heads",
    "MLP_UP": "MLP input projection (selected channels)",
    "MLP_DOWN": "MLP output projection (selected channels)",
    "CHANNEL_GROUP": "group of MLP neurons (up+down slices)",
    "RESIDUAL_PATH": "residual-stream intervention point",
    "ACTIVATION_SUBSPACE": "activation-space direction set (no weights)",
    "LOWRANK_SUBSPACE": "low-rank parameter subspace of a tensor",
}

TRANSFER_STATUS = ["UNKNOWN", "UNTESTED", "FAILED", "PARTIAL", "PROMISING",
                   "REPRODUCED"]  # 'PROVEN' deliberately absent (§36)


def data_dir() -> str:
    d = os.environ.get("FUSIONLAB_DATA")
    if not d:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        d = os.path.join(root, "fusionlab_data")
    os.makedirs(os.path.join(d, "components"), exist_ok=True)
    return d


@dataclass
class ComponentRecord:
    component_id: str
    source_model: str
    source_model_revision: str
    architecture: str
    component_type: str                 # one of COMPONENT_TYPES
    layer: int
    head_ids: List[int] = field(default_factory=list)
    channel_range: Optional[List[int]] = None       # [lo, hi)
    parameter_shapes: Dict[str, List[int]] = field(default_factory=dict)
    hidden_dimension: int = 0
    input_dimension: int = 0
    output_dimension: int = 0
    tokenizer: str = ""
    capability_target: str = ""
    # --- evidence (all measured; zeros mean "not measured") ---
    baseline_score: Optional[float] = None
    ablation_score: Optional[float] = None
    restoration_score: Optional[float] = None
    positive_intervention_score: Optional[float] = None
    transfer_score: Optional[float] = None
    random_control_result: Optional[float] = None
    confidence: str = "low"
    transfer_status: str = "UNTESTED"               # one of TRANSFER_STATUS
    dependencies: str = ""
    normalization_requirements: str = ""
    alignment_method: str = "none"
    refit_method: str = "none"
    calibration_method: str = "none"
    tensors_ref: str = ""                            # EXTERNAL npz path
    suite: str = ""
    n_examples: int = 0
    seed: int = 0
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    notes: str = ""

    # ------------------------------------------------------------------
    def validate(self):
        if self.component_type not in COMPONENT_TYPES:
            raise ValueError(f"unknown component_type {self.component_type}; "
                             f"known: {sorted(COMPONENT_TYPES)}")
        if self.transfer_status not in TRANSFER_STATUS:
            raise ValueError(f"transfer_status must be one of {TRANSFER_STATUS}")
        if self.baseline_score is None:
            raise ValueError("component without baseline evidence — run Test A first")

    def param_count(self) -> int:
        return int(sum(int(__import__("numpy").prod(s)) for s in self.parameter_shapes.values()))

    def tensors_abs_path(self) -> str:
        return self.tensors_ref if os.path.isabs(self.tensors_ref) else \
            os.path.join(data_dir(), self.tensors_ref)

    def save_tensors(self, tensors: Dict[str, "np.ndarray"]):
        import numpy as np
        rel = os.path.join("components", self.component_id + ".npz")
        path = os.path.join(data_dir(), rel)
        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32)
                                     for k, v in tensors.items()})
        self.tensors_ref = rel
        self.parameter_shapes = {k: list(v.shape) for k, v in tensors.items()}

    def load_tensors(self) -> Dict[str, "np.ndarray"]:
        import numpy as np
        z = np.load(self.tensors_abs_path())
        return {k: z[k] for k in z.files}

    def to_dict(self) -> dict:
        return asdict(self)

    def save_meta(self, library_dir: Optional[str] = None):
        self.validate()
        lib = library_dir or os.path.join(data_dir(), "library", self.capability_target or "misc")
        os.makedirs(lib, exist_ok=True)
        path = os.path.join(lib, self.component_id + ".json")
        json.dump(self.to_dict(), open(path, "w"), indent=1)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path_or_id: str) -> "ComponentRecord":
        if path_or_id.endswith(".json"):
            return cls.from_dict(json.load(open(path_or_id)))
        # search library
        root = os.path.join(data_dir(), "library")
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f == path_or_id + ".json":
                    return cls.from_dict(json.load(open(os.path.join(dirpath, f))))
        raise FileNotFoundError(f"component {path_or_id} not in library")

    def describe(self) -> str:
        L = [f"COMPONENT {self.component_id} [{self.component_type}] "
             f"capability={self.capability_target} status={self.transfer_status}"]
        L.append(f"  source: {self.source_model} ({self.architecture}, rev {self.source_model_revision})")
        L.append(f"  location: layer {self.layer}"
                 + (f" heads {self.head_ids}" if self.head_ids else "")
                 + (f" channels {self.channel_range}" if self.channel_range else ""))
        L.append(f"  params: {self.param_count():,} | dims in/out: {self.input_dimension}/{self.output_dimension}")
        L.append(f"  evidence: baseline={self.baseline_score} ablated={self.ablation_score} "
                 f"restored={self.restoration_score} transfer={self.transfer_score} "
                 f"random_ctrl={self.random_control_result}")
        L.append(f"  transforms: align={self.alignment_method} refit={self.refit_method} "
                 f"calib={self.calibration_method}")
        return "\n".join(L)


def new_component_id(capability: str, ctype: str) -> str:
    return f"{capability}_{ctype.lower()}_{uuid.uuid4().hex[:6]}"


def index_library() -> Dict[str, List[dict]]:
    root = os.path.join(data_dir(), "library")
    out: Dict[str, List[dict]] = {}
    if os.path.isdir(root):
        for dirpath, _d, files in os.walk(root):
            for f in sorted(files):
                if f.endswith(".json"):
                    d = json.load(open(os.path.join(dirpath, f)))
                    out.setdefault(d.get("capability_target", "misc"), []).append(d)
    return out
