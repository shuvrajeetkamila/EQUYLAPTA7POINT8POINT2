"""[WORKING] Model registry + architecture compatibility analysis.

Register models (synthetic or HF), inspect their facts WITHOUT downloading
weights when possible, and compute a merge-compatibility score. Never merges
models merely because their names look similar — only measured compat does.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .base import ArchSpec, ModelHandle


class Registry:
    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path
        self.models: Dict[str, dict] = {}
        if store_path and os.path.exists(store_path):
            self.models = json.load(open(store_path))

    def save(self):
        if self.store_path:
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            json.dump(self.models, open(self.store_path, "w"), indent=1)

    # ------------------------------------------------------------------
    def register(self, handle: ModelHandle, license_info: str = "Apache-2.0") -> dict:
        rec = {
            "name": handle.name,
            "backend": handle.backend,
            "spec": handle.spec.to_dict(),
            "param_count": handle.param_count(),
            "fingerprint": handle.fingerprint(),
            "license": license_info,
            "profile": getattr(handle, "profile", "unknown"),
        }
        self.models[handle.name] = rec
        self.save()
        return rec

    def register_hf_id(self, hf_id: str, revision: str = "main") -> dict:
        """Inspect a HuggingFace model's architecture WITHOUT downloading weights:
        fetch config.json only. [WORKING with internet; graceful offline degrade]"""
        from .hf_probe import probe_hf_config
        rec = probe_hf_config(hf_id, revision)
        self.models[hf_id] = rec
        self.save()
        return rec

    # ------------------------------------------------------------------
    def get(self, name: str) -> dict:
        return self.models[name]

    def names(self) -> List[str]:
        return sorted(self.models)


COMPAT_WEIGHTS = {"arch": 0.30, "hidden": 0.15, "layers": 0.15, "heads": 0.10,
                  "vocab": 0.10, "ctx": 0.05, "tokenizer": 0.10, "license": 0.05}


def compatibility(a: dict, b: dict) -> dict:
    """Measured compatibility between two registry records (0..1, per-factor)."""
    sa, sb = a["spec"], b["spec"]
    factors = {
        "arch": 1.0 if sa["arch"] == sb["arch"] else 0.0,
        "hidden": 1.0 if sa["hidden"] == sb["hidden"] else 0.0,
        "layers": 1.0 if sa["layers"] == sb["layers"] else 0.0,
        "heads": 1.0 if sa["heads"] == sb["heads"] else 0.0,
        "vocab_exact": 1.0 if sa["vocab"] == sb["vocab"] else 0.0,
        "ctx": 1.0 if sa["ctx_len"] == sb["ctx_len"] else 0.0,
        "positional": 1.0 if sa["positional"] == sb["positional"] else 0.0,
        "norm": 1.0 if sa["norm"] == sb["norm"] else 0.0,
        "attention": 1.0 if sa["attention"] == sb["attention"] else 0.0,
        "modality": 1.0 if sa["modality"] == sb["modality"] else 0.0,
        "moe": 1.0 if sa.get("moe") == sb.get("moe") else 0.0,
    }
    # merge feasibility: hard requirement — identical shapes everywhere
    shape_ok = all([factors["arch"], factors["hidden"], factors["layers"],
                    factors["heads"], factors["vocab_exact"], factors["ctx"],
                    factors["positional"], factors["norm"], factors["attention"],
                    factors["moe"]])
    if shape_ok:
        merge_compat = 1.0 if a.get("fingerprint") != b.get("fingerprint") else 0.9
        method = "WEIGHT MERGING [WORKING]"
    elif factors["arch"] == 0.0 and factors["modality"] == 1.0:
        # different arch, same modality -> distillation / routing possible
        merge_compat = 0.15
        method = "DISTILLATION [WORKING micro] or ROUTING [WORKING]"
    elif factors["modality"] == 0.0:
        merge_compat = 0.05
        method = "ADAPTER/PROJECTION [PARTIAL] or ROUTING [WORKING]"
    else:
        merge_compat = 0.10
        method = "DISTILLATION or ROUTING"
    return {"pair": f"{a['name']} + {b['name']}", "factors": factors,
            "shape_compatible": shape_ok, "merge_compatibility": round(merge_compat, 2),
            "recommended_fusion": method}


def compatibility_matrix(reg: Registry) -> List[dict]:
    names = reg.names()
    return [compatibility(reg.models[a], reg.models[b])
            for i, a in enumerate(names) for b in names[i + 1:]]
