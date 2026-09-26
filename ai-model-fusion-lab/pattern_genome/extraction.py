"""EQUYLAPTA4 — compact functional-circuit extraction.

A CircuitPack is the TRANSFERRED OBJECT of this milestone: the minimal
numerical payload needed to reconstruct the discovered functional structure
inside a compatible target model.  It contains:

  * manifest  : circuit identity, source model + spec, member references,
                per-member tensor shapes/roles, impedance/normalization
                metadata, evidence level, fp16 quantization error
  * tensors   : per-member parameter slices stored as float16 (the payloads
                are small O(10^3-10^4) params; fp16 keeps packs at a few KB)

The pack NEVER contains the full source model.  Size accounting is part of
the science: extraction_bytes is reported for every circuit.

Also provides the SHUFFLED-CIRCUIT control (E4 Step 14): a payload variant
that preserves every tensor's shape, placement and value distribution while
destroying the functional assignment (row permutation along each tensor's
output axis, fixed seed).
"""
from __future__ import annotations

import io
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np


class CircuitPack:
    """Compact, inspectable extraction of one functional circuit."""

    def __init__(self, manifest: dict, tensors: Dict[str, Dict[str, np.ndarray]]):
        self.manifest = manifest
        self.tensors = tensors  # {member_key: {tensor_name: fp16 array}}

    # ------------------------------------------------------------------
    @classmethod
    def from_model(cls, model, circuit: dict, capability: str,
                   evidence_level: int = 0) -> "CircuitPack":
        """Extract the pack straight from a live model's state dict.

        `circuit` is a CircuitRecord.to_dict() (members are MemberRef dicts).
        """
        from .transfer import member_tensors_direct
        from .schema import MemberRef

        members = circuit["members"]
        keys = [MemberRef.from_dict(m).key() for m in members]
        tensors: Dict[str, Dict[str, np.ndarray]] = {}
        fp32_err = 0.0
        sq_num, sq_den, n_params = 0.0, 0.0, 0
        for key, m in zip(keys, members):
            raw = member_tensors_direct(model, m)
            pack = {}
            for name, arr in raw.items():
                arr = np.asarray(arr, dtype=np.float32)
                d = arr.astype(np.float16).astype(np.float32) - arr
                fp32_err = max(fp32_err, float(np.max(np.abs(d)) /
                                              (float(np.max(np.abs(arr))) + 1e-12)))
                sq_num += float(np.sum(d.astype(np.float64) ** 2))
                sq_den += float(np.sum(arr.astype(np.float64) ** 2))
                pack[name] = arr.astype(np.float16)
                n_params += int(arr.size)
            tensors[key] = pack
        rms_rel = (sq_num ** 0.5) / ((sq_den / max(1, n_params)) ** 0.5 *
                                     (n_params ** 0.5) + 1e-12)
        manifest = {
            "format": "equylapta-circuit-pack-1",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "circuit_id": circuit.get("circuit_id"),
            "kind": circuit.get("kind"),
            "pattern_ids": circuit.get("pattern_ids", []),
            "capability": capability,
            "evidence_level": evidence_level,
            "source_model": model.name,
            "source_spec": {"hidden": model.spec.hidden,
                            "layers": model.spec.layers,
                            "heads": model.spec.heads,
                            "mlp_hidden": getattr(model.spec, "mlp_hidden", None),
                            "ctx_len": model.spec.ctx_len,
                            "vocab": model.spec.vocab},
            "members": members,
            "member_keys": keys,
            "tensor_shapes": {k: {n: list(np.asarray(v).shape)
                                  for n, v in p.items()}
                              for k, p in tensors.items()},
            "n_params_fp16": n_params,
            "fp16_max_rel_error": round(fp32_err, 6),
            "fp16_rms_rel_error": round(float(np.sqrt(sq_num / (sq_den + 1e-12))), 8),
            "normalization": "payloads are raw source-weight slices; the "
                             "transfer ladder applies impedance out-scaling "
                             "||final_hidden||_src/||final_hidden||_tgt and "
                             "an optional residual gate at insertion time",
        }
        return cls(manifest, tensors)

    # ------------------------------------------------------------------
    def size_bytes(self) -> int:
        buf = io.BytesIO()
        np.savez_compressed(buf, **{f"{k}||{n}": v
                                    for k, p in self.tensors.items()
                                    for n, v in p.items()})
        return buf.tell()

    def save(self, path_no_ext: str) -> str:
        os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
        npz = path_no_ext + ".npz"
        np.savez_compressed(npz, **{f"{k}||{n}": v
                                    for k, p in self.tensors.items()
                                    for n, v in p.items()})
        with open(path_no_ext + ".json", "w") as f:
            json.dump(self.manifest, f, indent=1)
        return npz

    @classmethod
    def load(cls, npz_path: str) -> "CircuitPack":
        with open(npz_path.replace(".npz", ".json")) as f:
            manifest = json.load(f)
        z = np.load(npz_path)
        tensors: Dict[str, Dict[str, np.ndarray]] = {}
        for full, v in z.items():
            k, n = full.split("||", 1)
            tensors.setdefault(k, {})[n] = np.asarray(v)
        return cls(manifest, tensors)

    # ------------------------------------------------------------------
    def fp32(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {k: {n: np.asarray(v, dtype=np.float32)
                    for n, v in p.items()}
                for k, p in self.tensors.items()}

    def verify_against_model(self, model) -> float:
        """Max relative deviation between pack tensors and the live model's
        current slices (should be ~fp16 epsilon when verified on the source
        model it was extracted from)."""
        from .transfer import member_tensors_direct
        from .schema import MemberRef
        worst = 0.0
        for m in self.manifest["members"]:
            key = MemberRef.from_dict(m).key()
            raw = member_tensors_direct(model, m)
            for name, arr in raw.items():
                pk = self.tensors[key][name].astype(np.float32)
                denom = float(np.max(np.abs(arr))) + 1e-12
                worst = max(worst, float(np.max(np.abs(pk - arr))) / denom)
        return worst

    # ------------------------------------------------------------------
    def shuffled_payload(self, seed: int = 11) -> Dict[str, Dict[str, np.ndarray]]:
        """SHUFFLED-CIRCUIT control (E4 Step 14): same members, same tensor
        shapes, same value histograms — different functional assignment.
        Each 2-D tensor's rows are permuted along its output axis; 1-D
        tensors are permuted elementwise.  Placement and parameter count are
        EXACTLY preserved, so any performance difference vs the real pack is
        attributable to the discovered organization, not capacity."""
        rng = np.random.default_rng(seed)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for key, pack in self.tensors.items():
            sp = {}
            for name, arr in pack.items():
                arr = np.asarray(arr)
                if arr.ndim >= 2:
                    perm = rng.permutation(arr.shape[0])
                    sp[name] = arr[perm]
                elif arr.size > 1:
                    sp[name] = arr[rng.permutation(arr.shape[0])]
                else:
                    sp[name] = arr
            out[key] = sp
        return out


def pack_summary_line(pack: CircuitPack) -> str:
    m = pack.manifest
    return (f"{m['circuit_id']} [{m['kind']}] from {m['source_model']} "
            f"({len(m['members'])} members, {m['n_params_fp16']:,} params, "
            f"{pack.size_bytes() / 1024:.1f} KB, fp16 err "
            f"{m['fp16_max_rel_error']:.4f})")
