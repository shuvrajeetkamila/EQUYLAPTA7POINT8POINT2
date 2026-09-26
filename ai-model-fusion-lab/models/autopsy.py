"""[WORKING] Model Autopsy Engine.

Structural inspection + weight statistics per layer, producing a visual map.
Works on any ModelHandle via param_groups(). Activation/attention statistics
are collected on a probe batch when the model can run locally [PARTIAL for
very large models: use activation sampling].
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _stats(w: np.ndarray) -> dict:
    # float64 ACCUMULATION, not conversion: avoids 2x-memory spikes on big
    # tensors (OOM-killed the sandbox on gpt2 during the M2 audit)
    w = np.asarray(w, dtype=np.float32)
    absf = np.abs(w)
    mean = float(w.mean(dtype=np.float64))
    std = float(w.std(dtype=np.float64))
    return {
        "shape": list(w.shape),
        "n": int(w.size),
        "mean": mean,
        "std": std,
        "norm": float(np.sqrt((w * w).sum(dtype=np.float64))),
        "sparsity": round(float((absf < 1e-6).mean()), 4),
        "outlier_pct": round(float((absf > absf.mean() + 6 * absf.std()).mean()), 5),
    }


def autopsy(m) -> dict:
    g = m.param_groups()
    report = {
        "name": m.name, "backend": m.backend,
        "arch": m.spec.to_dict(), "param_count": m.param_count(),
        "embedding": {k: _stats(v) for k, v in g["embedding"].items()},
        "final_norm": {k: _stats(v) for k, v in g["final_norm"].items()},
        "layers": {},
    }
    for l, mods in g["layers"].items():
        report["layers"][l] = {name: {k: _stats(v) for k, v in tensors.items()}
                               for name, tensors in mods.items()}
    return report


def tree_map(report: dict) -> str:
    """ASCII visual map (spec section 3)."""
    L = ["MODEL " + report["name"] + f"  ({report['param_count']:,} params, {report['backend']})"]
    emb = report["embedding"]
    L.append("├── Embedding  We" + str(emb.get("We", {}).get("shape", "?")) +
             f"  norm={emb.get('We', {}).get('norm', 0):.1f}")
    for l in sorted(report["layers"], key=lambda x: int(x)):
        ln = report["layers"][l]
        at = ln.get("attn", {}); ml = ln.get("mlp", {})
        def nm(d):
            return "/".join(str(v.get("norm", 0))[:6] for v in d.values())
        L.append(f"├── Layer {l}")
        L.append(f"│   ├── Attention  qkv norm={at.get('qkv.W', {}).get('norm', 0):.1f}  out norm={at.get('o.W', {}).get('norm', 0):.1f}")
        L.append(f"│   ├── MLP        W1 norm={ml.get('1.W', {}).get('norm', 0):.1f}  W2 norm={ml.get('2.W', {}).get('norm', 0):.1f}")
        L.append(f"│   └── Residual   (x + attn + mlp)")
    L.append(f"└── Output  final-norm + tied logits")
    return "\n".join(L)


def activation_autopsy(m, probe_ids: np.ndarray) -> dict:
    """Run one probe batch, collect activation statistics.
    [WORKING for micro backends; WORKING for HF via hooks since M2]
    attn_probs are None for HF backends (internal attention maps are not
    exposed uniformly) — entropy is skipped rather than faked."""
    logits, cache = m.forward(probe_ids, collect=True)
    out = {"probe_shape": list(probe_ids.shape), "logit_std": float(logits.std()),
           "layers": {}}
    for l in sorted(cache["resid_in"]):
        ri = cache["resid_in"][l]
        ao = cache.get("attn_out", {}).get(l)
        mo = cache.get("mlp_out", {}).get(l)
        pr = cache.get("attn_probs", {}).get(l)
        rec = {"resid_std": float(ri.std())}
        if ao is not None:
            rec["attn_out_std"] = float(ao.std())
            rec["attn_out_share"] = round(float(ao.std() / (ri.std() + 1e-9)), 3)
        if mo is not None:
            rec["mlp_out_std"] = float(mo.std())
            rec["mlp_out_share"] = round(float(mo.std() / (ri.std() + 1e-9)), 3)
        if pr is not None:
            rec["attn_entropy"] = float(-(pr * np.log(pr + 1e-9)).sum(-1).mean())
        out["layers"][l] = rec
    return out
