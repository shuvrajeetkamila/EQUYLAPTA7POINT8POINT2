"""[WORKING] Real-model selection mechanism (Phase 16).

Never assumes the machine can load a model: estimates parameter memory from
the HF config (fetched WITHOUT downloading weights) and compares against
available RAM. Users can always pass --model MODEL_ID explicitly.
"""
from __future__ import annotations

import json
import os
import urllib.request

UA = {"User-Agent": "equylapta-fusion-lab/0.2"}


def available_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0


def estimate_params_bytes(cfg: dict, dtype_bytes: int = 4) -> int:
    """Rough parameter estimate from config fields (no weight download)."""
    L = cfg.get("n_layer") or cfg.get("num_hidden_layers") or 0
    d = cfg.get("n_embd") or cfg.get("hidden_size") or 0
    V = cfg.get("vocab_size") or 0
    f = cfg.get("n_inner") or cfg.get("intermediate_size") or 4 * d
    per_layer = 4 * d * d + 2 * d * f          # attn qkvo + mlp up/down (rough; gated MLPs underestimate)
    emb = V * d
    return int(L * per_layer + emb)


def check_model_fits(hf_id: str, revision: str = "main",
                     safety: float = 1.5, dtype_bytes: int = 4) -> dict:
    from .hf_probe import _fetch_json
    ram = available_ram_gb()
    try:
        cfg = _fetch_json(f"https://huggingface.co/{hf_id}/resolve/{revision}/config.json")
    except Exception as e:
        return {"model": hf_id, "fits": False, "reason": f"config fetch failed: {e}",
                "available_ram_gb": round(ram, 2)}
    n_bytes = estimate_params_bytes(cfg)
    need_gb = n_bytes / 1e9 * dtype_bytes
    fits = n_bytes / 1e9 * dtype_bytes * safety < ram
    return {"model": hf_id, "est_params": n_bytes,
            "est_ram_gb": round(need_gb, 2), "available_ram_gb": round(ram, 2),
            "safety_factor": safety, "fits": bool(fits),
            "reason": "OK" if fits else
            f"needs ~{need_gb:.1f}GB (+{safety:.1f}x safety) > {ram:.1f}GB available"}


def recommend(revision: str = "main") -> list:
    """Small-to-large open models with license families; filtered by RAM."""
    candidates = [
        ("EleutherAI/pythia-70m", "Apache-2.0"),
        ("EleutherAI/pythia-160m", "Apache-2.0"),
        ("gpt2", "MIT"),
        ("gpt2-medium", "MIT"),
        ("EleutherAI/pythia-410m", "Apache-2.0"),
        ("Qwen/Qwen2.5-0.5B", "Apache-2.0"),
        ("mistralai/Mistral-7B-v0.3", "Apache-2.0"),
        ("meta-llama/Llama-3.1-8B", "llama-community"),
    ]
    out = []
    for mid, lic in candidates:
        r = check_model_fits(mid, revision)
        r["license"] = lic
        out.append(r)
    return out
