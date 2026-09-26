"""[WORKING with internet] HF model inspection WITHOUT downloading weights.

Fetches only config.json (+ optionally the safetensors index header list)
via HTTP. Enables the registry to analyze 7B+ models it can't load.
"""
from __future__ import annotations

import json
import urllib.request

from .base import ArchSpec

UA = {"User-Agent": "ai-model-fusion-lab/0.1"}


def _fetch_json(url: str, timeout: int = 12):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def probe_hf_config(hf_id: str, revision: str = "main") -> dict:
    base = f"https://huggingface.co/{hf_id}/resolve/{revision}"
    try:
        cfg = _fetch_json(f"{base}/config.json")
    except Exception as e:
        return {"name": hf_id, "backend": "hf-metadata-unavailable",
                "error": f"could not fetch config: {e}", "spec": {}, "license": "unknown"}
    arch = cfg.get("architectures", ["?"])[0]
    hidden = cfg.get("hidden_size") or cfg.get("d_model") or cfg.get("n_embd") or 0
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0
    heads = cfg.get("num_attention_heads") or cfg.get("n_head") or 0
    vocab = cfg.get("vocab_size") or 0
    ctx = cfg.get("max_position_embeddings") or cfg.get("n_positions") or 0
    moe = "num_experts" in cfg or "num_routed_experts" in cfg
    rope = "rope_theta" in cfg or "rotary" in json.dumps(cfg).lower()
    swiglu = cfg.get("intermediate_size") is not None and "gate" in json.dumps(cfg).lower()
    spec = ArchSpec(
        arch=arch, layers=int(layers), hidden=int(hidden), heads=int(heads),
        mlp_hidden=int(cfg.get("intermediate_size") or cfg.get("n_inner") or 0),
        vocab=int(vocab), ctx_len=int(ctx),
        positional="rope" if rope else "learned",
        norm="rmsnorm" if "rms_norm_eps" in cfg else "layernorm",
        attention=("mha" if heads else "?") + ("+gqa" if cfg.get("num_key_value_heads") not in (None, heads) else ""),
        mlp_act="swiglu" if swiglu else (cfg.get("activation") or "gelu"),
        tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        modality="vision+text" if cfg.get("vision_config") else "text",
        moe=moe, license=cfg.get("license", "unknown"), source="hf", hf_id=hf_id)
    return {"name": hf_id, "backend": "hf", "spec": spec.to_dict(),
            "param_count": cfg.get("n_ctx") and None,
            "license": cfg.get("license", "see model card"),
            "quant": None,
            "raw_config_keys": sorted(cfg.keys())}
