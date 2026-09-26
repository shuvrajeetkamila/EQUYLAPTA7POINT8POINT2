"""[WORKING] Architecture support gauntlet (Milestone 2, Phase 1B).

An architecture is marked SUPPORTED only after passing all five steps:
  LOAD -> AUTOPSY -> FORWARD -> BENCHMARK -> ABLATION (layer/module/head)

Real small models (gpt2, pythia-70m) verify full behaviour. Tiny *random*
checkpoints (hf-internal-testing/tiny-random-*) verify ARCHITECTURE mechanics
for families too large to load; their capability scores are chance-level by
construction and are never reported as capability results.

Usage:  python -m models.arch_gauntlet [--json OUT.json]
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# (model_id, family, is_random_weights)
GAUNTLET_MODELS = [
    ("gpt2", "GPT-2", False),
    ("EleutherAI/pythia-70m", "GPT-NeoX", False),
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", "Llama-style", True),
    ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "Qwen-style", True),
    ("hf-internal-testing/tiny-random-MistralForCausalLM", "Mistral-style", True),
    ("hmellor/tiny-random-Gemma2ForCausalLM", "Gemma-style", True),
    ("hf-internal-testing/tiny-random-GPTNeoXForCausalLM", "GPT-NeoX (tiny)", True),
]

PROMPT = "The capital of France is"


def _text_bench(h, n_items: int = 6) -> dict:
    from benchmarking.text_suites import load_text_suite
    from benchmarking.harness import eval_suite
    r = eval_suite(h, load_text_suite("text-math", 20), max_items=n_items)
    return {"accuracy": r["accuracy"], "n": r["n"]}


def gauntlet(model_id: str, family: str, random_weights: bool, log=None) -> dict:
    res = {"model": model_id, "family": family,
           "random_weights": random_weights, "steps": {}}
    def step(name, ok, detail=""):
        res["steps"][name] = {"pass": bool(ok), "detail": detail}
        if log:
            log(f"    {'PASS' if ok else 'FAIL'} {name}: {detail}")
        return ok
    try:
        from models.backend_hf import HFHandle
        h = HFHandle(model_id)
        step("LOAD", True, f"{h.param_count():,} params, L{h.spec.layers} d{h.spec.hidden} H{h.spec.heads}")
    except Exception as e:
        step("LOAD", False, f"{type(e).__name__}: {str(e)[:100]}")
        res["verdict"] = "UNSUPPORTED"
        return res
    try:
        from models.autopsy import autopsy
        rep = autopsy(h)
        ok = len(rep["layers"]) == h.spec.layers and h.spec.layers > 0
        step("AUTOPSY", ok, f"{len(rep['layers'])} layer groups in report")
    except Exception as e:
        step("AUTOPSY", False, f"{type(e).__name__}: {str(e)[:100]}")
    try:
        ids = h.tokenizer.encode(PROMPT)[:10]
        logits, _ = h.forward(np.array(ids)[None, :])
        ok = np.isfinite(logits).all() and logits.shape[-1] == h.spec.vocab
        step("FORWARD", ok, f"logits {tuple(logits.shape)}")
    except Exception as e:
        step("FORWARD", False, f"{type(e).__name__}: {str(e)[:100]}")
    try:
        r = _text_bench(h)
        ok = 0.0 <= r["accuracy"] <= 1.0 and r["n"] > 0
        note = "" if not random_weights else " (random weights: chance-level, mechanics only)"
        step("BENCHMARK", ok, f"acc={r['accuracy']:.2f} n={r['n']}{note}")
    except Exception as e:
        step("BENCHMARK", False, f"{type(e).__name__}: {str(e)[:100]}")
    try:
        ids = h.tokenizer.encode(PROMPT)[:10]
        x = np.array(ids)[None, :]
        base, _ = h.forward(x)
        d = {}
        h.skip_layers = [0]
        d["layer"], _ = h.forward(x)
        h.skip_layers = []
        h.skip_modules = {min(1, h.spec.layers - 1): ["mlp"]}
        d["module"], _ = h.forward(x)
        h.skip_modules = {}
        h.head_mask = np.ones((h.spec.layers, h.spec.heads), np.float32)
        h.head_mask[0, 0] = 0
        d["head"], _ = h.forward(x)
        h.head_mask = None
        dl = float(np.abs(base - d["layer"]).max())
        dm = float(np.abs(base - d["module"]).max())
        dh = float(np.abs(base - d["head"]).max())
        ok = dl > 1e-6 and dm > 1e-6 and dh > 1e-6
        step("ABLATION", ok, f"deltas layer={dl:.3f} module={dm:.3f} head={dh:.3f}")
    except Exception as e:
        step("ABLATION", False, f"{type(e).__name__}: {str(e)[:100]}")
    passed = all(s["pass"] for s in res["steps"].values())
    res["verdict"] = "SUPPORTED" if passed else "PARTIAL/UNSUPPORTED"
    return res


def run_all(log=None, out_json: str = None):
    results = []
    for mid, fam, rnd in GAUNTLET_MODELS:
        if log:
            log(f"  gauntlet: {mid} [{fam}]")
        results.append(gauntlet(mid, fam, rnd, log=log))
    supported = [r["family"] for r in results if r.get("verdict") == "SUPPORTED"]
    if log:
        log(f"  SUPPORTED architectures: {supported}")
    if out_json:
        json.dump(results, open(out_json, "w"), indent=1)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run_all(log=print, out_json=args.json)
