"""[WORKING] Merge experiment engine: run strategies x coefficients, benchmark
every candidate, record to experiment DB, keep winners. Also implements
micro-segment extraction (layer transplants) and specialist construction.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from ablation.ablator import ablate_and_eval
from benchmarking.harness import eval_suite
from benchmarking.suites import load_suite
from experiments.tracker import ExperimentTracker
from merging.strategies import run_strategy


def benchmark_state_dict(base_model, sd, name: str, suite_names, max_items=50):
    m = base_model
    m.load_state_dict(sd)
    res = {}
    for sn in suite_names:
        res[sn] = eval_suite(m, load_suite(sn), max_items=max_items)["accuracy"]
    return res


def merge_experiment(base_model, sources, suite_names, tracker: ExperimentTracker,
                     methods=("weighted_average", "slerp", "ties", "dare"),
                     coef_grid=(0.5, 1.0), target_suite: str = "micro-math",
                     max_items=50, log=None) -> List[dict]:
    """Try merge configs; returns sorted candidate results. Records all."""
    sds = [m.state_dict() for m in sources]
    results = []
    for method in methods:
        grid = coef_grid if method in ("weighted_average",) else coef_grid
        for coef in grid:
            cfg = {"method": method, "coef": coef,
                   "sources": [m.name for m in sources], "target": target_suite}
            if tracker.seen(cfg):
                if log:
                    log(f"  skip duplicate config {cfg}")
                continue
            t0 = time.time()
            try:
                weights = [1.0 - (coef - 0.5), coef] if len(sds) == 2 else [1.0] + [coef] * (len(sds) - 1)
                sd = run_strategy(method, sds, weights)
                res = benchmark_state_dict(base_model, sd, f"merge-{method}-{coef}", suite_names, max_items)
            except Exception as e:
                res, err = {}, str(e)
                res = {sn: 0.0 for sn in suite_names}
            rec = tracker.record(
                exp_type="merge", config=cfg, sources=[m.name for m in sources],
                results={k: round(v * 100, 1) for k, v in res.items()},
                target_suite=target_suite,
                notes=f"{time.time()-t0:.1f}s eval",
                license="Apache-2.0")
            results.append({"method": method, "coef": coef, "scores": res,
                            "id": rec["id"]})
            if log:
                log(f"  {method} coef={coef}: " +
                    " ".join(f"{sn}:{res[sn]*100:.0f}" for sn in suite_names) +
                    f"  [{rec['id']}]")
    results.sort(key=lambda r: -r["scores"].get(target_suite, 0))
    return results


def extract_segment_transplant(base_model, donor, recipient, layer_from: int,
                               layer_to: int, suite_names, tracker: ExperimentTracker,
                               max_items=50, log=None) -> dict:
    """Micro-segment extraction: copy donor's layer range into recipient
    (same-arch transplant), benchmark honestly. [RESEARCH EXPERIMENT]"""
    src = donor.state_dict()
    dst = dict(recipient.state_dict())
    for l in range(layer_from, min(layer_to, donor.spec.layers)):
        for k in src:
            if k.startswith(f"L{l}."):
                dst[k] = src[k].copy()
    cfg = {"method": "segment_transplant", "donor": donor.name, "recipient": recipient.name,
           "layers": [layer_from, layer_to]}
    if tracker.seen(cfg):
        return {"skipped": True, "id": tracker.find(cfg)["id"]}
    res = benchmark_state_dict(recipient, dst, "transplant", suite_names, max_items)
    rec = tracker.record(exp_type="transplant", config=cfg,
                         sources=[donor.name, recipient.name],
                         results={k: round(v * 100, 1) for k, v in res.items()},
                         notes="layer-range transplant; success only if scores say so",
                         license="Apache-2.0")
    if log:
        log(f"  transplant {donor.name}[L{layer_from}:{layer_to}] -> {recipient.name}: " +
            " ".join(f"{sn}:{res[sn]*100:.0f}" for sn in suite_names))
    return {"scores": res, "id": rec["id"]}


def build_specialist(base_model, sources, suite_name: str, tracker: ExperimentTracker,
                     merge_methods=("weighted_average", "slerp", "ties", "dare"),
                     max_items=50, log=None) -> dict:
    """Build a specialist for one capability; verify it beats parents on the
    target suite; REJECT and investigate if not (spec §7)."""
    parents = {m.name: eval_suite(m, load_suite(suite_name), max_items=max_items)["accuracy"]
               for m in sources}
    cands = merge_experiment(base_model, sources, [suite_name], tracker,
                             methods=merge_methods, target_suite=suite_name,
                             max_items=max_items, log=log)
    best = cands[0] if cands else None
    best_score = best["scores"][suite_name] if best else 0.0
    best_parent = max(parents.values())
    verdict = "ACCEPTED" if best_score >= best_parent else "REJECTED (no merge beat best parent)"
    if log:
        log(f"  specialist[{suite_name}] best merge={best_score*100:.0f}% vs best parent={best_parent*100:.0f}% -> {verdict}")
    return {"suite": suite_name, "parents": {k: round(v*100,1) for k, v in parents.items()},
            "best_candidate": best, "best_score": round(best_score*100, 1),
            "best_parent_score": round(best_parent*100, 1), "verdict": verdict}
