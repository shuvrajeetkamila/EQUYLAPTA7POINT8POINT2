"""[WORKING] Benchmark harness: multiple-choice log-likelihood evaluation.

Works for ANY ModelHandle (numpy micro models and HF models alike).
Scores: accuracy per suite + bootstrap 95% CI. Held-out suites only.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from models.base import ModelHandle

from .suites import Suite, load_suite, SuiteItem


def score_item(m: ModelHandle, prompt: str, options: List[str]) -> int:
    p_ids = m.tokenizer.encode(prompt)
    scores = [m.logprob_option(p_ids, m.tokenizer.encode(" " + o, max_len=8)) for o in options]
    return int(np.argmax(scores))


def eval_suite(m: ModelHandle, suite: Suite, max_items: int = 60, log=None) -> Dict:
    items = suite.items[:max_items]
    t0 = time.time()
    vals = [1.0 if score_item(m, it.prompt, it.options) == it.answer else 0.0
            for it in items]                      # single scoring pass (M2 fix)
    correct = int(sum(vals))
    acc = correct / max(1, len(items))
    rng = np.random.default_rng(0)
    bs = [float(np.mean(rng.choice(vals, len(vals)))) for _ in range(300)]
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    dt = time.time() - t0
    return {"suite": suite.name, "domain": suite.domain, "n": len(items),
            "accuracy": round(acc, 4), "ci95": [round(ci[0], 3), round(ci[1], 3)],
            "n_correct": correct, "seconds": round(dt, 2)}


def eval_model(m: ModelHandle, suite_names: List[str], max_items: int = 60,
               log=None) -> Dict[str, Dict]:
    out = {}
    for sn in suite_names:
        suite = load_suite(sn)
        r = eval_suite(m, suite, max_items=max_items, log=log)
        out[sn] = r
        if log:
            log(f"  [{m.name}] {sn}: {r['accuracy']*100:.1f}% (n={r['n']}, {r['seconds']}s)")
    return out


def capability_matrix(results: Dict[str, Dict[str, Dict]]) -> str:
    """Pretty-print matrix: rows=models, cols=suites."""
    models = list(results)
    suites = sorted({s for r in results.values() for s in r})
    head = f"{'MODEL':<22}" + "".join(f"{s[:12]:>14}" for s in suites)
    lines = [head, "-" * len(head)]
    for mo in models:
        row = f"{mo:<22}"
        for s in suites:
            v = results[mo].get(s, {}).get("accuracy")
            row += f"{v*100:>13.1f}%" if v is not None else f"{'—':>14}"
        lines.append(row)
    return "\n".join(lines)
