"""[WORKING] Specialists + routed super-model with output fusion.

A Specialist = micro model + capability tag + benchmark record.
The SuperModel routes inputs to 1..k specialists and fuses their outputs:
for MC scoring, fused logprob = mean of routed specialists' logprobs
(weighted by router confidence). Generation: top specialist produces.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from benchmarking.harness import eval_suite, score_item
from benchmarking.suites import load_suite
from router.router import LearnedRouter, rule_route


class Specialist:
    def __init__(self, capability: str, model, score: Optional[float] = None,
                 provenance: str = ""):
        self.capability = capability
        self.model = model
        self.score = score
        self.provenance = provenance
        self.latency_ms: Optional[float] = None

    def measure_latency(self, prompt: str = "12 + 14 =", reps: int = 3) -> float:
        ids = self.model.tokenizer.encode(prompt)
        t0 = time.perf_counter()
        for _ in range(reps):
            self.model.forward(ids[None, :])
        self.latency_ms = round((time.perf_counter() - t0) / reps * 1000, 1)
        return self.latency_ms


class SuperModel:
    """Router + specialists + fusion. Reports quality, latency, active params."""

    def __init__(self, specialists: List[Specialist], router: LearnedRouter):
        self.specialists = {s.capability: s for s in specialists}
        self.router = router

    def route(self, text: str) -> List[str]:
        return self.router.route(text)[0]

    def answer_mc(self, prompt: str, options: List[str]) -> int:
        caps, confs = self.router.route(prompt)
        caps = [c for c in caps if c in self.specialists] or list(self.specialists)[:1]
        scores = np.zeros(len(options))
        wsum = 0.0
        for c in caps:
            sp = self.specialists[c]
            w = max(0.05, confs.get(c, 0.5))
            wsum += w
            p_ids = sp.model.tokenizer.encode(prompt)
            for j, o in enumerate(options):
                scores[j] += w * sp.model.logprob_option(p_ids, sp.model.tokenizer.encode(" " + o, max_len=8))
        return int(np.argmax(scores / wsum))

    def answer_generate(self, prompt: str, max_new_tokens: int = 8) -> str:
        caps = self.route(prompt)
        caps = [c for c in caps if c in self.specialists] or list(self.specialists)[:1]
        sp = self.specialists[caps[0]]
        ids = sp.model.tokenizer.encode(prompt)
        out = sp.model.generate(ids, max_new_tokens=max_new_tokens)
        return sp.model.tokenizer.decode(out[len(ids):])

    def evaluate(self, suite_names: List[str], max_items: int = 50) -> dict:
        res = {}
        t0 = time.time()
        for sn in suite_names:
            suite = load_suite(sn)
            correct = 0
            for it in suite.items[:max_items]:
                if self.answer_mc(it.prompt, it.options) == it.answer:
                    correct += 1
            res[sn] = round(correct / max(1, len(suite.items[:max_items])), 3)
        self._eval = {"suites": res, "seconds": round(time.time() - t0, 1)}
        return self._eval

    def stats(self) -> dict:
        active = self.specialists
        return {
            "specialists": {c: {"params": s.model.param_count(),
                                "latency_ms": s.latency_ms,
                                "provenance": s.provenance} for c, s in active.items()},
            "total_params": sum(s.model.param_count() for s in active.values()),
            "avg_latency_ms": round(float(np.mean([s.latency_ms or 0 for s in active.values()])), 1),
        }

    def demonstrate_routing(self, queries: List[str]) -> str:
        lines = []
        for q in queries:
            caps, conf = self.router.route(q)
            caps = [c for c in caps if c in self.specialists]
            lines.append(f"  {q!r}\n    -> activate: {', '.join(caps)}  (confidence: " +
                         ", ".join(f"{c}:{conf[c]:.2f}" for c in caps) + ")")
        return "\n".join(lines)
