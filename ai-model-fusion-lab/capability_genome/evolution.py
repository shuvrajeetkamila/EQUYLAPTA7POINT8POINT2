"""Evolutionary component search (Phase 12).

Genome individual = candidate specialist construction:
  {components: [(source_model, region)], mechanism, coefficient}

Fitness (Phase 12, NOT benchmark-only):
  target capability score
  + mean secondary retention (capability-destruction guard)
  - small complexity penalty per component
  - latency penalty (measured)

Mutations: replace / remove / add component; change coefficient; change
mechanism. Crossover: mix components of two fit parents. All evaluated
honestly via the same benchmark path; history recorded in the tracker.
"""
from __future__ import annotations

import copy
import random
import time
from typing import Dict, List

import numpy as np

from benchmarking.harness import eval_suite
from benchmarking.suites import load_suite
from component_extraction.transplant import transplant_component, Component, _parse_region


def _apply(individual, base_model, models: Dict[str, object]) -> object:
    """Build the candidate model state: start from base, apply components."""
    snap = {k: v.copy() for k, v in base_model.state_dict().items()}
    try:
        for (src_name, region) in individual["components"]:
            src = models[src_name]
            comp = Component(source_model=src_name, architecture=src.spec.arch,
                             target_capability=individual["capability"],
                             region=region, level=_parse_region(region)["level"],
                             evidence_delta=0.0, confidence="search",
                             dependencies="n/a", tokenizer="shared",
                             hidden_dimension=src.spec.hidden)
            transplant_component(base_model, src, comp)
        if individual.get("mechanism") == "WEIGHT_MERGE" and len(individual["components"]) >= 0:
            pass  # weight mixing handled by 'coefficient' blend below
        c = individual.get("coefficient", 0.5)
        if "blend_with" in individual:
            other = models[individual["blend_with"]]
            osd, bsd = other.state_dict(), base_model.state_dict()
            mixed = {k: (c * osd[k] + (1 - c) * bsd[k]).astype(np.float32) for k in bsd}
            base_model.load_state_dict(mixed)
        return base_model
    finally:
        pass  # caller snapshots/restores


def evaluate_individual(individual, base_model, models, primary_suite: str,
                        secondary_suites: List[str], max_items: int = 30,
                        latency_prompt: str = "12 + 14 =") -> dict:
    snap = {k: v.copy() for k, v in base_model.state_dict().items()}
    try:
        _apply(individual, base_model, models)
        primary = eval_suite(base_model, load_suite(primary_suite),
                             max_items=max_items)["accuracy"] * 100
        secs = {sn: eval_suite(base_model, load_suite(sn), max_items=max_items)["accuracy"] * 100
                for sn in secondary_suites}
        ids = base_model.tokenizer.encode(latency_prompt)
        t0 = time.perf_counter()
        for _ in range(3):
            base_model.forward(ids[None, :])
        lat = (time.perf_counter() - t0) / 3 * 1000
        retention = float(np.mean(list(secs.values()))) if secs else 0.0
        complexity = len(individual["components"]) + (1 if "blend_with" in individual else 0)
        fitness = primary + 0.25 * retention - 0.5 * complexity - 0.01 * lat
        return {"primary": round(primary, 1), "secondary": {k: round(v, 1) for k, v in secs.items()},
                "latency_ms": round(lat, 1), "fitness": round(fitness, 2)}
    finally:
        base_model.load_state_dict(snap)


def evolve(base_model, models: Dict[str, object], pool_regions: List[tuple],
           primary_suite: str, secondary_suites: List[str],
           tracker=None, generations: int = 6, pop_size: int = 8,
           seed: int = 7, max_items: int = 30, log=None) -> dict:
    rng = random.Random(seed)
    def random_individual():
        n = rng.randint(1, 2)
        comps = [rng.choice(pool_regions) for _ in range(n)]
        ind = {"components": comps, "capability": primary_suite,
               "coefficient": round(rng.uniform(0.2, 0.8), 2)}
        if rng.random() < 0.4:
            ind["blend_with"] = rng.choice(list(models))
        if rng.random() < 0.5:
            ind["mechanism"] = "WEIGHT_MERGE"
        return ind

    population = [random_individual() for _ in range(pop_size)]
    history = []
    best = None
    for gen in range(generations):
        scored = []
        for ind in population:
            r = evaluate_individual(ind, base_model, models, primary_suite,
                                    secondary_suites, max_items)
            scored.append((r["fitness"], ind, r))
            if tracker:
                tracker.record(exp_type="evolution", config={"gen": gen, **{k: v for k, v in ind.items()}},
                               sources=[base_model.name],
                               results={"fitness": r["fitness"], "primary": r["primary"],
                                        "latency_ms": r["latency_ms"],
                                        "secondary": r["secondary"]},
                               target_suite=primary_suite, license="Apache-2.0")
        scored.sort(key=lambda x: -x[0])
        history.append({"gen": gen, "best_fitness": scored[0][0],
                        "best": scored[0][2]})
        if log:
            log(f"  gen {gen}: best fitness {scored[0][0]:.2f} "
                f"(primary {scored[0][2]['primary']:.0f}%) {scored[0][1]['components']}")
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        # --- build the next generation: elites + crossover/mutation ---
        elites = [s[1] for s in scored[:2]]
        next_pop = list(elites)
        while len(next_pop) < pop_size:
            if rng.random() < 0.5 and len(scored) >= 2:
                p1, p2 = rng.choice(scored[:4])[1], rng.choice(scored[:4])[1]
                child = {"components": (p1["components"][:1] + p2["components"][-1:]),
                         "capability": primary_suite,
                         "coefficient": round((p1["coefficient"] + p2["coefficient"]) / 2, 2)}
                if "blend_with" in p1 or "blend_with" in p2:
                    child["blend_with"] = p1.get("blend_with") or p2.get("blend_with")
            else:
                child = copy.deepcopy(rng.choice(elites))
                r = rng.random()
                if r < 0.35 and child["components"]:
                    child["components"][rng.randrange(len(child["components"]))] = rng.choice(pool_regions)
                elif r < 0.55 and child["components"]:
                    child["components"].pop(rng.randrange(len(child["components"])))
                    if not child["components"]:
                        child["components"] = [rng.choice(pool_regions)]
                elif r < 0.75:
                    child["components"].append(rng.choice(pool_regions))
                elif r < 0.9:
                    child["coefficient"] = round(rng.uniform(0.1, 0.9), 2)
                else:
                    child["blend_with"] = rng.choice(list(models))
            next_pop.append(child)
        population = next_pop
    return {"best_fitness": best[0], "best_individual": best[1],
            "best_metrics": best[2], "history": history}
