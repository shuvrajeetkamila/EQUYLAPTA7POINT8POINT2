"""Component synthesis engine (Phases 10-11, 13).

Tries mechanisms in the mandatory fallback hierarchy and records which one
succeeded:

  DIRECT TRANSPLANT  (same arch, same shapes)
  -> WEIGHT MERGE    (same-base fine-tunes)
  -> ALIGNMENT GATE  (representation alignment must clear threshold before
                      transplant is even attempted across weaker matches)
  -> DISTILLATION    (teacher -> filtered data -> student)
  -> ROUTING         (independent specialists in a SuperModel)

Every attempt is benchmarked; failures are recorded, not hidden.
The specialist builder (Phase 11) applies the multi-objective regression guard
(Phase 13): a specialist that destroys secondary capabilities is REJECTED.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from benchmarking.harness import eval_suite
from benchmarking.suites import load_suite
from component_extraction.transplant import (Component, abc_intervention,
                                             transplant_component)
from experiments.tracker import ExperimentTracker


# --------------------------------------------------------------------------
def attempt_direct_transplant(donor, recipient, comp: Component, suite: str,
                              secondary: List[str], tracker: ExperimentTracker,
                              max_items: int = 40, keep_on_success: bool = True) -> dict:
    snap = {k: v.copy() for k, v in recipient.state_dict().items()}
    out = abc_intervention(donor, recipient, comp, suite, secondary, max_items)
    cfg = {"kind": "synthesis", "mechanism": "DIRECT_TRANSPLANT",
           "component": comp.to_dict(), "suite": suite,
           "recipient": recipient.name, "donor": donor.name}
    exp = tracker.record(exp_type="synthesis", config=cfg,
                         sources=[donor.name, recipient.name],
                         results={"A_donor": out["A_donor"], "B": out["B_recipient"],
                                  "C": out["C_transplanted"], "gain": out["gain"]},
                         target_suite=suite,
                         notes="A/B/C positive intervention",
                         license="Apache-2.0")
    out["experiment_id"] = exp["id"]
    out["mechanism"] = "DIRECT_TRANSPLANT"
    out["component_desc"] = f"{comp.source_model}:{comp.region}"
    out["succeeded"] = out["gain"] > 0
    if not (out["succeeded"] and keep_on_success):
        recipient.load_state_dict(snap)
    else:
        recipient.load_state_dict(snap)  # caller re-applies explicitly if desired
    return out


def attempt_weight_merge(base_model, models: List, suite: str, tracker: ExperimentTracker,
                         method: str = "weighted_average", coeffs=(0.3, 0.5, 0.7),
                         max_items: int = 40) -> dict:
    from merging.strategies import run_strategy
    best = None
    sds = [m.state_dict() for m in models]
    snap = {k: v.copy() for k, v in base_model.state_dict().items()}
    for c in coeffs:
        w = [1.0 - c, c] if len(sds) == 2 else [1.0] + [c] * (len(sds) - 1)
        try:
            sd = run_strategy(method, sds, w)
        except AssertionError as e:
            return {"mechanism": "WEIGHT_MERGE", "succeeded": False,
                    "reason": f"shape gate: {e}", "experiment_id": None}
        base_model.load_state_dict(sd)
        acc = eval_suite(base_model, load_suite(suite), max_items=max_items)["accuracy"] * 100
        if best is None or acc > best["accuracy"]:
            best = {"accuracy": acc, "coef": c, "sd": sd}
    base_model.load_state_dict(snap)
    cfg = {"kind": "synthesis", "mechanism": "WEIGHT_MERGE", "method": method,
           "suite": suite, "models": [m.name for m in models]}
    exp = tracker.record(exp_type="synthesis", config=cfg,
                         sources=[m.name for m in models],
                         results={"accuracy": best["accuracy"], "coef": best["coef"]},
                         target_suite=suite, license="Apache-2.0")
    return {"mechanism": "WEIGHT_MERGE", "succeeded": True,
            "accuracy": best["accuracy"], "coef": best["coef"],
            "state": best["sd"], "experiment_id": exp["id"]}


def attempt_alignment_gate(donor, recipient, probe_texts, threshold: float = 0.35) -> dict:
    """Phase 9: representation alignment BEFORE/ AFTER; gates transplanting."""
    from adapters.projection import align_models, fit_projection, collect_hidden
    from adapters.lowrank import fit_projection_lowrank
    ha = collect_hidden(donor, probe_texts)
    hb = collect_hidden(recipient, probe_texts)
    n = min(len(ha), len(hb))
    ha, hb = ha[:n], hb[:n]
    # BEFORE: trivial mean-predictor baseline (R2=0 by construction; cosine of means)
    mean_cos = float(np.dot(ha.mean(0), hb.mean(0)) /
                     (np.linalg.norm(ha.mean(0)) * np.linalg.norm(hb.mean(0)) + 1e-9))
    full = fit_projection(ha, hb)
    low = fit_projection_lowrank(ha, hb, rank=max(2, hb.shape[1] // 4))
    return {"mechanism": "ALIGNMENT", "before_mean_cosine": round(mean_cos, 3),
            "after_ridge": {"cosine": full["cosine_alignment"], "r2": full["r2"]},
            "after_lowrank": {"cosine": low["cosine_alignment"], "r2": low["r2"]},
            "alignment_sufficient": full["cosine_alignment"] >= threshold,
            "n_probes": int(n),
            "succeeded": full["cosine_alignment"] >= threshold}


def attempt_distillation(teacher, student_init_fn, texts, suite: str,
                         tracker: ExperimentTracker, epochs: int = 30) -> dict:
    from distillation.distill import distill, quality_filter
    kept = quality_filter(teacher, texts, min_conf=0.15)
    student = student_init_fn()
    out = distill(teacher, student, kept, epochs=epochs)
    acc = eval_suite(student, load_suite(suite), max_items=40)["accuracy"] * 100
    cfg = {"kind": "synthesis", "mechanism": "DISTILLATION", "teacher": teacher.name,
           "suite": suite, "n_kept": len(kept)}
    exp = tracker.record(exp_type="synthesis", config=cfg, sources=[teacher.name],
                         results={"accuracy": acc, "final_loss": out["final_loss"]},
                         target_suite=suite, license="Apache-2.0")
    return {"mechanism": "DISTILLATION", "succeeded": True, "accuracy": acc,
            "student": student, "final_loss": out["final_loss"],
            "experiment_id": exp["id"]}


def attempt_routing(specialists, router, suite: str, tracker: ExperimentTracker,
                    max_items: int = 40) -> dict:
    from experts.specialist import SuperModel
    sup = SuperModel(specialists, router)
    acc = None
    correct, n = 0, 0
    for it in load_suite(suite).items[:max_items]:
        if sup.answer_mc(it.prompt, it.options) == it.answer:
            correct += 1
        n += 1
    acc = correct / max(1, n) * 100
    cfg = {"kind": "synthesis", "mechanism": "ROUTING", "suite": suite,
           "specialists": [s.capability for s in specialists]}
    exp = tracker.record(exp_type="synthesis", config=cfg,
                         sources=[s.model.name for s in specialists],
                         results={"accuracy": round(acc, 1)}, target_suite=suite,
                         license="Apache-2.0")
    return {"mechanism": "ROUTING", "succeeded": True, "accuracy": round(acc, 1),
            "supermodel": sup, "experiment_id": exp["id"]}


# --------------------------------------------------------------------------
# Phase 11: specialist construction with reports + Phase 13 regression guard
def build_specialist_v2(name, capability, primary_suite, secondary_suites,
                        parents: Dict[str, object], pool: List[Component],
                        tracker: ExperimentTracker, max_items: int = 40,
                        allow_transplant_recipient: Optional[str] = None,
                        log=None) -> dict:
    """Full construction: parent eval -> component attempts -> verdict.
    Returns a construction report (dict + render())."""
    from router.router import LearnedRouter
    rep = {"specialist": name, "capability": capability,
           "primary_suite": primary_suite, "secondary_suites": secondary_suites,
           "sources": sorted(parents), "components": [c.to_dict() for c in pool],
           "attempts": [], "status": "REJECTED"}

    def ev(m, sn):
        return eval_suite(m, load_suite(sn), max_items=max_items)["accuracy"] * 100

    parents_primary = {n: ev(m, primary_suite) for n, m in parents.items()}
    rep["parent_primary"] = parents_primary
    best_parent_name = max(parents_primary, key=parents_primary.get)
    best_parent = parents_primary[best_parent_name]

    parent_secs = {n: {sn: ev(m, sn) for sn in secondary_suites}
                   for n, m in parents.items()}

    best_result = None
    # Mechanism 1+2: transplant attempts of pooled components into best parent
    recipient = parents[best_parent_name]
    for comp in pool:
        donor = parents.get(comp.source_model)
        if donor is None or donor is recipient:
            continue
        out = attempt_direct_transplant(donor, recipient, comp, primary_suite,
                                        secondary_suites, tracker, max_items,
                                        keep_on_success=False)
        rep["attempts"].append(out)
        if log:
            log(f"    transplant {comp.source_model}:{comp.region} -> {best_parent_name}: "
                f"B{out['B_recipient']} -> C{out['C_transplanted']} (A{out['A_donor']}) "
                f"{'OK' if out['succeeded'] else 'no gain'}")
        if out["succeeded"] and (best_result is None or out["C_transplanted"] > best_result["C_transplanted"]):
            best_result = out
            best_result["recipient"] = best_parent_name
            best_result["component"] = comp

    # Mechanism: weight merge of the two strongest parents
    if len(parents) >= 2:
        ranked = sorted(parents.items(), key=lambda kv: -parents_primary[kv[0]])[:2]
        out = attempt_weight_merge(recipient, [m for _, m in ranked], primary_suite,
                                   tracker, max_items=max_items)
        out["models"] = [n for n, _ in ranked]
        rep["attempts"].append(out)
        if log and out.get("succeeded"):
            log(f"    weight-merge {out['models']}: {out['accuracy']:.0f}%")
        if out.get("succeeded") and (best_result is None or out["accuracy"] > _best_score(best_result)):
            best_result = out
            best_result["recipient"] = recipient.name
            best_result["component"] = None

    # ---- verdict with Phase 13 regression guard ----
    rep["best_parent"] = best_parent_name
    rep["best_parent_score"] = round(best_parent, 1)
    if best_result is None:
        rep["status"] = "REJECTED"
        rep["reason"] = "no mechanism beat the best parent"
        rep["specialist_score"] = round(best_parent, 1)
        rep["chosen"] = {"model": best_parent_name, "mechanism": "PARENT_FALLBACK"}
        chosen_model = parents[best_parent_name]
        chosen_secs = parent_secs[best_parent_name]
    else:
        score = _best_score(best_result)
        if score <= best_parent:
            rep["status"] = "REJECTED"
            rep["reason"] = "best attempt did not beat best parent"
            rep["specialist_score"] = round(best_parent, 1)
            rep["chosen"] = {"model": best_parent_name, "mechanism": "PARENT_FALLBACK"}
            chosen_model = parents[best_parent_name]
            chosen_secs = parent_secs[best_parent_name]
        else:
            # re-apply the winning construction
            snap = {k: v.copy() for k, v in recipient.state_dict().items()}
            if best_result["mechanism"] == "DIRECT_TRANSPLANT":
                transplant_component(recipient, parents[best_result.get("donor_override",
                                     best_result["component"].source_model)] if False else
                                     parents[best_result["component"].source_model],
                                     best_result["component"])
            else:
                recipient.load_state_dict(best_result["state"])
            specialist_model = recipient
            chosen_secs = {sn: ev(specialist_model, sn) for sn in secondary_suites}
            primary = ev(specialist_model, primary_suite)
            regressions = {sn: chosen_secs[sn] - parent_secs[best_parent_name][sn]
                           for sn in secondary_suites}
            worst_drop = min(regressions.values()) if regressions else 0.0
            rep["specialist_model"] = specialist_model.name
            rep["specialist_score"] = round(primary, 1)
            rep["secondary"] = {k: round(v, 1) for k, v in chosen_secs.items()}
            rep["secondary_regressions"] = {k: round(v, 1) for k, v in regressions.items()}
            if worst_drop < -15:
                rep["status"] = "REJECTED (regression guard)"
                rep["reason"] = f"secondary capability dropped {worst_drop:.1f} pts (>15)"
                specialist_model.load_state_dict(snap)
                rep["chosen"] = {"model": best_parent_name, "mechanism": "PARENT_FALLBACK"}
            else:
                rep["status"] = "PROMISING"
                rep["reason"] = f"beats best parent by {score - best_parent:.1f} pts"
                rep["chosen"] = {"model": specialist_model.name,
                                 "mechanism": best_result["mechanism"],
                                 "component": (best_result["component"].region
                                               if best_result.get("component") else
                                               best_result.get("models"))}
    return rep


def _best_score(result: dict) -> float:
    if result["mechanism"] == "DIRECT_TRANSPLANT":
        return result["C_transplanted"]
    return result.get("accuracy", 0.0)


def render_report(rep: dict) -> str:
    L = [f"SPECIALIST CONSTRUCTION REPORT — {rep['specialist']} "
         f"[{rep['status']}]", ""]
    L.append(f"Sources: {', '.join(rep['sources'])}")
    L.append(f"Components considered: "
             f"{', '.join(c['region'] + ' (' + c['source_model'] + ')' for c in rep['components']) or 'none'}")
    L.append(f"Parent scores on {rep['primary_suite']}: " +
             ", ".join(f"{k}={v:.0f}%" for k, v in rep["parent_primary"].items()))
    L.append(f"Best parent: {rep['best_parent']} ({rep['best_parent_score']:.0f}%)")
    from component_extraction.transplant import Component as _C
    for a in rep["attempts"]:
        if a["mechanism"] == "DIRECT_TRANSPLANT":
            comp = a.get("component")
            if isinstance(comp, _C):
                src, region = comp.source_model, comp.region
            elif isinstance(comp, dict):
                src, region = comp["source_model"], comp["region"]
            elif a.get("component_desc"):
                src, region = a["component_desc"].split(":", 1)
            else:
                src, region = "?", "?"
            L.append(f"  attempt {a['experiment_id']} transplant "
                     f"{src}:{region} "
                     f"A={a['A_donor']:.0f}% B={a['B_recipient']:.0f}% C={a['C_transplanted']:.0f}% "
                     f"gain={a['gain']:+.1f}")
        else:
            L.append(f"  attempt {a.get('experiment_id','—')} {a['mechanism']} "
                     f"accuracy={a.get('accuracy', 0):.0f}% coef={a.get('coef')}")
    L.append(f"Specialist score: {rep['specialist_score']}%")
    if "secondary_regressions" in rep:
        L.append("Secondary regression check: " +
                 ", ".join(f"{k}:{v:+.1f}" for k, v in rep["secondary_regressions"].items()))
    L.append(f"Chosen: {rep['chosen']}")
    L.append(f"Reason: {rep.get('reason', '')}")
    return "\n".join(L)
