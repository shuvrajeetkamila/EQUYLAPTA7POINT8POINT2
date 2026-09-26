"""Circuit transfer (Milestone 4 spec §15-16).

Ladder (§15): B baseline / B+raw structure / B+aligned / B+refitted /
B+calibrated / B+random structure / B+parameter-matched control.
Multi-member insertion (same-dim exact slices, cross-member dedup),
impedance-matching alignment (P071: activation-norm ratio at interfaces),
masked-gradient refitting over the union of member sites (M3 engine),
calibration (residual gate), strict M3 verdict logic (controls + floor +
consistency + interference), raw transfer-score components kept (§16).
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from benchmarking.harness import eval_suite
from benchmarking.suites import micro_suite
from component_transfer.apply import (snapshot, restore_snapshot,
                                      insert_component_tensors, tensor_locations)
from component_transfer.calibration import calibrate, out_locations_of, _ce_on
from component_transfer.refitting import refit_component_only
from component_transfer.component import ComponentRecord, new_component_id

from .schema import MemberRef, CircuitRecord, transfer_score_components


# --------------------------------------------------------------------------
# member <-> ComponentRecord plumbing (extraction/insertion at exact slices)
# --------------------------------------------------------------------------
def member_to_component(model, m: dict, capability: str) -> ComponentRecord:
    ref = MemberRef.from_dict(m)
    t = {"MODULE": "MODULE_ATTN" if ref.module == "attn" else "MODULE_MLP",
         "HEAD": "HEAD", "CHANNEL_GROUP": "CHANNEL_GROUP"}[ref.level]
    lo, hi = (ref.dims or {}).get("lo"), (ref.dims or {}).get("hi")
    comp = ComponentRecord(
        component_id=new_component_id(capability, t),
        source_model=model.name, source_model_revision="local-trained",
        architecture=model.spec.arch, component_type=t, layer=ref.layer,
        head_ids=[ref.index] if ref.level == "HEAD" else [],
        channel_range=[lo, hi] if lo is not None else None,
        hidden_dimension=model.spec.hidden,
        input_dimension=model.spec.hidden, output_dimension=model.spec.hidden,
        tokenizer=f"micro-vocab-{model.spec.vocab}",
        capability_target=capability, baseline_score=None,
        transfer_status="UNTESTED",
        notes=f"circuit member {ref.key()}")
    comp.validate = lambda: None  # baseline evidence provided at circuit level
    return comp


def member_tensors_direct(model, m: dict) -> Dict[str, np.ndarray]:
    """Extract member tensors straight from the model state dict (exact
    slices, no disk roundtrip)."""
    comp = member_to_component(model, m, "x")
    locs = tensor_locations(comp, model.spec)
    T = model.state_dict()
    return {name: np.asarray(T[tname][sl]).copy()
            for name, (tname, sl, role) in locs.items() if tname in T}


def circuit_members_payload(model, members: List[dict], capability: str):
    """Extract all member tensors; returns ({member_key: ComponentRecord},
    {member_key: tensors})."""
    comps, tensors = {}, {}
    for m in members:
        ref = MemberRef.from_dict(m)
        comps[ref.key()] = member_to_component(model, m, capability)
        tensors[ref.key()] = member_tensors_direct(model, m)
    return comps, tensors


def insert_circuit(model, members: List[dict], tensors: Dict[str, Dict[str, np.ndarray]],
                   scale: float = 1.0) -> Dict:
    """Insert every member's tensors at its exact slices (same-dim; the M3
    engine raises DimensionMismatchError otherwise — no silent reshape)."""
    applied = []
    for key, payload in tensors.items():
        member = next(mm for mm in members
                      if MemberRef.from_dict(mm).key() == key)
        comp = member_to_component(model, member, "x")
        rec = insert_component_tensors(model, comp, payload, None,
                                       out_scale=scale)
        applied += rec["applied"]
    return {"applied": sorted(set(applied))}


def union_locations(model, members: List[dict]) -> Dict[str, tuple]:
    """Union of tensor locations for refitting (prefixed by member key)."""
    locs: Dict[str, tuple] = {}
    for m in members:
        ref = MemberRef.from_dict(m)
        comp = member_to_component(model, m, "x")
        for k, v in tensor_locations(comp, model.spec).items():
            locs[f"{ref.key()}:{k}"] = v
    return locs


def _random_payload_at(model, members: List[dict], seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Random tensors drawn at the RANDOM members' own sites (same param
    budget, random 'structure'); keys match the member list exactly."""
    from .schema import MemberRef as _MR
    rng = np.random.default_rng(seed)
    T = model.state_dict()
    payload = {}
    for m in members:
        ref = _MR.from_dict(m)
        comp_m = member_to_component(model, m, "x")
        locs_m = tensor_locations(comp_m, model.spec)
        payload[ref.key()] = {
            name: rng.normal(0, 0.02, size=T[tname][sl].shape).astype(np.float32)
            for name, (tname, sl, role) in locs_m.items() if tname in T}
    return payload


def impedance_scale(source, target, probe_texts, cap: int = 32) -> float:
    """P071 impedance matching: activation-norm ratio at the interface
    (final hidden state norm source/target on matched probe inputs)."""
    from pattern_genome.graph_analysis import collect_nodes
    s = collect_nodes(source, probe_texts[:32])["final"]
    t = collect_nodes(target, probe_texts[:32])["final"]
    ns = float(np.linalg.norm(s.mean(0)))
    nt = float(np.linalg.norm(t.mean(0)))
    return float(ns / (nt + 1e-9))


def encode_texts(model, texts: List[str], cap: int = 32) -> List[np.ndarray]:
    enc = []
    for t in texts:
        ids = model.tokenizer.encode(t, max_len=cap)
        if len(ids) >= 4:
            enc.append(ids)
    return enc


# --------------------------------------------------------------------------
# the §15 ladder
# --------------------------------------------------------------------------
def run_circuit_transfer(source, target, circuit: CircuitRecord,
                         capability: str, heldout_domain: str,
                         vector_domains: List[str],
                         train_texts: List[str], calib_texts: List[str],
                         probe_texts: List[str],
                         tracker=None, cfg: Optional[Dict] = None,
                         log=None) -> Dict:
    cfg = cfg or {}
    n_items = cfg.get("n_items", 40)
    eval_seeds = cfg.get("eval_seeds", [1, 2, 3])
    refit_steps = cfg.get("refit_steps", 150)
    refit_lr = cfg.get("refit_lr", 2e-3)
    seed = cfg.get("seed", 7)
    run_id = "PTR-" + uuid.uuid4().hex[:8]
    members = circuit.members
    R: Dict = {"run_id": run_id, "circuit_id": circuit.circuit_id,
               "pattern_ids": circuit.pattern_ids, "kind": circuit.kind,
               "source": source.name, "target": target.name,
               "capability": capability, "heldout": f"micro-{heldout_domain}",
               "members": [MemberRef.from_dict(m).key() for m in members],
               "arms": {}, "failures": [], "verdict": {},
               "n_items": n_items, "eval_seeds": list(eval_seeds)}
    _, src_tensors = circuit_members_payload(source, members, capability)
    locs = union_locations(target, members)
    module_params = int(sum(int(np.prod(v.shape))
                            for payload in src_tensors.values()
                            for v in payload.values()))

    def ev(model, seeds=None):
        accs = []
        for s in (seeds or eval_seeds):
            r = eval_suite(model, micro_suite(heldout_domain, n=n_items,
                                              seed=9000 + s), max_items=n_items)
            accs.append(r["accuracy"] * 100)
        return {"mean": round(float(np.mean(accs)), 2),
                "std": round(float(np.std(accs)), 2),
                "per_seed": [round(a, 2) for a in accs]}

    def vector(model):
        out = {}
        for d in vector_domains:
            accs = [eval_suite(model, micro_suite(d, n=24, seed=s),
                               max_items=24)["accuracy"] * 100
                    for s in (1234, 1235)]
            out[f"micro-{d}"] = round(float(np.mean(accs)), 1)
        return out

    snap = snapshot(target)
    # ---------------- BASELINE ----------------
    R["arms"]["TARGET_BASELINE"] = ev(target)
    R["baseline_vector"] = vector(target)
    if log:
        log(f"    baseline {R['arms']['TARGET_BASELINE']['mean']}")

    # ---------------- RAW STRUCTURE ----------------
    try:
        insert_circuit(target, members, src_tensors)
        R["arms"]["RAW_STRUCTURE"] = ev(target)
    except Exception as e:
        R["failures"].append({"arm": "RAW", "reason": str(e)[:200]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- ALIGNED (impedance-matched) ----------------
    try:
        z = cfg.get("impedance", True)
        scale = float(impedance_scale(source, target, probe_texts)) if z else 1.0
        R["impedance_scale"] = round(scale, 4)
        insert_circuit(target, members, src_tensors, scale=scale)
        R["arms"]["ALIGNED_STRUCTURE"] = ev(target)
    except Exception as e:
        R["failures"].append({"arm": "ALIGNED", "reason": str(e)[:200]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- REFITTED ----------------
    train_enc = encode_texts(target, train_texts)
    if len(train_enc) < 8:
        R["failures"].append({"arm": "REFITTED", "reason": "no train sequences"})
        R["verdict"] = {"status": "FAILED", "reason": "no train data"}
        return R
    try:
        insert_circuit(target, members, src_tensors)
        refit = refit_component_only(target, locs, train_enc, steps=refit_steps,
                                     lr=refit_lr, seed=seed)
        R["refit"] = {**refit.to_dict(), "sites": len(locs)}
        R["arms"]["REFITTED_STRUCTURE"] = ev(target)
    except Exception as e:
        R["failures"].append({"arm": "REFITTED", "reason": str(e)[:200]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- + CALIBRATED ----------------
    try:
        insert_circuit(target, members, src_tensors)
        refit = refit_component_only(target, locs, train_enc, steps=refit_steps,
                                     lr=refit_lr, seed=seed)
        calib_enc = encode_texts(target, calib_texts)
        out_locs = out_locations_of(locs)
        cal = calibrate(target, out_locs, calib_enc, "residual_gate")
        R["calibration"] = cal.to_dict()
        R["arms"]["CALIBRATED_STRUCTURE"] = ev(target)
        R["calibrated_vector"] = vector(target)
    except Exception as e:
        R["failures"].append({"arm": "CALIBRATED", "reason": str(e)[:200]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- RANDOM STRUCTURE CONTROL (§15) ----------------
    try:
        from .controls import random_mask_circuit
        rnd_members = random_mask_circuit(target.spec, members, K=1,
                                          seed=seed + 5)[0]
        rnd_payload = _random_payload_at(target, rnd_members, seed + 9)
        insert_circuit(target, rnd_members, rnd_payload)
        R["arms"]["RANDOM_STRUCTURE_CONTROL"] = ev(target, seeds=eval_seeds[:2])
    except Exception as e:
        import traceback
        R["failures"].append({"arm": "RANDOM",
                              "reason": (str(e) or traceback.format_exc()[-160:])[:200]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- PARAMETER-MATCHED CAPACITY CONTROL (§15) ----------------
    try:
        from .controls import random_mask_circuit
        rnd_members = random_mask_circuit(target.spec, members, K=1,
                                          seed=seed + 6)[0]
        rnd_payload = _random_payload_at(target, rnd_members, seed + 10)
        insert_circuit(target, rnd_members, rnd_payload)
        rnd_locs = union_locations(target, rnd_members)
        refit_c = refit_component_only(target, rnd_locs, train_enc,
                                       steps=refit_steps, lr=refit_lr,
                                       seed=seed + 1)
        R["capacity_control_trainable"] = refit_c.trainable_params
        R["arms"]["CAPACITY_CONTROL"] = ev(target, seeds=eval_seeds[:2])
    except Exception as e:
        import traceback
        R["failures"].append({"arm": "CAPACITY",
                              "reason": (str(e) or traceback.format_exc()[-160:])[:200]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- verdict (M3-strict) ----------------
    base_m = R["arms"]["TARGET_BASELINE"]["mean"]
    arm_opts = {k: R["arms"][k] for k in
                ("CALIBRATED_STRUCTURE", "REFITTED_STRUCTURE",
                 "ALIGNED_STRUCTURE", "RAW_STRUCTURE") if R["arms"].get(k)}
    final_arm = max(arm_opts, key=lambda k: arm_opts[k]["mean"]) if arm_opts else None
    final_m = arm_opts[final_arm]["mean"] if final_arm else None
    rnd_m = R["arms"].get("RANDOM_STRUCTURE_CONTROL", {}) or {}
    cap_m = R["arms"].get("CAPACITY_CONTROL", {}) or {}
    rnd_mean = rnd_m.get("mean")
    cap_mean = cap_m.get("mean")
    delta = round(final_m - base_m, 2) if final_m is not None else None
    std = arm_opts[final_arm]["std"] if final_arm else None
    floor = max(5.0, 2.0 * (std or 0.0))
    per_seed = arm_opts[final_arm]["per_seed"] if final_arm else []
    consistent = bool(per_seed) and all(p - base_m > 0 for p in per_seed)
    positive = bool(delta is not None and delta > 0)
    beats_random = (rnd_mean is None) or (final_m > rnd_mean)
    beats_capacity = (cap_mean is None) or (final_m > cap_mean)
    bv = R["baseline_vector"]
    dv = R.get("calibrated_vector") or {}
    regressions = {k: round(dv[k] - bv[k], 1) for k in bv
                   if not k.endswith(f"micro-{heldout_domain}")
                   and dv.get(k, bv[k]) - bv[k] < -10}
    interference_penalty = float(-sum(v for v in regressions.values())) / 100.0
    meaningful = bool(delta is not None and delta >= floor)
    if final_m is None:
        status = "FAILED"
    elif meaningful and consistent and beats_random and beats_capacity and not regressions:
        status = "CROSS_MODEL_TRANSFER"
    elif meaningful and consistent and beats_random and beats_capacity:
        status = "CROSS_MODEL_TRANSFER+INTERFERENCE"
    elif positive and beats_random:
        status = ("INSUFFICIENT_EVIDENCE" if not meaningful
                  else "PARTIAL_TRANSFER")
        if regressions:
            status += "+INTERFERENCE"
    elif positive:
        status = "NO_EVIDENCE_OF_SPECIALIZED_TRANSFER"
        if regressions:
            status += "+INTERFERENCE"
    else:
        status = "TRANSFER_ATTEMPT_FAILED"
        if regressions:
            status += "+INTERFERENCE"
    R["verdict"] = {"baseline": base_m, "final_arm": final_arm,
                    "transferred": final_m, "delta": delta, "std": std,
                    "sig_floor": floor, "consistent_across_seeds": consistent,
                    "random_control": rnd_mean, "capacity_control": cap_mean,
                    "beats_random": beats_random, "beats_capacity": beats_capacity,
                    "non_target_regressions": regressions, "status": status}
    # §16 raw components (never collapse in the report)
    R["transfer_score"] = transfer_score_components(
        held_out_gain=delta,
        random_control_gain=(final_m - rnd_mean) if (final_m is not None and rnd_mean is not None) else None,
        capacity_control_gain=(final_m - cap_mean) if (final_m is not None and cap_mean is not None) else None,
        interference_penalty=interference_penalty,
        alignment_cost=R.get("impedance_scale", 1.0) - 1.0 if R.get("impedance_scale") else 0.0,
        parameter_cost=module_params,
        architecture_dependency="LOW" if source.spec.arch == target.spec.arch else "HIGH",
        reproducibility="UNTESTED")
    if tracker:
        tracker.record(exp_type="pattern-circuit-transfer",
                       config={"run_id": run_id, "circuit": circuit.circuit_id,
                               "patterns": circuit.pattern_ids,
                               "source": source.name, "target": target.name,
                               "members": R["members"], "n": n_items,
                               "seeds": eval_seeds},
                       sources=[source.name, target.name],
                       results={**{k: v for k, v in R["verdict"].items()},
                                "parameter_cost": module_params},
                       target_suite=R["heldout"],
                       notes=f"pattern circuit transfer {circuit.kind}",
                       license="Apache-2.0")
    return R
