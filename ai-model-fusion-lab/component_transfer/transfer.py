"""The M3 canonical transfer experiment (spec §§9-16).

Runs the A->B transfer ladder with mandatory controls:

  TARGET_BASELINE  (held-out)
  DIRECT_TRANSFER            raw component insertion (same-dim only)
  ALIGNED_TRANSFER           activation-scale matched or projected insertion
  REFITTED_TRANSFER          + masked-gradient training of inserted slices
  REFITTED+CALIBRATED        + residual-gate calibration (calib split)
  RANDOM_HARVEST_CONTROL     same-size component from a random-init model
  CAPACITY_CONTROL           random-init slices, trained identically
                             (parameter-matched generic adapter)

Success logic (§10, §15): improvement on HELD-OUT data, beyond the random
control, and - for the strongest claim - beyond the parameter-matched
capacity control. Structured failure reasons are recorded, never hidden.

Label vocabulary (§16): DIRECT_TRANSFER | ALIGNED_TRANSFER | REFITTED_TRANSFER
| ADAPTER_TRANSFER | DISTILLED_TRANSFER | ROUTED_SPECIALIST.
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from ablation.ablator import ablate_and_eval
from benchmarking.harness import eval_suite
from benchmarking.suites import load_suite, micro_suite
from component_extraction.transplant import _parse_region
from .alignment import DimensionMismatchError, fit_alignment
from .apply import (extract_component_tensors, insert_component_tensors,
                    tensor_locations, snapshot, restore_snapshot)
from .calibration import calibrate, out_locations_of
from .component import ComponentRecord, new_component_id
from .refitting import refit_component_only, lowrank_factorize


# --------------------------------------------------------------------------
def component_from_genome_record(rec, model, capability: str) -> ComponentRecord:
    path = rec.component["path"]
    layer = rec.component["layer"]
    p = _parse_region(path)
    if p["level"] == "HEAD":
        ctype, head_ids, chan = "HEAD", [p["index"]], None
    elif p["level"] == "CHANNEL_GROUP":
        ctype, head_ids, chan = "CHANNEL_GROUP", [], [p["dims"]["lo"], p["dims"]["hi"]]
    elif p["level"] == "MODULE":
        ctype, head_ids, chan = ("MODULE_ATTN" if p["module"] == "attn" else "MODULE_MLP"), [], None
    else:
        raise ValueError(f"genome level {p['level']} not transferable as a fine-grained "
                         "component; use a MODULE/HEAD/CHANNEL_GROUP candidate")
    comp = ComponentRecord(
        component_id=new_component_id(capability, ctype),
        source_model=model.name, source_model_revision="local-trained",
        architecture=model.spec.arch, component_type=ctype, layer=layer,
        head_ids=head_ids, channel_range=chan,
        hidden_dimension=model.spec.hidden,
        input_dimension=model.spec.hidden,
        output_dimension=model.spec.hidden,
        tokenizer=f"micro-vocab-{model.spec.vocab}",
        capability_target=capability,
        baseline_score=rec.baseline, ablation_score=rec.intervened,
        restoration_score=rec.restored,
        confidence=rec.confidence, transfer_status="UNTESTED",
        dependencies=f"residual stream up to layer {layer}",
        normalization_requirements="output-side scale must match target residual stats",
        suite=rec.suite, n_examples=rec.n, seed=rec.seed,
        notes=f"nominated by {rec.intervention} on {rec.suite}"
              + (f" ({rec.notes})" if getattr(rec, "notes", "") else ""))
    from .apply import extract_component_tensors
    comp.save_tensors(extract_component_tensors(model, comp))
    return comp


def eval_heldout(model, domain: str, n: int, seeds: List[int]) -> Dict:
    """Held-out evaluation: fresh value draws per seed; mean ± std."""
    accs = []
    for s in seeds:
        r = eval_suite(model, micro_suite(domain, n=n, seed=9000 + s), max_items=n)
        accs.append(r["accuracy"] * 100)
    return {"mean": round(float(np.mean(accs)), 2),
            "std": round(float(np.std(accs)), 2),
            "per_seed": [round(a, 2) for a in accs], "n": n, "seeds": seeds}


def capability_vector(model, domains: List[str], n: int = 24,
                      seeds=(1234, 1235, 1236)) -> Dict[str, float]:
    """Multi-seed held-out micro-suite vector (§21: never a single draw)."""
    out = {}
    for d in domains:
        accs = [eval_suite(model, micro_suite(d, n=n, seed=s), max_items=n)["accuracy"] * 100
                for s in seeds]
        out[f"micro-{d}"] = round(sum(accs) / len(accs), 1)
    return out


def specificity_screen(source, candidates: List, capability: str,
                       heldout_domain: str, domains: Optional[List[str]] = None,
                       top_k: int = 8, n: int = 24, seed: int = 1234, log=None):
    """Re-rank genome candidates by CROSS-SUITE SPECIFICITY (M2 lesson, §5).

    A transferable component must matter for the target capability while NOT
    being generically critical: forward-time ablation of the component in the
    SOURCE must drop the target suite more than it drops any other suite.
    Generic fragility (any-ablation-kills-everything) is exactly what produced
    destructive interference in the first M3 dry run; this screen filters it.
    Uses the SAME ablation op (forward masks) that produced the genome.

    Returns list of dicts sorted by specificity = target_drop - max_other_drop:
      {record, component, target_drop, max_other_drop, other_drops, specificity}
    """
    domains = domains or ["math", "code", "reason", "lang", "know", "multi", "agent"]
    tgt_key = f"micro-{heldout_domain}"
    suite_names = [f"micro-{d}" for d in domains]

    def _controls(ref: dict):
        lvl, l = ref["level"], ref["layer"]
        if lvl == "LAYER":
            return {"skip_layers": [l]}
        if lvl == "MODULE":
            return {"skip_modules": {l: [ref.get("module") or "attn"]}}
        if lvl == "HEAD":
            h = ref.get("index")
            if h is None:
                h = int(ref["path"].split(".head")[-1])
            hm = np.ones((source.spec.layers, source.spec.heads), dtype=np.float32)
            hm[l, h] = 0.0
            return {"head_mask": hm}
        if lvl == "CHANNEL_GROUP":
            dims = (ref.get("dims") or {})
            lo, hi = dims.get("lo"), dims.get("hi")
            if lo is None:
                lo, hi = map(int, ref["path"].split(".chan[")[-1].rstrip("]").split(":"))
            m = np.ones(source.spec.mlp_hidden, dtype=np.float32)
            m[lo:hi] = 0.0
            return {"channel_mask": {l: m}}
        return None  # PARAM_REGION etc.: not screenable with forward masks

    def _vec(controls):
        res = ablate_and_eval(source, suite_names, max_items=n, **controls)
        return {sn: res[sn]["accuracy"] * 100 for sn in suite_names}

    base_vec = _vec({})
    out = []
    for rec in candidates[:top_k]:
        ref = rec.component
        controls = _controls(ref)
        if controls is None:
            if log:
                log(f"    screen {ref['path']:<20} SKIPPED (level {ref['level']} "
                    f"not forward-maskable)")
            continue
        comp = component_from_genome_record(rec, source, capability)
        try:
            vec = _vec(controls)
        except Exception as e:
            if log:
                log(f"    screen {ref['path']:<20} FAILED ({e})")
            continue
        target_drop = base_vec[tgt_key] - vec[tgt_key]
        other_drops = {k: base_vec[k] - vec[k] for k in base_vec if k != tgt_key}
        max_other = max(other_drops.values())
        entry = {"record": rec, "component": comp,
                 "target_drop": round(target_drop, 1),
                 "max_other_drop": round(max_other, 1),
                 "other_drops": {k: round(v, 1) for k, v in other_drops.items()},
                 "specificity": round(target_drop - max_other, 1)}
        out.append(entry)
        if log:
            log(f"    screen {ref['path']:<20} target_drop {target_drop:+6.1f} "
                f"max_other_drop {max_other:+6.1f} specificity {entry['specificity']:+6.1f}")
    out.sort(key=lambda d: -d["specificity"])
    return out


def _encode(train_texts, model, cap=32):
    enc = []
    for t in train_texts:
        ids = model.tokenizer.encode(t, max_len=cap)
        if len(ids) >= 4:
            enc.append(ids)
    return enc


def _fit_cross_dim_transforms(source, target, probe_texts, seed=0):
    """Paired-activation fits for projected cross-dim insertion (§7)."""
    from adapters.projection import collect_hidden
    hs = collect_hidden(source, probe_texts)
    ht = collect_hidden(target, probe_texts)
    n = min(len(hs), len(ht))
    # source -> target (for down/output side)
    t_st, res_st = fit_alignment(hs[:n], ht[:n], "linear", seed=seed)
    # target -> source (for up/input side): map target residual into source space
    t_ts, res_ts = fit_alignment(ht[:n], hs[:n], "linear", seed=seed)
    return {"U_in": t_ts["W"].T.copy(),          # [d_t, d_s]: source w -> target w
            "V_out": t_st["W"].copy(),           # [d_s, d_t]: source out -> target out
            "align_source_to_target": res_st.to_dict(),
            "align_target_to_source": res_ts.to_dict()}


def candidates_from_training_delta(specialist, base_model, capability: str,
                                   suite: str, top_k: int = 6,
                                   group: int = 32) -> List:
    """Same-lineage component identification by TRAINING-DELTA analysis (§5).

    For a specialist fine-tuned from a base with the SAME init, the parameter
    regions that moved most encode the new skill by construction — no ablation
    required to *nominate* them (ablation+restoration validation still happens
    in the specificity screen). Ranks attention heads and MLP channel groups
    by L2 displacement vs the base. These candidates carry rank evidence, not
    accuracy-point deltas; `delta` is 0 and must NOT be read as effect size.

    Returns genome-record-compatible namespaces (component/delta/confidence/...).
    """
    from types import SimpleNamespace
    ss, sb = specialist.state_dict(), base_model.state_dict()
    L = specialist.spec.layers
    H = specialist.spec.heads
    d = specialist.spec.hidden
    e = d // H
    F = specialist.spec.mlp_hidden

    def dv(key):
        if key in ss and key in sb:
            return (np.asarray(ss[key], dtype=np.float64)
                    - np.asarray(sb[key], dtype=np.float64))
        return None

    scored = []
    for l in range(L):
        dW, dO = dv(f"L{l}.attn.qkv.W"), dv(f"L{l}.attn.o.W")
        dB, dOB = dv(f"L{l}.attn.qkv.b"), dv(f"L{l}.attn.o.b")
        if dW is not None and dO is not None:
            for h in range(H):
                h0 = h * e
                nrm = (float(np.linalg.norm(dW[:, h0:h0 + e]))
                       + float(np.linalg.norm(dO[h0:h0 + e, :])))
                if dB is not None:
                    nrm += float(np.linalg.norm(dB[h0:h0 + e]))
                if dOB is not None:
                    nrm += float(np.linalg.norm(dOB[h0:h0 + e]))
                scored.append((nrm, {"level": "HEAD", "path": f"L{l}.attn.head{h}",
                                     "layer": l, "module": "attn", "index": h,
                                     "dims": {"head_dim": e}}))
        d1, d2 = dv(f"L{l}.mlp.1.W"), dv(f"L{l}.mlp.2.W")
        d1b = dv(f"L{l}.mlp.1.b")
        if d1 is not None and d2 is not None:
            for g0 in range(0, F, group):
                g1 = min(g0 + group, F)
                # channels are COLUMNS of mlp.1.W [d,F] and ROWS of mlp.2.W [F,d]
                nrm = (float(np.linalg.norm(d1[:, g0:g1]))
                       + float(np.linalg.norm(d2[g0:g1, :])))
                if d1b is not None:
                    nrm += float(np.linalg.norm(d1b[g0:g1]))
                scored.append((nrm, {"level": "CHANNEL_GROUP",
                                     "path": f"L{l}.mlp.chan[{g0}:{g1}]",
                                     "layer": l, "module": "mlp", "index": g0 // group,
                                     "dims": {"lo": g0, "hi": g1}}))
    scored.sort(key=lambda t: -t[0])
    out = []
    for nrm, ref in scored[:top_k]:
        out.append(SimpleNamespace(
            component=ref, capability=capability, suite=suite,
            intervention="training_delta", baseline=0.0, intervened=0.0,
            delta=0.0, restored=None, n=0, seed=0, confidence="medium",
            status="CANDIDATE",
            notes=f"training-delta norm {nrm:.3f} vs same-init base (rank evidence only)"))
    return out


def run_transfer(source, target, comp: ComponentRecord, capability: str,
                 heldout_domain: str, vector_domains: List[str],
                 train_texts: List[str], calib_texts: List[str],
                 tracker=None, cfg: Optional[Dict] = None, log=None) -> Dict:
    cfg = cfg or {}
    n_items = cfg.get("n_items", 40)
    eval_seeds = cfg.get("eval_seeds", [1, 2, 3])
    refit_steps = cfg.get("refit_steps", 150)
    refit_lr = cfg.get("refit_lr", 2e-3)
    lowrank_rank = cfg.get("lowrank_rank", 8)
    seed = cfg.get("seed", 7)
    run_id = "XFER-" + uuid.uuid4().hex[:8]
    R: Dict = {"run_id": run_id, "source": source.name, "target": target.name,
               "capability": capability, "heldout": f"micro-{heldout_domain}",
               "component": comp.to_dict(), "arms": {}, "failures": [],
               "n_items": n_items, "eval_seeds": list(eval_seeds),
               "verdict": {}, "label": "REFITTED_TRANSFER"}
    locs = tensor_locations(comp, target.spec)
    locations_full = {k: v for k, v in locs.items()}
    out_locs = out_locations_of(locs)

    def ev(model):
        return eval_heldout(model, heldout_domain, n_items, eval_seeds)

    def vector(model):
        return capability_vector(model, vector_domains)

    snap = snapshot(target)
    cross = None
    if target.spec.hidden != comp.hidden_dimension:
        try:
            cross = _fit_cross_dim_transforms(source, target,
                                              cfg.get("probe_texts", train_texts[:60]),
                                              seed=seed)
            R["alignment"] = {"U_in": cross["align_target_to_source"],
                              "V_out": cross["align_source_to_target"]}
            if log:
                log(f"    cross-dim alignment: s->t cos {res_sum(cross['align_source_to_target'])}, "
                    f"t->s cos {res_sum(cross['align_target_to_source'])}")
        except Exception as e:
            R["failures"].append({"arm": "ALIGNED", "reason":
                                  f"alignment fit failed: {e}"})
            target.load_state_dict(snap)
            R["verdict"] = {"status": "FAILED",
                            "reason": "cross-dimension alignment could not be fitted"}
            return R

    # ---------------- TARGET_BASELINE ----------------
    base = ev(target)
    R["arms"]["TARGET_BASELINE"] = base
    R["baseline_vector"] = vector(target)
    if log:
        log(f"    baseline {base['mean']:.1f}±{base['std']:.1f}")

    # ---------------- DIRECT_TRANSFER (raw) ----------------
    try:
        insert_component_tensors(target, comp, comp.load_tensors(), cross)
        R["arms"]["DIRECT_TRANSFER"] = ev(target)
        direct_vec = vector(target)
        R["direct_vector"] = direct_vec
    except DimensionMismatchError as e:
        R["arms"]["DIRECT_TRANSFER"] = None
        R["failures"].append({"arm": "DIRECT", "reason": str(e)})
        R["direct_vector"] = None
    finally:
        restore_snapshot(target, snap)

    # ---------------- tensors to insert (aligned form) ----------------
    tensors = comp.load_tensors()
    module_params = int(sum(int(np.prod(v.shape)) for v in tensors.values()))
    if cfg.get("lowrank", False):
        tensors, module_params, energy = lowrank_factorize(tensors, lowrank_rank, seed)
        R["lowrank"] = {"rank": lowrank_rank, "module_params": module_params,
                        "energy_kept": round(energy, 3)}

    # ---------------- ALIGNED_TRANSFER ----------------
    # same-dim: activation-scale matched via row-norm normalization (below with calib)
    # cross-dim: projected insertion IS the aligned arm
    try:
        insert_component_tensors(target, comp, tensors, cross)
        R["arms"]["ALIGNED_INSERT"] = ev(target)
    except DimensionMismatchError as e:
        R["arms"]["ALIGNED_INSERT"] = None
        R["failures"].append({"arm": "ALIGNED", "reason": str(e)})
    finally:
        restore_snapshot(target, snap)

    # ---------------- REFITTED_TRANSFER ----------------
    train_enc = _encode(train_texts, target)
    if len(train_enc) < 8:
        R["failures"].append({"arm": "REFITTED", "reason": "insufficient train sequences"})
        R["verdict"] = {"status": "FAILED", "reason": "no training data for refit"}
        return R
    try:
        insert_component_tensors(target, comp, tensors, cross)
    except DimensionMismatchError as e:
        R["failures"].append({"arm": "REFITTED", "reason": str(e)})
        R["verdict"] = {"status": "FAILED", "reason": str(e).splitlines()[0]}
        return R
    refit = refit_component_only(target, locs, train_enc, steps=refit_steps,
                                 lr=refit_lr, seed=seed, log=log)
    R["refit"] = refit.to_dict()
    R["arms"]["REFITTED_TRANSFER"] = ev(target)
    R["label"] = "REFITTED_TRANSFER"

    # ---------------- + CALIBRATED ----------------
    calib_enc = _encode(calib_texts, target)
    calib = calibrate(target, out_locs, calib_enc, "residual_gate")
    R["calibration"] = calib.to_dict()
    R["arms"]["CALIBRATED"] = ev(target)
    R["calibrated_vector"] = vector(target)
    restore_snapshot(target, snap)

    # ---------------- ADAPTER_TRANSFER (§16 mechanism: task arithmetic) ----
    # Replacement-based arms destroy the target's own module function; the
    # adapter interpolates: W' = (1-g)*W_target + g*W_source at the component
    # sites. Gate selection: target-calib CE + damage penalty on general-text
    # CE (both from CALIB/PROBE splits — held-out is never touched). A DISTINCT
    # label — never merged with the replacement ladder arms in reporting.
    g_best = None
    if cfg.get("adapter", True) and "lowrank" not in R:
        try:
            from .apply import adapter_insertion
            from .calibration import _ce_on
            guard_enc = _encode(cfg.get("guard_texts") or cfg.get("probe_texts", []),
                                target)
            guard0 = float(_ce_on(target, guard_enc)) if guard_enc else None
            grid = cfg.get("adapter_grid",
                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
            rows = {}
            for g in grid:
                adapter_insertion(target, comp, tensors, cross, g)
                lt = float(_ce_on(target, calib_enc))
                dmg = (max(0.0, float(_ce_on(target, guard_enc)) - guard0)
                       if guard_enc else 0.0)
                rows[g] = {"target_ce": round(lt, 4), "guard_damage": round(dmg, 4),
                           "score": round(lt + dmg, 4)}
                restore_snapshot(target, snap)
            g_best = min(rows, key=lambda g: rows[g]["score"])
            adapter_insertion(target, comp, tensors, cross, g_best)
            R["adapter"] = {"mechanism": "task_arithmetic (1-g)*W_t + g*W_s",
                            "gate": g_best, "grid": list(grid),
                            "guard": "general-text CE damage penalty (probe split)"
                            if guard_enc else "none (no guard texts supplied)",
                            "gate_search": {str(k): v for k, v in rows.items()}}
            R["arms"]["ADAPTER_TRANSFER"] = ev(target)
            R["adapter_vector"] = vector(target)
            if log:
                log(f"    adapter: gate {g_best} (target CE {rows[g_best]['target_ce']:.3f}, "
                    f"guard damage {rows[g_best]['guard_damage']:.3f}) "
                    f"held-out {R['arms']['ADAPTER_TRANSFER']['mean']:.1f}")
        except Exception as e:
            R["arms"]["ADAPTER_TRANSFER"] = None
            R["failures"].append({"arm": "ADAPTER", "reason": str(e)[:300]})
            g_best = None
        finally:
            restore_snapshot(target, snap)

    # ---------------- ADAPTER_REFITTED (distinct label) --------------------
    # refitting (§8) applied to the blended slices: adapter(g*) then masked-
    # gradient training of exactly the component sites on the train split.
    if cfg.get("adapter_refit", True) and g_best is not None and len(train_enc) >= 8:
        try:
            from .apply import adapter_insertion as _ains
            _ains(target, comp, tensors, cross, g_best)
            refit2 = refit_component_only(target, locs, train_enc,
                                          steps=refit_steps, lr=refit_lr,
                                          seed=seed + 1)
            R["adapter_refit"] = {**refit2.to_dict(), "gate": g_best}
            R["arms"]["ADAPTER_REFITTED"] = ev(target)
            R["adapter_refit_vector"] = vector(target)
            if log:
                log(f"    adapter+refit: held-out "
                    f"{R['arms']['ADAPTER_REFITTED']['mean']:.1f} "
                    f"(loss {refit2.initial_loss:.2f}->{refit2.final_loss:.2f})")
        except Exception as e:
            R["arms"]["ADAPTER_REFITTED"] = None
            R["failures"].append({"arm": "ADAPTER_REFIT", "reason": str(e)[:300]})
        finally:
            restore_snapshot(target, snap)

    # ---------------- RANDOM_HARVEST_CONTROL ----------------
    try:
        from models.synthetic import make_micro, master_tokenizer
        rnd = make_micro("random-harvest", d_model=source.spec.hidden,
                         n_layers=source.spec.layers, n_heads=source.spec.heads,
                         ctx_len=source.spec.ctx_len,
                         vocab_size=source.tokenizer.vocab_size,
                         tokenizer=source.tokenizer, seed=seed + 999)
        rnd_comp = ComponentRecord(**{**comp.to_dict(), "component_id": comp.component_id + "-rnd",
                                      "source_model": rnd.name})
        rnd_tensors = extract_component_tensors(rnd, comp)
        insert_component_tensors(target, comp, rnd_tensors, cross)
        refit_r = refit_component_only(target, locs, train_enc, steps=refit_steps,
                                       lr=refit_lr, seed=seed, log=None)
        calib_r = calibrate(target, out_locs, calib_enc, "residual_gate")
        R["arms"]["RANDOM_HARVEST_CONTROL"] = ev(target)
        R["random_control_refit_final_loss"] = refit_r.final_loss
    except Exception as e:
        R["arms"]["RANDOM_HARVEST_CONTROL"] = None
        R["failures"].append({"arm": "RANDOM_HARVEST_CONTROL", "reason": str(e)[:300]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- CAPACITY_CONTROL (parameter-matched generic adapter) ----
    try:
        rng = np.random.default_rng(seed + 555)
        rnd_tensors = {}
        for name, (tname, sl, role) in locs.items():
            shape = target.state_dict()[tname][sl].shape
            rnd_tensors[name] = (rng.normal(0, 0.02, shape)).astype(np.float32)
        insert_component_tensors(target, comp, rnd_tensors, cross)
        refit_c = refit_component_only(target, locs, train_enc, steps=refit_steps,
                                       lr=refit_lr, seed=seed, log=None)
        calib_c = calibrate(target, out_locs, calib_enc, "residual_gate")
        R["arms"]["CAPACITY_CONTROL"] = ev(target)
        R["capacity_control_refit_final_loss"] = refit_c.final_loss
    except Exception as e:
        R["arms"]["CAPACITY_CONTROL"] = None
        R["failures"].append({"arm": "CAPACITY_CONTROL", "reason": str(e)[:300]})
    finally:
        restore_snapshot(target, snap)

    # ---------------- verdict (§10, §15) ----------------
    base_m = R["arms"]["TARGET_BASELINE"]["mean"]
    # final system = better of the two labeled mechanisms (replacement ladder
    # vs adapter); both are always reported separately (§16)
    arm_opts = {k: R["arms"][k] for k in ("CALIBRATED", "ADAPTER_TRANSFER",
                                          "ADAPTER_REFITTED")
                if R["arms"].get(k)}
    final_arm = (max(arm_opts, key=lambda k: arm_opts[k]["mean"])
                 if arm_opts else None)
    R["final_arm"] = final_arm
    cal_m = arm_opts[final_arm]["mean"] if final_arm else None
    rnd_m = R["arms"]["RANDOM_HARVEST_CONTROL"]["mean"] if R["arms"].get("RANDOM_HARVEST_CONTROL") else None
    cap_m = R["arms"]["CAPACITY_CONTROL"]["mean"] if R["arms"].get("CAPACITY_CONTROL") else None
    delta = round(cal_m - base_m, 2) if cal_m is not None else None
    verdict = {"baseline": base_m, "calibrated": cal_m, "delta": delta,
               "final_arm": final_arm,
               "random_control": rnd_m, "capacity_control": cap_m}
    if cal_m is None:
        verdict["status"] = "FAILED"
    else:
        per_seed = R["arms"][final_arm]["per_seed"]
        n_per_seed = R.get("n_items")
        std = R["arms"][final_arm]["std"] or 0.0
        # §21 significance floor: at n=20/seed, 5 pts = 1 item; demand more than
        # noise (2 std) AND at least one full item per seed on average.
        sig_floor = max(5.0, 2.0 * std)
        consistent = all((p - base_m) > 0 for p in per_seed)
        positive = delta > 0
        beats_random = (rnd_m is None) or (cal_m > rnd_m)
        beats_capacity = (cap_m is None) or (cal_m > cap_m)
        verdict.update({"positive": positive, "beats_random": beats_random,
                        "beats_capacity": beats_capacity,
                        "consistent_across_seeds": consistent,
                        "sig_floor": sig_floor, "n_per_seed": n_per_seed,
                        "std": std})
        # capability-destruction check (§6 Test F) — computed first so it can
        # demote the top label (transfers that wreck other capabilities are
        # PARTIAL, never clean evidence).
        vec_key = {"CALIBRATED": "calibrated_vector",
                   "ADAPTER_TRANSFER": "adapter_vector",
                   "ADAPTER_REFITTED": "adapter_refit_vector"}[final_arm]
        dv = R.get(vec_key) or R["calibrated_vector"]
        bv = R["baseline_vector"]
        regressions = {k: round(dv[k] - bv[k], 1) for k in bv
                       if not k.endswith(f"micro-{heldout_domain}") and dv[k] - bv[k] < -10}
        verdict["non_target_regressions"] = regressions
        clean = not regressions
        meaningful = delta >= sig_floor
        if meaningful and consistent and beats_random and beats_capacity:
            verdict["status"] = ("SPECIALIZED_TRANSFER_EVIDENCE" if clean
                                 else "PARTIAL_TRANSFER")
        elif positive and beats_random:
            verdict["status"] = ("INSUFFICIENT_EVIDENCE" if not meaningful
                                 else "PARTIAL_TRANSFER")
        elif positive:
            verdict["status"] = "NO_EVIDENCE_OF_SPECIALIZED_TRANSFER"
        else:
            verdict["status"] = "NO_TRANSFER"
        if regressions and "INTERFERENCE" not in verdict["status"]:
            verdict["status"] += "+INTERFERENCE"
    R["verdict"] = verdict
    comp.transfer_score = delta
    st = verdict["status"]
    if st.startswith("SPECIALIZED_TRANSFER_EVIDENCE"):
        comp.transfer_status = "PROMISING"
    elif st.startswith("PARTIAL_TRANSFER"):
        comp.transfer_status = "PARTIAL"
    else:  # NO_TRANSFER / NO_EVIDENCE / INSUFFICIENT_EVIDENCE / FAILED
        comp.transfer_status = "FAILED"
    if st == "INSUFFICIENT_EVIDENCE":
        comp.notes = (comp.notes + "; " if comp.notes else "") + \
            f"insufficient evidence: delta {delta} < significance floor {verdict['sig_floor']}"
    comp.refit_method = "component_only"
    comp.calibration_method = R["calibration"]["method"] if "calibration" in R else "none"
    comp.alignment_method = "projected" if cross else "activation_scale"

    if tracker:
        tracker.record(exp_type="component-transfer",
                       config={"run_id": run_id, "source": source.name,
                               "target": target.name, "component": comp.component_id,
                               "ctype": comp.component_type, "layer": comp.layer,
                               "capability": capability,
                               "arms": {k: (v["mean"] if v else None)
                                        for k, v in R["arms"].items()},
                               "heldout": R["heldout"], "n": n_items,
                               "seeds": eval_seeds,
                               "refit_steps": refit_steps, "refit_lr": refit_lr},
                       sources=[source.name, target.name],
                       results={"baseline": base_m, "calibrated": cal_m,
                                "delta": delta, "random_control": rnd_m,
                                "capacity_control": cap_m,
                                "status": verdict["status"],
                                "final_arm": final_arm, "std": verdict.get("std"),
                                "per_seed": per_seed if cal_m is not None else None,
                                "sig_floor": verdict.get("sig_floor"),
                                "consistent_across_seeds": verdict.get("consistent_across_seeds"),
                                "non_target_regressions": verdict.get("non_target_regressions"),
                                "n_per_seed": n_per_seed},
                       target_suite=R["heldout"],
                       notes=f"label={R['label']}; module_params={module_params}; "
                             f"trainable={R.get('refit', {}).get('trainable_params')}",
                       license="Apache-2.0")
    comp.save_tensors(tensors)
    comp.save_meta()
    return R


def res_sum(d):
    return f"{d.get('aligned_similarity', float('nan')):.3f}"


def render_transfer_report(R: Dict) -> str:
    v = R["verdict"]
    L = [f"TRANSFER RUN {R['run_id']} — {R['source']} -> {R['target']} "
         f"[{v.get('status', '?')}]", ""]
    L.append(f"component: {R['component']['component_id']} "
             f"[{R['component']['component_type']} L{R['component']['layer']}"
             f"{' heads ' + str(R['component']['head_ids']) if R['component']['head_ids'] else ''}"
             f"{' chan ' + str(R['component']['channel_range']) if R['component']['channel_range'] else ''}]"
             f" capability={R['capability']}")
    L.append(f"held-out: {R['heldout']} (n={R['arms']['TARGET_BASELINE']['n']}, "
             f"seeds={R['arms']['TARGET_BASELINE']['seeds']})")
    L.append("")
    L.append("| system | held-out mean ± std |")
    L.append("|---|---|")
    for arm in ["TARGET_BASELINE", "DIRECT_TRANSFER", "ALIGNED_INSERT",
                "REFITTED_TRANSFER", "CALIBRATED", "ADAPTER_TRANSFER",
                "ADAPTER_REFITTED", "RANDOM_HARVEST_CONTROL", "CAPACITY_CONTROL"]:
        a = R["arms"].get(arm)
        marker = "  <-- final" if arm == v.get("final_arm") else ""
        L.append(f"| {arm}{marker} | {a['mean']:.1f} ± {a['std']:.1f} |" if a else
                 f"| {arm} | not executed |")
    L.append("")
    if "refit" in R:
        L.append(f"refit: {R['refit']['trainable_params']:,} trainable params "
                 f"({R['refit']['steps']} steps, loss {R['refit']['initial_loss']:.2f}"
                 f"->{R['refit']['final_loss']:.2f})")
    if "calibration" in R:
        c = R["calibration"]
        L.append(f"calibration: {c['method']} g={c.get('gate')} "
                 f"calib-loss {c['loss_before']:.3f}->{c['loss_after']:.3f}")
    if "adapter" in R:
        a = R["adapter"]
        L.append(f"adapter: {a['mechanism']} gate={a['gate']} guard={a['guard']}")
        for gk, gv in a["gate_search"].items():
            L.append(f"  gate {gk}: {gv}")
    if "adapter_refit" in R:
        ar = R["adapter_refit"]
        L.append(f"adapter+refit: gate {ar.get('gate')}, "
                 f"{ar['trainable_params']:,} params, loss "
                 f"{ar['initial_loss']:.2f}->{ar['final_loss']:.2f}")
    L.append(f"verdict: {v}")
    for f in R["failures"]:
        L.append(f"FAILURE [{f['arm']}]: {f['reason'][:200]}")
    return "\n".join(L)
