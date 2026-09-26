"""MILESTONE 3 CANONICAL RUN — the real component transfer engine.

Central question (spec):
  Can an experimentally identified capability-associated component from
  Model A be transformed (aligned/refitted/calibrated) and transferred into
  Model B, producing measurable HELD-OUT capability improvement?

Design (honest by construction):
  * Target B0 "nomath-base": same init/lineage as sources, ZERO math training
    data -> genuine math deficit, intact other capabilities.
  * Sources A_math/A_code/A_reason: B0's exact weights + short capability
    fine-tunes (same-init lineage = the regime where transfer is possible).
  * Controls: random-harvest component (same size) + parameter-matched
    capacity adapter trained identically.
  * Evaluation: HELD-OUT micro-math (frozen private), 3 value-draw seeds;
    full capability vector for non-target interference (Test F).
  * ALSO runs the hard case: independent-lineage transfer (math-wiz -> B0)
    which M2 evidence predicts will fail — recorded either way.

Outputs: reports/*M3* + TRANSFER_MATRIX + FAILURE_ANALYSIS inputs.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

DATA = os.environ.get("FUSIONLAB_DATA",
                      os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                          os.path.abspath(__file__))), os.pardir, "fusionlab_data")))
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS, exist_ok=True)


def log(m=""):
    print(m, flush=True)


def stage(n, t):
    log("\n" + "=" * 74)
    log(f"M3-STAGE {n}: {t}")
    log("=" * 74)


def main(quick: bool = False, skip_real: bool = False):
    from models.synthetic import (build_profile_corpus, data_dir, finetune_from,
                                  load_model, make_micro, master_tokenizer,
                                  save_model, train_micro)
    from benchmarking.suites import micro_suite
    from models.corpora import gen_train
    from models.modular_corpora import modular_suite, MODULAR_DOMAINS
    from experiments.tracker import ExperimentTracker
    from capability_genome.discovery import discover
    from capability_genome.genome import CapabilityGenome
    from component_transfer.transfer import (component_from_genome_record,
                                             run_transfer, render_transfer_report,
                                             capability_vector, specificity_screen,
                                             candidates_from_training_delta)
    from benchmarking.harness import eval_suite

    t0 = time.time()
    tracker = ExperimentTracker(DATA)
    tok = master_tokenizer()
    S = 20 if quick else 40

    # ---------------- sources + deficit target ----------------
    stage(1, "LINEAGE SETUP: nomath target + same-init capability fine-tunes")
    marker = os.path.join(DATA, ".m3_lineage")
    need = not (os.path.exists(marker) and not quick)
    if need:
        B0 = train_micro("nomath-base", "nomath-base",
                         epochs=40 if quick else 150, batch=128, lr=1e-2,
                         d_model=64, n_layers=2, n_heads=4, log=log)
        save_model(B0)
        lineage = {"nomath-base": "lineage-A", "A-math": "lineage-A",
                   "A-code": "lineage-A", "A-reason": "lineage-A"}
        # Same-init fine-tunes from micro-format TRAIN seed draws (200-209) —
        # same generator family as eval suites but a DIFFERENT item draw;
        # eval seeds [1,2,3] and screen seeds [1234-1236] are never trained on
        # (in-distribution held-out protocol, labeled per §36). Recipes come
        # from a measured sweep: math needs 3e-3x45 to actually learn (55.6%),
        # code is already at ceiling (no recipe beat the base's 84.7), reason
        # gains at gentle 1e-3x30 (27.8 vs base 13.9). 500 general retention
        # texts slow catastrophic forgetting (does not prevent it — honest).
        retain = build_profile_corpus("nomath-base", 500, seed=77)
        def render_micro_train(domain, per_seed=120, seeds=range(200, 210)):
            out = []
            for sd in seeds:
                for it in micro_suite(domain, n=per_seed, seed=sd).items:
                    out.append(it.prompt + " " + it.options[it.answer])
            return out
        for name, dom, ep, lr in [("A-math", "math", 45, 3e-3),
                                  ("A-code", "code", 30, 1e-3),
                                  ("A-reason", "reason", 30, 1e-3)]:
            base = load_model("nomath-base")
            texts = render_micro_train(dom) + retain
            finetune_from(base, texts, epochs=ep, lr=lr, seed=7,
                          log=log if name == "A-math" else None)
            base.name = name
            base.profile = dom
            save_model(base)
            log(f"  fine-tuned {name} from nomath-base weights (lr 1e-3, {ep} ep, "
                f"+{len(retain)} retention texts)")
        json.dump(lineage, open(os.path.join(DATA, "lineage.json"), "w"))
        open(marker, "w").write("ok")
    B0 = load_model("nomath-base")
    A_math, A_code, A_reason = (load_model(n) for n in ("A-math", "A-code", "A-reason"))
    lineage_models = {"A-math": A_math, "A-code": A_code, "A-reason": A_reason,
                      "nomath-base": B0}
    # baseline capability snapshot + specialist-quality verification
    log("  capability signatures (SYNTHETIC models, held-out micro suites, 3-seed mean):")
    sigs = {}
    for n, m in [("nomath-base", B0), ("A-math", A_math),
                 ("A-code", A_code), ("A-reason", A_reason)]:
        vec = capability_vector(m, ["math", "code", "reason", "lang", "know", "multi", "agent"],
                                n=S)
        sigs[n] = vec
        log(f"    {n:<22} " + "  ".join(f"{k.replace('micro-','')}:{v}" for k, v in vec.items()))
    prof_cap = {"A-math": "math", "A-code": "code", "A-reason": "reason"}
    for n, capk in prof_cap.items():
        gain = sigs[n][f"micro-{capk}"] - sigs["nomath-base"][f"micro-{capk}"]
        if gain < 5:
            log(f"    WARNING: {n} gained only {gain:+.1f} on its own capability "
                f"({capk}) vs the nomath base — weak specialist; transfer headroom limited")
        else:
            log(f"    {n}: +{gain:.1f} on {capk} vs base — specialist gap confirmed")

    # ---------------- genomes for sources ----------------
    stage(2, "COMPONENT IDENTIFICATION (ablation + restoration on sources)")
    gspec = [("A-math", A_math, "math", "micro-math"),
             ("A-code", A_code, "coding", "micro-code"),
             ("A-reason", A_reason, "reasoning", "micro-reason")]
    genomes = {}
    for name, model, cap, suite in gspec:
        gd = discover(model, {cap: suite}, tracker, max_items=S, log=None)
        gd.save(os.path.join(DATA, "genomes", f"{name}.json"))
        genomes[name] = gd
        cands = gd.candidates(capability=cap, min_abs_delta=3.0,
                              levels=["HEAD", "CHANNEL_GROUP", "MODULE"])
        log(f"  {name}: {len(cands)} candidates; top: " +
            "; ".join(f"{c.component['path']} ({c.delta:+.1f})" for c in cands[:3]))
    # independent-lineage source genome (math-wiz from M1/M2)
    genomes["math-wiz"] = CapabilityGenome.load(os.path.join(DATA, "genomes", "math-wiz.json"))

    # ---------------- data splits ----------------
    train_texts = build_profile_corpus("math-wiz", 400, seed=31)[:300] + \
        [it.prompt + " " + it.options[it.answer]
         for d in ["algebra", "pattern"] for it in modular_suite(d, 50, "train").items]
    calib_texts = build_profile_corpus("math-wiz", 200, seed=31)[100:200] + \
        [it.prompt + " " + it.options[it.answer]
         for it in modular_suite("algebra", 20, "train").items]
    probe_texts = build_profile_corpus("nomath-base", 120, seed=4242)

    def _enc_probe(model, texts, cap=32):
        enc = []
        for t in texts:
            ids = model.tokenizer.encode(t, max_len=cap)
            if len(ids) >= 4:
                enc.append(ids)
        return enc

    # ---------------- HEADLINE: single-component ladder ----------------
    stage(3, "CANONICAL TRANSFER: A_math component -> nomath target (full ladder)")
    results = {}

    screen_store: dict = {}
    def best_component(src_name, cap, domain, verbose=False):
        """Genome + training-delta candidates -> specificity screen -> best."""
        key = (src_name, cap)
        if key in screen_store:
            return screen_store[key]
        g = genomes[src_name]
        cands = g.candidates(capability=cap, min_abs_delta=3.0,
                             levels=["HEAD", "CHANNEL_GROUP", "MODULE"])
        cands = [c for c in cands if c.suite == f"micro-{domain}"]
        src_model = lineage_models.get(src_name) or load_model(src_name)
        # second identification channel for same-init lineages: the tensors
        # that MOVED MOST during fine-tuning are the skill location by
        # construction (rank evidence; screened like everything else)
        base_m = None
        lin = json.load(open(os.path.join(DATA, "lineage.json"))) \
            if os.path.exists(os.path.join(DATA, "lineage.json")) else {}
        src_lin = lin.get(src_name)
        if src_lin and lin.get("nomath-base") == src_lin:
            try:
                base_m = load_model("nomath-base")
                cands = cands + candidates_from_training_delta(
                    src_model, base_m, cap, f"micro-{domain}", top_k=6)
            except Exception as e:
                log(f"  training-delta channel unavailable for {src_name}: {e}")
        seen, pool = set(), []
        for c in cands:
            pth = c.component["path"]
            if pth not in seen:
                seen.add(pth)
                pool.append(c)
        if not pool:
            screen_store[key] = None
            return None
        if verbose:
            log(f"  specificity screen: {src_name} [{cap}] over top "
                f"{min(10, len(pool))} candidates "
                f"({len(cands)} nominated: genome-ablation + training-delta)")
        entries = specificity_screen(src_model, pool, cap, domain, top_k=10, n=S,
                                     log=log if verbose else None)
        entry = entries[0] if entries else None
        if entry and verbose:
            rec = entry["record"]
            log(f"  selected {entry['component'].component_type} at "
                f"{entry['component'].component_id} "
                f"(genome delta {rec.delta:+.1f}, target_drop "
                f"{entry['target_drop']:+.1f}, max_other_drop "
                f"{entry['max_other_drop']:+.1f}, specificity "
                f"{entry['specificity']:+.1f})")
        screen_store[key] = entry
        return entry

    entry = best_component("A-math", "math", "math", verbose=True)
    if entry is None:
        log("  !! no specific transferable candidate found in A-math genome")
        return
    comp = entry["component"]
    cfg = {"n_items": S, "eval_seeds": [1, 2, 3], "refit_steps": 60 if quick else 150,
           "probe_texts": probe_texts, "lowrank": False}
    R = run_transfer(A_math, B0, comp, "math", "math",
                     ["code", "reason", "lang", "know", "multi", "agent"],
                     train_texts, calib_texts, tracker=tracker, cfg=cfg, log=log)
    results["headline_same_lineage"] = R
    open(os.path.join(REPORTS, "TRANSFER_HEADLINE.txt"), "w").write(render_transfer_report(R))
    log("\n" + render_transfer_report(R))

    # ---- diagnostic bound (NOT a component transfer; labeled as such) ----
    # full-model task arithmetic B0->A_math: the best ANY weight-space
    # interpolation (component or otherwise) could do on held-out math.
    from component_transfer.apply import snapshot as _snap, restore_snapshot as _rest
    from component_transfer.calibration import _ce_on
    log("\n  DIAGNOSTIC (not a component transfer): full-model interpolation bound")
    snapD = _snap(B0)
    sA = A_math.state_dict(); sB = B0.state_dict()
    guard_enc = _enc_probe(B0, probe_texts)
    guard0 = float(_ce_on(B0, guard_enc))
    diag = []
    for g in [0.2, 0.4, 0.6, 0.8, 1.0]:
        merged = {k: ((1 - g) * np.asarray(sB[k], np.float64)
                      + g * np.asarray(sA[k], np.float64)).astype(np.float32)
                  for k in sB}
        B0.load_state_dict(merged)
        ho = np.mean([eval_suite(B0, micro_suite("math", n=S, seed=9000 + s),
                                 max_items=S)["accuracy"] * 100 for s in (1, 2, 3)])
        gd = float(_ce_on(B0, guard_enc)) - guard0
        diag.append({"gate": g, "heldout_math": round(float(ho), 1),
                     "guard_ce_damage": round(gd, 4)})
        log(f"    g={g}: held-out math {ho:.1f} (guard CE damage {gd:+.3f})")
    _rest(B0, snapD)
    json.dump(diag, open(os.path.join(REPORTS, "interpolation_bound.json"), "w"), indent=1)

    # ---------------- transfer matrix (§11) ----------------
    stage(4, "TRANSFER MATRIX: bidirectional + third target + independent lineage")
    matrix = []
    def cell(src, tgt_model, cap, suite, label):
        e = best_component(src, cap, suite.replace("micro-", ""))
        if e is None:
            matrix.append({"source": src, "target": tgt_model.name, "capability": cap,
                           "status": "NO_CANDIDATE"})
            return
        src_model = lineage_models.get(src) or load_model(src)
        r = run_transfer(src_model, tgt_model, e["component"], cap,
                         suite.replace("micro-", ""), ["code", "reason", "lang"],
                         train_texts, calib_texts, tracker=tracker,
                         cfg={"n_items": S, "eval_seeds": [1, 2], "refit_steps": 60,
                              "probe_texts": probe_texts}, log=None)
        matrix.append({"source": src, "target": tgt_model.name, "capability": cap,
                       "component": e["component"].component_id,
                       "specificity": e["specificity"],
                       "baseline": r["verdict"].get("baseline"),
                       "transferred": r["verdict"].get("calibrated"),
                       "delta": r["verdict"].get("delta"),
                       "std": r["verdict"].get("std"),
                       "per_seed": r["arms"][r["final_arm"]]["per_seed"]
                       if r.get("final_arm") else None,
                       "sig_floor": r["verdict"].get("sig_floor"),
                       "random_control": r["verdict"].get("random_control"),
                       "capacity_control": r["verdict"].get("capacity_control"),
                       "status": r["verdict"].get("status"),
                       "method": "specificity-screened; aligned+refit+calibrated"})
        log(f"  {src} -> {tgt_model.name} [{cap}]: "
            f"{r['verdict'].get('baseline')} -> {r['verdict'].get('calibrated')} "
            f"(delta {r['verdict'].get('delta')}) {r['verdict'].get('status')}")
    cell("A-math", B0, "math", "micro-math", "lineage")
    cell("A-code", B0, "coding", "micro-code", "lineage")
    cell("A-reason", B0, "reasoning", "micro-reason", "lineage")
    cell("A-math", A_code, "math", "micro-math", "lineage-sibling")   # B -> C
    cell("math-wiz", B0, "math", "micro-math", "independent")         # hard case
    cell("A-math", load_model("weak-base"), "math", "micro-math", "independent-target")
    json.dump(matrix, open(os.path.join(REPORTS, "transfer_matrix.json"), "w"), indent=1)

    # ---------------- multi-component + interference (§12-13) ----------------
    stage(5, "MULTI-COMPONENT: singles, pairs, triple + interference matrix")
    from component_transfer.apply import (adapter_insertion, snapshot,
                                          restore_snapshot)
    from component_transfer.refitting import refit_component_only
    from component_transfer.apply import tensor_locations
    from component_transfer.calibration import _ce_on
    comps = {}
    for src_name, cap, dom in [("A-math", "math", "math"),
                               ("A-code", "coding", "code"),
                               ("A-reason", "reasoning", "reason")]:
        e = best_component(src_name, cap, dom)
        if e is not None:
            comps[cap] = e["component"]
    guard_enc = _enc_probe(B0, probe_texts
                           + render_micro_train("code", per_seed=20, seeds=[200, 201])
                           + render_micro_train("reason", per_seed=20, seeds=[200, 201]))
    guard0 = float(_ce_on(B0, guard_enc))
    target_enc = _enc_probe(B0, train_texts[:120])

    def guarded_multi_insert(cs, refit=False):
        """Greedy per-component guarded gates on calib splits (never held-out);
        optional joint refit of the inserted sites. Returns capability vector."""
        snap = snapshot(B0)
        gates = {}
        try:
            for c in cs:
                locs_c = tensor_locations(c, B0.spec)
                tens_c = c.load_tensors()
                best_g, best_score = 0.0, None
                for g in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    gates[c.component_id] = g
                    # apply all current gates from scratch
                    restore_snapshot(B0, snap)
                    for c2 in cs:
                        g2 = gates.get(c2.component_id, 0.0)
                        if g2 > 0:
                            adapter_insertion(B0, c2, c2.load_tensors(), None, g2)
                    lt = float(_ce_on(B0, target_enc))
                    dmg = max(0.0, float(_ce_on(B0, guard_enc)) - guard0)
                    score = lt + dmg
                    if best_score is None or score < best_score:
                        best_score, best_g = score, g
                gates[c.component_id] = best_g
            restore_snapshot(B0, snap)
            for c2 in cs:
                g2 = gates.get(c2.component_id, 0.0)
                if g2 > 0:
                    adapter_insertion(B0, c2, c2.load_tensors(), None, g2)
            if refit:
                train_enc = _enc_probe(B0, train_texts)
                if len(train_enc) >= 8:
                    all_locs = {}
                    for c2 in cs:
                        for k, v in tensor_locations(c2, B0.spec).items():
                            all_locs[f"{c2.component_id}:{k}"] = v
                    refit_component_only(B0, all_locs, train_enc, steps=60, lr=2e-3, seed=5)
            vec = capability_vector(B0, ["math", "code", "reason", "lang"], n=S)
        finally:
            restore_snapshot(B0, snap)
        return vec, gates
    def _enc(texts):
        return _enc_probe(B0, texts)
    base_vec = capability_vector(B0, ["math", "code", "reason", "lang"], n=S)
    singles, pairs = {}, {}
    for cap, c in comps.items():
        v, gates = guarded_multi_insert([c], refit=True)
        singles[cap] = {"vec": v, "gates": gates}
    keys = list(comps)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            v, gates = guarded_multi_insert([comps[keys[i]], comps[keys[j]]], refit=True)
            pairs[f"{keys[i]}+{keys[j]}"] = {"vec": v, "gates": gates}
    triple = None
    if len(comps) >= 3:
        v, gates = guarded_multi_insert(list(comps.values()), refit=True)
        triple = {"vec": v, "gates": gates}

    # ---- component-scaling study (§12): top-K training-delta channel ----
    # groups from the math specialist, ONE shared guarded gate. Measures
    # whether "more components = more skill" actually holds (it must be
    # measured, never assumed). All gates chosen on calib/probe splits only.
    log("  component-scaling study (top-K training-delta channel groups, "
        "shared guarded gate):")
    dcomps = candidates_from_training_delta(A_math, B0, "math", "micro-math",
                                            top_k=16, group=32)
    dchan = [component_from_genome_record(c, A_math, "math")
             for c in dcomps if c.component["level"] == "CHANNEL_GROUP"]
    scaling_snap = snapshot(B0)
    scaling = []
    try:
        for K in [1, 2, 4, 8]:
            cs = dchan[:K]
            if len(cs) < K:
                break
            best = None
            for g in [0.2, 0.4, 0.6, 0.8, 1.0]:
                restore_snapshot(B0, scaling_snap)
                for c in cs:
                    adapter_insertion(B0, c, c.load_tensors(), None, g)
                lt = float(_ce_on(B0, target_enc))
                dmg = max(0.0, float(_ce_on(B0, guard_enc)) - guard0)
                score = lt + 2.0 * dmg
                if best is None or score < best["score"]:
                    best = {"gate": g, "score": score}
            restore_snapshot(B0, scaling_snap)
            for c in cs:
                adapter_insertion(B0, c, c.load_tensors(), None, best["gate"])
            vec = capability_vector(B0, ["math", "code", "reason", "lang"], n=S)
            best.update({"K": K, "vec": vec,
                         "components": [c.component_id for c in cs]})
            scaling.append(best)
            log(f"    K={K} gate={best['gate']}: math {vec['micro-math']} "
                f"(base {base_vec['micro-math']}) code {vec['micro-code']} "
                f"reason {vec['micro-reason']}")
    finally:
        restore_snapshot(B0, scaling_snap)
    log(f"  baseline vector: {base_vec}")
    for cap, r in singles.items():
        v = r["vec"]
        log(f"  single {cap} {r['gates']}: math {v['micro-math']} (base {base_vec['micro-math']}) | "
            f"code {v['micro-code']} | reason {v['micro-reason']}")
    for k, r in pairs.items():
        v = r["vec"]
        log(f"  pair {k} {r['gates']}: math {v['micro-math']} code {v['micro-code']} reason {v['micro-reason']}")
    if triple:
        v = triple["vec"]
        log(f"  triple {triple['gates']}: math {v['micro-math']} code {v['micro-code']} reason {v['micro-reason']}")
    json.dump({"baseline": base_vec, "singles": singles, "pairs": pairs,
               "triple": triple, "scaling_study": scaling},
              open(os.path.join(REPORTS, "interference_matrix.json"), "w"), indent=1)

    # ---------------- real-model alignment experiment ----------------
    if not skip_real:
        stage(6, "REAL MODEL EXPERIMENT: gpt2 -> pythia-70m activation alignment")
        try:
            real_alignment(tracker, quick)
        except Exception as e:
            import traceback
            log(f"  REAL MODEL PHASE failed: {e}")
            log(traceback.format_exc()[-400:])

    # ---------------- genome/library updates + reports ----------------
    stage(7, "GENOME UPDATE + COMPONENT LIBRARY")
    from component_transfer.component import index_library
    lib = index_library()
    n_comp = sum(len(v) for v in lib.values())
    log(f"  component library: {n_comp} components across {len(lib)} capabilities")
    json.dump({"headline": {k: v for k, v in results["headline_same_lineage"].items()
                            if k != "component"},
               "matrix": matrix, "library_count": n_comp},
              open(os.path.join(REPORTS, "m3_summary.json"), "w"), indent=1, default=str)
    log(f"\nM3 CANONICAL RUN COMPLETE in {time.time()-t0:.0f}s")


def real_alignment(tracker, quick: bool):
    import numpy as np
    from models.backend_hf import HFHandle
    from component_transfer.alignment import fit_alignment
    from benchmarking.text_suites import build_text_suites
    gpt2 = HFHandle("gpt2")
    py = HFHandle("EleutherAI/pythia-70m")
    prompts = [it.prompt for s in build_text_suites(n=10) for it in s.items][:80]
    prompts = list(dict.fromkeys(prompts))[:60] or ["The capital of France is"]
    def pooled(h, texts):
        outs = []
        for t in texts:
            ids = h.tokenizer.encode(t, max_len=48)
            if len(ids) < 2:
                continue
            _, cache = h.forward(np.array(ids)[None, :], collect=True)
            outs.append(cache["final_hidden"][0].mean(0))
        return np.stack(outs)
    hs = pooled(gpt2, prompts)
    ht = pooled(py, prompts)
    n = min(len(hs), len(ht))
    res_rows = {"tokenizer_match": False,
                "token_alignment_method": "sentence-mean-pooling (token IDs differ; "
                "semantic sentence-level positions only)",
                "source": "gpt2 (REAL MODEL, d768)", "target": "pythia-70m (REAL MODEL, d512)",
                "n_pairs": int(n)}
    for method in ["linear", "low_rank", "learned"]:
        t, res = fit_alignment(hs[:n], ht[:n], method, seed=0)
        res_rows[method] = res.to_dict()
        log(f"  {method}: {res.summary()}")
    from models.registry import compatibility
    verdict = ("EXACT_COMPATIBLE: no | STRUCTURALLY_COMPATIBLE: no | ALIGNABLE: yes "
               "(activation space) | REFITTABLE: no (different archs + tokenizers; "
               "weight-level transfer impossible) | DISTILL_ONLY / ROUTING")
    res_rows["compatibility_verdict"] = verdict
    json.dump(res_rows, open(os.path.join(REPORTS, "real_model_alignment.json"),
                             "w"), indent=1)
    log(f"  verdict: {verdict}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, skip_real=a.skip_real)
