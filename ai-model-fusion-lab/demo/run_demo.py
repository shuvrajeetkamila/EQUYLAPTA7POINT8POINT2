"""[WORKING] END-TO-END MILESTONE-1 DEMO (spec §27).

Runs the ENTIRE pipeline with micro models (and real HF models when
available), printing honest measurements at every stage:

  1. register + inspect            6. micro-segment extraction (transplant)
  2. autopsy                       7. merge experiments (4 algorithms)
  3. compatibility matrix          8. specialists (verified vs parents)
  4. capability benchmarking       9. distillation
  5. ablation/localization        10. router + super-model + global eval
                                  11. leaderboard + license checks

Exit criterion (§27): pipeline executes end-to-end and produces a
leaderboard of REAL numbers. Duration ~15-25 min CPU.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

STEPS = []


def log(msg=""):
    print(msg, flush=True)


def stage(n, title):
    log("\n" + "=" * 74)
    log(f"STAGE {n}: {title}")
    log("=" * 74)


def main(real_models: bool = True, quick: bool = False):
    from models.synthetic import (build_profile_corpus, data_dir, load_model,
                                  master_tokenizer, save_model, train_micro)
    from models.registry import Registry, compatibility_matrix
    from models.autopsy import autopsy, tree_map, activation_autopsy
    from models.base import Tokenizer
    from benchmarking.suites import all_micro_suites, freeze_private, load_suite
    from benchmarking.harness import eval_model, eval_suite, capability_matrix
    from ablation.ablator import localize_capabilities, contribution_report
    from merging.engine import (build_specialist, extract_segment_transplant,
                                merge_experiment)
    from distillation.distill import distill, quality_filter
    from adapters.projection import align_models
    from experiments.tracker import ExperimentTracker
    from licensing.registry import check_configuration
    from router.router import LearnedRouter, router_training_data
    from experts.specialist import Specialist, SuperModel

    t_start = time.time()
    suites = all_micro_suites(n=40)
    suite_names = [s.name for s in suites]
    freeze_private(suites)  # spec §16: frozen private holdout
    tracker = ExperimentTracker(os.path.join(data_dir()))
    reg = Registry(os.path.join(data_dir(), "registry.json"))
    tok = master_tokenizer()
    log(f"micro tokenizer vocab: {tok.vocab_size} | data dir: {data_dir()}")

    # optional real models -------------------------------------------------
    hf_handles = {}
    if real_models:
        stage(0, "REAL MODEL PATH (optional; falls back to synthetic if unavailable)")
        try:
            from models.backend_hf import HFHandle, hf_available
            for hf_id in ["gpt2", "EleutherAI/pythia-70m"]:
                if hf_available(hf_id):
                    try:
                        h = HFHandle(hf_id)
                        hf_handles[hf_id] = h
                        rec = reg.register(h, license_info="see model card")
                        log(f"  loaded {hf_id}: {h.param_count():,} params [REQUIRES EXTERNAL MODEL: OK]")
                    except Exception as e:
                        log(f"  {hf_id}: load failed ({e})")
                else:
                    log(f"  {hf_id}: unavailable offline -> skipped (synthetic path carries demo)")
        except Exception as e:
            log(f"  HF path unavailable: {e} -> proceeding with synthetic only")

    # ---------------- 1. source models ------------------------------------
    stage(1, "SOURCE MODELS: train capability micro-models [WORKING]")
    profiles = ["math-wiz", "code-smith", "logic-owl", "polyglot", "generalist", "weak-base"]
    epochs = 40 if quick else 220
    models = {}
    for pname in profiles:
        m = train_micro(pname, pname, epochs=epochs, batch=128, lr=1e-2,
                        d_model=64, n_layers=2, n_heads=4)
        save_model(m)
        reg.register(m, license_info="Apache-2.0")
        models[pname] = m
        log(f"  trained {pname} ({m.backend}, {m.param_count():,} params, "
            f"final val loss {m.train_history[-1]['val_loss']:.2f})")

    # ---------------- 2. autopsy ------------------------------------------
    stage(2, "MODEL AUTOPSY [WORKING]")
    rep = autopsy(models["math-wiz"])
    log(tree_map(rep))
    probe = tok.encode("12 + 14 = 26")[None, :]
    act = activation_autopsy(models["math-wiz"], probe)
    log(f"  activation probe: resid stds per layer = "
        f"{[round(act['layers'][l]['resid_std'],2) for l in act['layers']]}")

    # ---------------- 3. compatibility ------------------------------------
    stage(3, "ARCHITECTURE COMPATIBILITY ANALYSIS [WORKING]")
    for cm in compatibility_matrix(reg):
        if cm["pair"].startswith(("math-wiz + code-smith", "math-wiz + weak-base")) or "gpt2" in cm["pair"]:
            log(f"  {cm['pair']}: shape_ok={cm['shape_compatible']} "
                f"merge_compat={cm['merge_compatibility']} -> {cm['recommended_fusion']}")
    log("  (synthetic models share arch+tokenizer -> HIGH; cross-backend -> LOW, "
        "fallback to distillation/routing per spec §26)")

    # ---------------- 4. capability benchmarking --------------------------
    stage(4, "CAPABILITY DISCOVERY: held-out MC suites [WORKING]")
    results = {}
    for name, m in models.items():
        log(f" benchmarking {name} ...")
        results[name] = eval_model(m, suite_names, max_items=40)
    log("\nCAPABILITY MATRIX (accuracy %, n=40/suite, see CI in experiments db):")
    log(capability_matrix(results))
    if hf_handles:
        from benchmarking.text_suites import load_text_suite
        for hf_id, h in hf_handles.items():
            log(f" benchmarking real model {hf_id} on text suites ...")
            tn = ["text-math", "text-reasoning", "text-coding", "text-knowledge",
                  "text-multilingual", "text-agent"]
            tres = {}
            for t in tn:
                r = eval_suite(h, load_text_suite(t), max_items=15)
                tres[t] = r
                log(f"  {t}: {r['accuracy']*100:.0f}% (n={r['n']})")

    # ---------------- 5. ablation + localization --------------------------
    stage(5, "ABLATION + CAPABILITY LOCALIZATION [WORKING]")
    target = models["generalist"]
    suite_subset = ["micro-math", "micro-code", "micro-reason"]
    log(f" localizing {target.name} on {suite_subset}")
    local = localize_capabilities(target, suite_subset, max_items=40, log=log)
    log(contribution_report(local))
    tracker.record(exp_type="ablation", config={"model": target.name, "suites": suite_subset},
                   sources=[target.name], results=local["baseline"],
                   notes="layer/module ablation scan", license="Apache-2.0")

    # ---------------- 6. segment extraction -------------------------------
    stage(6, "MICRO-SEGMENT EXTRACTION: layer transplant [RESEARCH EXPERIMENT]")
    extract_segment_transplant(models["generalist"], models["math-wiz"],
                               models["weak-base"], 0, 2, ["micro-math", "micro-code"],
                               tracker, max_items=40, log=log)
    log("  (honest expectation: transplants often do NOT help; recorded either way)")

    # ---------------- 7. merge experiments --------------------------------
    stage(7, "MERGE LAB: 4 algorithms x coefficients [WORKING]")
    merge_experiment(models["generalist"], [models["math-wiz"], models["code-smith"]],
                     suite_names[:4], tracker, target_suite="micro-math",
                     max_items=40, log=log)

    # ---------------- 8. specialists --------------------------------------
    stage(8, "SPECIALIST GENERATION (verified vs parents) [WORKING]")
    spec_defs = [("math", ["math-wiz", "generalist"]),
                 ("coding", ["code-smith", "generalist"]),
                 ("reasoning", ["logic-owl", "generalist"])]
    cap_to_suite = {"math": "micro-math", "coding": "micro-code", "reasoning": "micro-reason"}
    specialists = []
    specialist_models = {}
    for cap, srcs in spec_defs:
        log(f" building {cap} specialist from {srcs}")
        out = build_specialist(models["generalist"], [models[s] for s in srcs],
                               cap_to_suite[cap], tracker, max_items=40, log=log)
        # specialist object uses the best parent-or-merge model that we can load:
        best_name = max(srcs, key=lambda s: out["parents"][s])
        sp_model = models[best_name]
        if out["best_candidate"] and out["verdict"].startswith("ACCEPTED"):
            # clone a fresh model so the source 'generalist' object stays intact
            from models.synthetic import make_micro
            from merging.strategies import run_strategy
            src_models = [models[s] for s in srcs]
            c = out["best_candidate"]["coef"]
            w = [1.0 - (c - 0.5), c] if len(src_models) == 2 else [1.0] + [c] * (len(src_models) - 1)
            sd = run_strategy(out["best_candidate"]["method"],
                              [mm.state_dict() for mm in src_models], w)
            sp_model = make_micro(f"specialist-{cap}", d_model=64, n_layers=2, n_heads=4,
                                  vocab_size=tok.vocab_size, tokenizer=tok, seed=99)
            sp_model.load_state_dict(sd)
            sp_model.profile = f"merged({out['best_candidate']['method']},{srcs})"
        sp = Specialist(cap, sp_model, score=out["best_score"],
                        provenance=f"merge of {srcs} ({out['verdict']})")
        sp.measure_latency()
        specialists.append(sp)
        specialist_models[cap] = sp_model
        log(f"  specialist[{cap}] = {sp_model.name} (target {out['best_score']}% vs parent "
            f"{out['best_parent_score']}%) [{out['verdict']}]")

    # add the other capability models as direct specialists
    for cap, pname in [("language", "polyglot"), ("knowledge", "polyglot"),
                       ("multilingual", "polyglot"), ("agent", "generalist")]:
        sp = Specialist(cap, models[pname], provenance=f"direct source model {pname}")
        sp.measure_latency()
        specialists.append(sp)

    # ---------------- 9. distillation -------------------------------------
    stage(9, "DISTILLATION: teacher -> filtered data -> student [WORKING micro]")
    teacher = specialist_models["math"]
    raw_texts = build_profile_corpus("math-wiz", 900, seed=555) + \
                build_profile_corpus("generalist", 300, seed=556)
    kept = quality_filter(teacher, raw_texts, min_conf=0.15)
    log(f"  quality filter: kept {len(kept)}/{len(raw_texts)} examples")
    student = train_micro("student-raw", "weak-base", epochs=20, batch=128, lr=1e-2,
                          texts=build_profile_corpus("weak-base", 1200, seed=777))
    out = distill(teacher, student, kept, epochs=30, log=log)
    log(f"  distilled student final loss {out['final_loss']} on {out['n_examples']} examples")
    sm = eval_suite(student, load_suite("micro-math"), max_items=40)
    tm = eval_suite(teacher, load_suite("micro-math"), max_items=40)
    log(f"  student math {sm['accuracy']*100:.0f}% vs teacher {tm['accuracy']*100:.0f}%")
    tracker.record(exp_type="distillation", config={"teacher": teacher.name, "filtered": len(kept)},
                   sources=[teacher.name], results={"student_math": sm["accuracy"] * 100,
                                                    "teacher_math": tm["accuracy"] * 100},
                   license="Apache-2.0")

    # ---------------- 9b. representation alignment ------------------------
    stage("9b", "ADAPTER / REPRESENTATION ALIGNMENT [WORKING micro]")
    al = align_models(models["math-wiz"], models["generalist"],
                      build_profile_corpus("generalist", 80, seed=888))
    log(f"  math-wiz -> generalist hidden alignment: cosine={al['cosine_alignment']}, "
        f"R2={al['r2']} (ridge projection fitted)")

    # ---------------- 10. router + super model ----------------------------
    stage(10, "ROUTER + SUPER MODEL [WORKING]")
    X, Y = router_training_data()
    router = LearnedRouter()
    router.fit(X, Y)
    superm = SuperModel(specialists, router)
    demo_queries = [
        "What is 23 + 45 ?",
        "Write a python function def add ( a , b ) :",
        "Ann is taller than Ben . Ben is taller than Cal . Who is tallest ?",
        "Write a Python program proving this mathematical theorem.",
        "Hello in French is",
        "To compute 12 * 9 use the",
    ]
    log(superm.demonstrate_routing(demo_queries))
    log(" evaluating routed super-model on held-out suites ...")
    sup_res = superm.evaluate(suite_names[:4], max_items=40)
    log(f"  super-model routed scores: {sup_res['suites']}")

    # single-best baseline for comparison (honest routing benefit check)
    best_single = {}
    for sn in suite_names[:4]:
        accs = {name: results[name][sn]["accuracy"] for name in models}
        best_single[sn] = round(max(accs.values()) * 100, 1)
    log(f"  best single source per suite: {best_single}")

    # ---------------- 11. leaderboard + licenses --------------------------
    stage(11, "LEADERBOARD + LICENSE CHECK [WORKING]")
    lb = tracker.leaderboard()
    log(f"  {len(lb)} experiments recorded; top 8 by mean score:")
    for row in lb[:8]:
        log(f"    {row['id']} [{row['type']}] {row['target'] or '-':<12} mean={row['score']}")
    lic = check_configuration([
        {"name": "math-wiz", "license": "Apache-2.0"},
        {"name": "code-smith", "license": "Apache-2.0"},
        {"name": "gpt2", "license": "MIT"},
        {"name": "claude-distill", "license": "proprietary-closed"},
    ])
    log(f"  license check (incl. a deliberately-closed source): "
        f"{lic['license_compatibility']}")
    log(f"  attribution: {lic['required_attributions']}")

    log("\n" + "=" * 74)
    log(f"PIPELINE COMPLETE in {time.time()-t_start:.0f}s — all stages executed. "
        f"Artifacts in {data_dir()}")
    log("=" * 74)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer epochs for smoke test")
    ap.add_argument("--no-real", action="store_true", help="skip real HF models")
    args = ap.parse_args()
    main(real_models=not args.no_real, quick=args.quick)
