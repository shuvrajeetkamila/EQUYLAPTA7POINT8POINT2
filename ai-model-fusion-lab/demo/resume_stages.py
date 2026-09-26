"""Resume demo from stage 7b (same-base merge) + 9-11 after a crash."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def log(msg=""):
    print(msg, flush=True)


def main():
    from models.synthetic import (build_profile_corpus, data_dir, load_model,
                                  master_tokenizer, save_model, train_micro)
    from benchmarking.suites import all_micro_suites, load_suite
    from benchmarking.harness import eval_suite
    from merging.engine import benchmark_state_dict
    from experiments.tracker import ExperimentTracker
    from distillation.distill import distill, quality_filter
    from adapters.projection import align_models
    from licensing.registry import check_configuration
    from router.router import LearnedRouter, router_training_data
    from experts.specialist import Specialist, SuperModel

    tok = master_tokenizer()
    tracker = ExperimentTracker(data_dir())
    names = ["math-wiz", "code-smith", "logic-owl", "polyglot", "generalist", "weak-base"]
    models = {n: load_model(n) for n in names}
    suites = all_micro_suites(n=40)
    suite_names = [s.name for s in suites]

    log("\n" + "=" * 74)
    log("STAGE 7b: MERGING IN ITS VALID DOMAIN — same-base fine-tune pair [WORKING]")
    log("=" * 74)
    log("  training shared base (balanced corpus), then two narrow fine-tunes")
    base = train_micro("shared-base", "generalist", epochs=90, batch=128, lr=1e-2)
    import numpy as _np
    def finetune(model, texts, epochs=35, lr=5e-3):
        rng = _np.random.default_rng(3)
        enc = [model.tokenizer.encode(t, max_len=model.spec.ctx_len) for t in texts]
        enc = [e for e in enc if len(e) >= 4]
        opt = {}
        for ep in range(epochs):
            order = rng.permutation(len(enc))
            for bs in range(0, len(order), 128):
                seqs = [enc[i] for i in order[bs:bs+128]]
                L = max(len(s) for s in seqs)
                arr = _np.zeros((len(seqs), L), _np.int64)
                for r, s in enumerate(seqs):
                    arr[r, :len(s)] = s
                model.train_step(arr, None, lr * (1 - .5*ep/epochs), opt)
        return model
    finetune(ftm, build_profile_corpus("math-wiz", 1400, seed=31))
    finetune(ftc, build_profile_corpus("code-smith", 1400, seed=32))
    log(f"  ft-math vs base, ft-code vs base — now merging the fine-tune pair:")
    parents = {}
    for nm, mm in [("base", base), ("ft-math", ftm), ("ft-code", ftc)]:
        parents[nm] = {sn: eval_suite(mm, load_suite(sn), max_items=40)["accuracy"]
                       for sn in ["micro-math", "micro-code"]}
        log(f"    {nm}: " + " ".join(f"{k}={v*100:.0f}%" for k, v in parents[nm].items()))
    merged_sd = run_strategy("weighted_average", [ftm.state_dict(), ftc.state_dict()], [0.5, 0.5])
    res = benchmark_state_dict(base, merged_sd, "merged-ft-pair", ["micro-math", "micro-code"], 40)
    log(f"    merged(ft-math,ft-code) avg: " + " ".join(f"{k}={v*100:.0f}%" for k, v in res.items()))
    for method in ["slerp", "ties", "dare"]:
        sd = run_strategy(method, [ftm.state_dict(), ftc.state_dict()], [0.5, 0.5])
        r2 = benchmark_state_dict(base, sd, f"merged-{method}", ["micro-math", "micro-code"], 40)
        log(f"    {method}: " + " ".join(f"{k}={v*100:.0f}%" for k, v in r2.items()))
        tracker.record(exp_type="merge-same-base", config={"method": method, "coef": 0.5},
                       sources=["ft-math", "ft-code"],
                       results={k: round(v*100, 1) for k, v in r2.items()},
                       target_suite="micro-math", license="Apache-2.0")
    tracker.record(exp_type="merge-same-base", config={"method": "weighted_average", "coef": 0.5},
                   sources=["ft-math", "ft-code"],
                   results={k: round(v*100, 1) for k, v in res.items()},
                   target_suite="micro-math", license="Apache-2.0")

    log("\n" + "=" * 74)
    log("STAGE 9: DISTILLATION: teacher -> filtered data -> student [WORKING micro]")
    log("=" * 74)
    teacher = models["math-wiz"]
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
    tracker.record(exp_type="distillation", config={"teacher": teacher.name,
                   "filtered": len(kept)},
                   sources=[teacher.name],
                   results={"student_math": round(sm["accuracy"]*100, 1),
                            "teacher_math": round(tm["accuracy"]*100, 1)},
                   license="Apache-2.0")

    log("\nSTAGE 9b: ADAPTER / REPRESENTATION ALIGNMENT [WORKING micro]")
    al = align_models(models["math-wiz"], models["generalist"],
                      build_profile_corpus("generalist", 80, seed=888))
    log(f"  math-wiz -> generalist hidden alignment: cosine={al['cosine_alignment']}, R2={al['r2']}")

    log("\n" + "=" * 74)
    log("STAGE 10-11: ROUTER + SUPER MODEL + LEADERBOARD [WORKING]")
    log("=" * 74)
    X, Y = router_training_data()
    router = LearnedRouter()
    router.fit(X, Y)
    specialists = []
    for cap, pname in [("math", "math-wiz"), ("coding", "code-smith"),
                       ("reasoning", "logic-owl"), ("language", "polyglot"),
                       ("knowledge", "polyglot"), ("multilingual", "polyglot"),
                       ("agent", "generalist")]:
        sp = Specialist(cap, models[pname], provenance=f"source model {pname} "
                        "(merges rejected: independently-trained, see EXP log)")
        sp.measure_latency()
        specialists.append(sp)
    superm = SuperModel(specialists, router)
    for q in ["What is 23 + 45 ?", "Write a python function def add ( a , b ) :",
              "Ann is taller than Ben . Ben is taller than Cal . Who is tallest ?",
              "Write a Python program proving this mathematical theorem.",
              "Hello in French is", "To compute 12 * 9 use the"]:
        caps, conf = router.route(q)
        log(f"  route {q!r}\n    -> {', '.join(caps)}")
    sup = superm.evaluate(["micro-math", "micro-code", "micro-reason", "micro-lang"], max_items=40)
    log(f"  routed super-model: {sup['suites']}")
    lic = check_configuration([{"name": "all-micro-sources", "license": "Apache-2.0"},
                               {"name": "gpt2", "license": "MIT"}])
    log(f"  license: {lic['license_compatibility']} | attributions: {lic['required_attributions']}")
    lb = tracker.leaderboard()
    log(f"  {len(lb)} experiments recorded; top 10:")
    for row in lb[:10]:
        log(f"    {row['id']} [{row['type']}] {row['target'] or '-':<12} mean={row['score']}")


if __name__ == "__main__":
    main()
