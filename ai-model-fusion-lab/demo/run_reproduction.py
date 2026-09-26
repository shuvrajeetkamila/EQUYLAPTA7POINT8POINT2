"""REPRODUCTION pass (M3 spec §23): re-run the matrix cells that produced a
positive or borderline verdict, with the FULL seed set [1,2,3] and n=40.
A SPECIALIZED_TRANSFER_EVIDENCE claim is only upgraded to REPRODUCED if it
repeats across independent runs. Everything is recorded either way.
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.environ.get("FUSIONLAB_DATA",
                      os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                          os.path.abspath(__file__))), os.pardir, "fusionlab_data")))
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def log(m=""):
    print(m, flush=True)


def main():
    from models.synthetic import (build_profile_corpus, load_model)
    from models.modular_corpora import modular_suite
    from experiments.tracker import ExperimentTracker
    from capability_genome.genome import CapabilityGenome
    from component_transfer.transfer import (run_transfer, specificity_screen,
                                             candidates_from_training_delta,
                                             component_from_genome_record)
    tracker = ExperimentTracker(DATA)
    B0 = load_model("nomath-base")
    A_math, A_code = load_model("A-math"), load_model("A-code")
    genomes = {n: CapabilityGenome.load(os.path.join(DATA, "genomes", f"{n}.json"))
               for n in ("A-math", "A-code", "A-reason", "math-wiz")}
    lineage = {"A-math": A_math, "A-code": A_code, "nomath-base": B0}

    train_texts = build_profile_corpus("math-wiz", 400, seed=31)[:300] + \
        [it.prompt + " " + it.options[it.answer]
         for d in ["algebra", "pattern"] for it in modular_suite(d, 50, "train").items]
    calib_texts = build_profile_corpus("math-wiz", 200, seed=31)[100:200] + \
        [it.prompt + " " + it.options[it.answer]
         for it in modular_suite("algebra", 20, "train").items]
    probe_texts = build_profile_corpus("nomath-base", 120, seed=4242)

    def best_component(src_name, cap, domain, src_model):
        cands = genomes[src_name].candidates(capability=cap, min_abs_delta=3.0,
                                             levels=["HEAD", "CHANNEL_GROUP", "MODULE"])
        cands = [c for c in cands if c.suite == f"micro-{domain}"]
        try:
            lin = json.load(open(os.path.join(DATA, "lineage.json")))
            if lin.get(src_name) == lin.get("nomath-base"):
                cands = cands + candidates_from_training_delta(
                    src_model, load_model("nomath-base"), cap, f"micro-{domain}", top_k=6)
        except Exception:
            pass
        entries = specificity_screen(src_model, cands, cap, domain, n=40, log=None)
        return entries[0] if entries else None

    runs = []
    cells = [("A-math", "A-code", "math", A_math, A_code, "REPRODUCTION of the single "
              "SPECIALIZED_TRANSFER_EVIDENCE cell from the canonical matrix"),
             ("A-math", "nomath-base", "math", A_math, B0,
              "headline pair, full seed set for comparison"),
             ("math-wiz", "nomath-base", "math", None, B0,
              "borderline INSUFFICIENT cell (independent lineage)")]
    for src_name, tgt_name, cap, src_m, tgt_m, why in cells:
        src_model = src_m or load_model(src_name)
        e = best_component(src_name, cap, "math" if cap == "math" else cap, src_model)
        if e is None:
            log(f"{src_name}->{tgt_name}: no candidate")
            continue
        log(f"\n=== {src_name} -> {tgt_name} [{cap}] — {why}")
        log(f"    component {e['component'].component_id} "
            f"(specificity {e['specificity']:+.1f})")
        R = run_transfer(src_model, tgt_m, e["component"], cap, "math",
                         ["code", "reason", "lang", "know", "multi", "agent"],
                         train_texts, calib_texts, tracker=tracker,
                         cfg={"n_items": 40, "eval_seeds": [1, 2, 3],
                              "refit_steps": 150, "probe_texts": probe_texts},
                         log=None)
        v = R["verdict"]
        runs.append({"source": src_name, "target": tgt_name, "capability": cap,
                     "component": e["component"].component_id, "why": why,
                     "verdict": v,
                     "arms": {k: (val or {}).get("mean")
                              for k, val in R["arms"].items()}})
        log(f"    {v.get('baseline')} -> {v.get('calibrated')} (delta {v.get('delta')}, "
            f"std {v.get('std')}, floor {v.get('sig_floor')}) {v.get('status')}")

    out = os.path.join(REPORTS, "REPRODUCTION.json")
    json.dump(runs, open(out, "w"), indent=1)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
