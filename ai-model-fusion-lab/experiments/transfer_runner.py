"""Automated transfer experiment runner (M3, spec §22).

Usage (integrated with the existing EQUYLAPTA CLI):
  python cli.py transfer --source math-wiz --target weak-base \
      --capability math --component auto \
      --alignment activation_stats --refit component_only \
      --calibration residual_gate

'--component auto' selects the strongest genome candidate for the capability.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(source: str, target: str, capability: str, component: str = "auto",
        alignment: str = "activation_stats", refit: str = "component_only",
        calibration: str = "residual_gate", n_items: int = 40,
        heldout: str = None, quick: bool = False, json_out: str = None):
    from models.synthetic import (build_profile_corpus, data_dir, load_model,
                                  make_micro, master_tokenizer, train_micro)
    from experiments.tracker import ExperimentTracker
    from capability_genome.genome import CapabilityGenome
    from component_transfer.transfer import (component_from_genome_record,
                                             run_transfer, render_transfer_report)

    DATA = data_dir()
    tracker = ExperimentTracker(DATA)
    src = load_model(source)
    tgt = load_model(target)
    same_lineage = os.path.exists(os.path.join(DATA, "lineage.json")) and \
        json.load(open(os.path.join(DATA, "lineage.json"))).get(source) == \
        json.load(open(os.path.join(DATA, "lineage.json"))).get(target)

    # --- component selection ---
    gpath = os.path.join(DATA, "genomes", f"{source}.json")
    if not os.path.exists(gpath):
        raise SystemExit(f"no genome for {source}; run discovery first (cli.py genome)")
    genome = CapabilityGenome.load(gpath)
    suite = heldout or f"micro-{capability if capability != 'coding' else 'code'}"
    cands = genome.candidates(capability=capability, min_abs_delta=1.0,
                              levels=["HEAD", "CHANNEL_GROUP", "MODULE"])
    cands = [c for c in cands if c.suite == suite]
    if not cands and component != "auto":
        raise SystemExit(f"no {capability} candidates in {source}'s genome")
    if component == "auto":
        # specificity-screened selection over genome-ablation candidates plus
        # (same-lineage) training-delta nominees — the same pipeline as the
        # canonical demo run
        from component_transfer.transfer import (specificity_screen,
                                                 candidates_from_training_delta)
        pool = list(cands)
        try:
            lin = json.load(open(os.path.join(DATA, "lineage.json")))
            if lin.get(source) and lin.get(source) == lin.get("nomath-base"):
                base = load_model("nomath-base")
                pool += candidates_from_training_delta(src, base, capability,
                                                       suite, top_k=6)
        except Exception:
            pass
        entries = specificity_screen(src, pool, capability,
                                     suite.replace("micro-", ""), log=print)
        if not entries:
            raise SystemExit("no screenable candidates for this capability")
        rec, comp = entries[0]["record"], entries[0]["component"]
        print(f"component: {comp.component_id} [{comp.component_type} "
              f"L{comp.layer}] specificity {entries[0]['specificity']:+.1f} "
              f"(screened over {len(pool)} nominated)")
    else:
        rec = next(c for c in cands if c.component["path"] == component)
        print(f"component: {rec.component['path']} (evidence {rec.delta:+.1f} pts, "
              f"{rec.confidence})")
        comp = component_from_genome_record(rec, src, capability)

    # --- data splits (never leak held-out into refit/calibration) ---
    from models.corpora import gen_train
    from models.modular_corpora import modular_suite
    dom_texts = build_profile_corpus(
        "math-wiz" if capability == "math" else
        "code-smith" if capability == "coding" else "logic-owl",
        500, seed=31)
    modular_map = {"math": ["algebra", "pattern"], "coding": ["cxform"],
                   "reasoning": ["pattern"]}
    train_texts = list(dom_texts[:350])
    calib_texts = list(dom_texts[350:450])
    for md in modular_map.get(capability, []):
        train_texts += [it.prompt + " " + it.options[it.answer]
                        for it in modular_suite(md, 60, "train").items]
        calib_texts += [it.prompt + " " + it.options[it.answer]
                        for it in modular_suite(md, 20, "train").items]

    cfg = {"n_items": 20 if quick else n_items,
           "eval_seeds": [1, 2] if quick else [1, 2, 3],
           "refit_steps": 80 if quick else 150,
           "probe_texts": dom_texts[:60],
           "train_texts_placeholder": None}
    R = run_transfer(src, tgt, comp, capability,
                     suite.replace("micro-", ""), ["code", "lang", "reason", "know"],
                     train_texts, calib_texts, tracker=tracker, cfg=cfg, log=print)
    print()
    print(render_transfer_report(R))
    if json_out:
        json.dump(R, open(json_out, "w"), indent=1, default=str)
        print(f"\nwrote {json_out}")
    return R


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--capability", default="math")
    ap.add_argument("--component", default="auto")
    ap.add_argument("--alignment", default="activation_stats")
    ap.add_argument("--refit", default="component_only")
    ap.add_argument("--calibration", default="residual_gate")
    ap.add_argument("--n-items", type=int, default=40)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    run(a.source, a.target, a.capability, a.component, a.alignment, a.refit,
        a.calibration, a.n_items, a.heldout, a.quick, a.json_out)
