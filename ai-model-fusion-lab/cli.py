"""EQUYLAPTA [AI MODEL] — Fusion Lab CLI (spec §21 + Milestone-2 commands).

Run from the project root:  python cli.py <command> ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_inspect(args):
    from models.registry import Registry
    from models.autopsy import autopsy, tree_map
    from models.synthetic import load_model, saved_names, data_dir
    reg = Registry(os.path.join(data_dir(), "registry.json"))
    if args.model.startswith(("gpt", "EleutherAI", "Qwen", "meta-llama", "mistralai", "google", "microsoft", "ibm")):
        rec = reg.register_hf_id(args.model)
        print(json.dumps(rec, indent=1)[:2000])
        return
    m = load_model(args.model)
    rep = autopsy(m)
    print(tree_map(rep))
    print(f"\nfingerprint: {m.fingerprint()}\nprofile: {getattr(m, 'profile', '?')}")


def cmd_benchmark(args):
    from models.synthetic import load_model
    from models.backend_hf import HFHandle
    from benchmarking.suites import all_micro_suites
    from benchmarking.harness import eval_model, capability_matrix
    from benchmarking.text_suites import load_text_suite
    m = HFHandle(args.model) if "/" in args.model or args.model == "gpt2" else load_model(args.model)
    if args.text:
        res = {}
        for t in ["text-math", "text-reasoning", "text-coding", "text-knowledge", "text-multilingual", "text-agent"]:
            from benchmarking.harness import eval_suite
            r = eval_suite(m, load_text_suite(t), max_items=args.n)
            res[t] = r
            print(f"  {t}: {r['accuracy']*100:.1f}%")
    else:
        res = eval_model(m, [s.name for s in all_micro_suites()], max_items=args.n,
                         log=print)
        print(capability_matrix({m.name: res}))


def cmd_ablate(args):
    from models.synthetic import load_model
    from ablation.ablator import localize_capabilities, contribution_report
    m = load_model(args.model)
    suites = args.suites.split(",") if args.suites else ["micro-math", "micro-code"]
    local = localize_capabilities(m, suites, max_items=args.n, log=print)
    print()
    print(contribution_report(local))


def cmd_discover(args):
    """Discover capability regions for one task (spec §21)."""
    from models.synthetic import load_model
    from ablation.ablator import localize_capabilities, contribution_report
    m = load_model(args.model)
    suite = f"micro-{args.task}"
    local = localize_capabilities(m, [suite], max_items=args.n, log=print)
    print()
    print(contribution_report(local))


def cmd_extract(args):
    from models.synthetic import load_model
    from merging.engine import extract_segment_transplant
    from experiments.tracker import ExperimentTracker
    from models.synthetic import data_dir
    donor, recipient = load_model(args.model), load_model(args.into)
    tr = ExperimentTracker(data_dir())
    lo, hi = (args.layers.split(":") if ":" in args.layers else [args.layers, str(int(args.layers) + 1)])
    out = extract_segment_transplant(recipient, donor, recipient, int(lo), int(hi),
                                     ["micro-math", "micro-code"], tr, log=print)
    print(out)


def cmd_merge(args):
    from models.synthetic import load_model, data_dir
    from merging.engine import merge_experiment
    from experiments.tracker import ExperimentTracker
    srcs = [load_model(n) for n in args.models]
    tr = ExperimentTracker(data_dir())
    out = merge_experiment(srcs[0], srcs, ["micro-math", "micro-code", "micro-reason"],
                           tr, target_suite=args.target, log=print)
    print("\nbest:", out[0] if out else "none")


def cmd_distill(args):
    from models.synthetic import load_model, make_micro, data_dir, build_profile_corpus
    from distillation.distill import distill, quality_filter
    teacher = load_model(args.teachers.split(",")[0])
    texts = build_profile_corpus("math-wiz", 600, seed=999) + build_profile_corpus("generalist", 200, seed=998)
    kept = quality_filter(teacher, texts)
    print(f"kept {len(kept)}/{len(texts)} after filtering")
    student = make_micro("distill-student", d_model=64, n_layers=2, n_heads=4,
                         vocab_size=teacher.tokenizer.vocab_size, tokenizer=teacher.tokenizer, seed=5)
    out = distill(teacher, student, kept, epochs=30, log=print)
    print(out["final_loss"])


def cmd_build_specialist(args):
    from models.synthetic import load_model, data_dir
    from merging.engine import build_specialist
    from experiments.tracker import ExperimentTracker
    suite = {"reasoning": "micro-reason", "coding": "micro-code", "math": "micro-math",
             "language": "micro-lang", "knowledge": "micro-know",
             "multilingual": "micro-multi", "agent": "micro-agent"}[args.capability]
    srcs = [load_model(n) for n in args.models.split(",")]
    tr = ExperimentTracker(data_dir())
    out = build_specialist(srcs[0], srcs, suite, tr, log=print)
    print(json.dumps({k: v for k, v in out.items() if k != "best_candidate"}, indent=1))


def cmd_evaluate(args):
    cmd_benchmark(args)


def cmd_license_check(args):
    from licensing.registry import check_configuration
    models = [{"name": n, "license": l} for n, l in
              (pair.split("=") for pair in args.models.split(","))]
    print(json.dumps(check_configuration(models), indent=1))



def cmd_genome(args):
    """Show the capability genome of a model (or all)."""
    import glob
    from capability_genome.genome import CapabilityGenome
    files = sorted(glob.glob(os.path.join(_data_dir(), "genomes", "*.json")))
    if args.model:
        files = [f for f in files if args.model in f]
    if not files:
        print("no genomes on disk; run: python cli.py m2-demo")
        return
    for f in files:
        print(CapabilityGenome.load(f).summary())
        print()


def cmd_gauntlet(args):
    """Architecture support gauntlet (LOAD->AUTOPSY->FORWARD->BENCH->ABLATE)."""
    from models.arch_gauntlet import run_all
    run_all(log=print, out_json=args.json)


def cmd_select_model(args):
    """Recommend real models that FIT this machine's RAM (no downloads)."""
    from models.selector import recommend, check_model_fits
    if args.model:
        import json as _j
        print(_j.dumps(check_model_fits(args.model), indent=1))
        return
    for r in recommend():
        mark = "OK " if r["fits"] else "NO "
        print(f"  [{mark}] {r['model']:<34} est {r.get('est_ram_gb','?'):>6}GB "
              f"vs {r.get('available_ram_gb',0):.1f}GB free | {r.get('license','?')}")


def cmd_transfer(args):
    from experiments.transfer_runner import run as _run
    _run(args.source, args.target, args.capability, args.component,
         args.alignment, args.refit, args.calibration, args.n_items,
         args.heldout, args.quick, args.json_out)


def cmd_m2_demo(args):
    """Full Milestone-2 pipeline (genomes, extraction, synthesis, evolution)."""
    from demo.run_milestone2 import main
    main(real_models=not args.no_real, quick=args.quick, retrain=args.retrain)


def _data_dir():
    from models.synthetic import data_dir
    return data_dir()

def cmd_demo(args):
    from demo.run_demo import main
    main(real_models=not args.no_real, quick=args.quick)


def cmd_serve(args):
    from app.server import serve
    serve(host="0.0.0.0", port=args.port)


def cmd_pattern_genome(args):
    """M4: pattern-genome registry summary, or run the canonical discovery."""
    if args.run:
        import demo.run_milestone4 as m4
        m4.main(quick=args.quick, skip_transfer=False)
        return
    from pattern_genome.registry import PatternRegistry
    from models.synthetic import data_dir
    path = os.path.join(data_dir(), "pattern_genome.json")
    if not os.path.exists(path):
        print("no pattern genome yet — run: python cli.py pattern-genome --run "
              "[--quick]")
        return
    reg = PatternRegistry.load(path)
    for line in reg.summary_lines():
        print(line)
    print("\nfull report: M4_REPORT.txt | demo/reports/PATTERN_GENOME_RESULTS.md")


def cmd_e4(args):
    """EQUYLAPTA4: cross-model functional transfer results summary."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results.json")
    if not os.path.exists(path):
        print("no E4 results yet — run: python3 demo/run_equylapta4.py")
        return
    r = json.load(open(path))
    fb = r.get("final_block")
    if not fb:
        print("E4 run incomplete — current depths:",
              [x.get("depth") for x in r.get("depth_sweep", [])])
        return
    for k, v in fb.items():
        print(f"{k}:")
        if isinstance(v, list):
            for item in v:
                print(f"  {item}")
        else:
            print(f"  {v}")


def cmd_e5(args):
    """EQUYLAPTA5: functional-identity decomposition results summary."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_e5.json")
    if not os.path.exists(path):
        print("no E5 results yet — run: python3 demo/run_equylapta5.py")
        return
    r = json.load(open(path))
    c = r.get("cells", {})
    if not c:
        print("E5 run incomplete")
        return
    print("EQUYLAPTA5 — what actually transfers? (held-out math, baseline "
          f"{c['baseline0']['accs_mean']})")
    for k in ("A_real_correct", "B_real_wrong_L1", "B_real_wrong_L2",
              "B_real_wrong_L3", "C_real_randloc_s21", "C_real_randloc_s22",
              "C_real_randloc_s23", "D_random_correct", "F_topology_only",
              "G_weight_stats_only", "H_representation_scrambled",
              "I_recon_matched", "I_recon_best", "J_function_destroyed",
              "K_function_restored"):
        if k in c:
            print(f"  {k:26s} {c[k]['accs_mean']:>6}  "
                  f"(gain {c[k].get('gain_vs_baseline', 0):+.2f})")
    print(f"  ladder rungs: {r.get('ladder_rungs')}")
    print(f"  FINAL CLASSIFICATION: {r.get('final_classification')}")
    print("  full report: EQUYLAPTA5_REPORT.txt")


def cmd_e6(args):
    """EQUYLAPTA6: cross-architecture functional translation summary."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "equylapta6_results.json")
    if not os.path.exists(path):
        print("no E6 results yet — run: python3 demo/run_equylapta6.py")
        return
    r = json.load(open(path))
    print("EQUYLAPTA6 — CROSS-ARCHITECTURE FUNCTIONAL TRANSLATION")
    print(f"  Source: {r['arch_a']['model_name']} (d={r['arch_a']['hidden']}, H={r['arch_a']['heads']})")
    print(f"  Target: {r['arch_b']['model_name']} (d={r['arch_b']['hidden']}, H={r['arch_b']['heads']})")
    print(f"  Target Baseline Math: {r['target_baseline_math']['mean']:.2f}%")
    print(f"  Naive Weight Control: {r['naive_weight_control']['status']}")
    print(f"  Random Target Control: {r['random_control_math']['mean']:.2f}% (collateral lang collapse to {r['random_control_vector']['lang']}%)")
    print(f"  Target-Trained Control: {r['target_trained_control_math']['mean']:.2f}%")
    print(f"  Translated Target Math: {r['translated_target_math']['mean']:.2f}% (gain: +{r['translated_target_math']['mean'] - r['target_baseline_math']['mean']:.2f}%)")
    print(f"  Functional Agreement: {r['functional_agreement']['agreement_pct']:.2f}% (Threshold >=90.0%: {r['functional_agreement']['verdict']})")
    print(f"  Independent Reconstruction Agreement: {r['independent_reconstruction']['convergence']['agreement_pct']:.2f}%")
    print(f"  EVIDENCE LADDER LEVEL: LEVEL {r['evidence_level']}")
    print(f"  FINAL CLASSIFICATION: {r['final_classification']}")
    print("  full report: EQUYLAPTA6_REPORT.txt")


def main():
    ap = argparse.ArgumentParser(prog="equylapta", description="EQUYLAPTA [AI MODEL] — AI Model Fusion Lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect"); p.add_argument("model"); p.set_defaults(f=cmd_inspect)
    p = sub.add_parser("benchmark"); p.add_argument("model"); p.add_argument("--n", type=int, default=40)
    p.add_argument("--text", action="store_true"); p.set_defaults(f=cmd_benchmark)
    p = sub.add_parser("ablate"); p.add_argument("model"); p.add_argument("--n", type=int, default=40)
    p.add_argument("--suites", default=None); p.set_defaults(f=cmd_ablate)
    p = sub.add_parser("discover"); p.add_argument("model"); p.add_argument("--task", default="math")
    p.add_argument("--n", type=int, default=40); p.set_defaults(f=cmd_discover)
    p = sub.add_parser("extract"); p.add_argument("model"); p.add_argument("--into", required=True)
    p.add_argument("--layers", default="0:2"); p.set_defaults(f=cmd_extract)
    p = sub.add_parser("merge"); p.add_argument("models", nargs="+")
    p.add_argument("--target", default="micro-math"); p.set_defaults(f=cmd_merge)
    p = sub.add_parser("distill"); p.add_argument("teachers"); p.set_defaults(f=cmd_distill)
    p = sub.add_parser("build-specialist"); p.add_argument("capability"); p.add_argument("--models", required=True)
    p.set_defaults(f=cmd_build_specialist)
    p = sub.add_parser("evaluate"); p.add_argument("model"); p.add_argument("--n", type=int, default=40)
    p.add_argument("--text", action="store_true"); p.set_defaults(f=cmd_evaluate)
    p = sub.add_parser("license-check"); p.add_argument("models")
    p.set_defaults(f=cmd_license_check)
    p = sub.add_parser("demo"); p.add_argument("--quick", action="store_true")
    p.add_argument("--no-real", action="store_true"); p.set_defaults(f=cmd_demo)
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000)
    p.set_defaults(f=cmd_serve)
    p = sub.add_parser("genome"); p.add_argument("model", nargs="?", default=None)
    p.set_defaults(f=cmd_genome)
    p = sub.add_parser("gauntlet"); p.add_argument("--json", default=None)
    p.set_defaults(f=cmd_gauntlet)
    p = sub.add_parser("select-model"); p.add_argument("--model", default=None)
    p.set_defaults(f=cmd_select_model)
    p = sub.add_parser("m2-demo"); p.add_argument("--quick", action="store_true")
    p.add_argument("--no-real", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.set_defaults(f=cmd_m2_demo)
    p = sub.add_parser("transfer"); p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--capability", default="math")
    p.add_argument("--component", default="auto")
    p.add_argument("--alignment", default="activation_stats")
    p.add_argument("--refit", default="component_only")
    p.add_argument("--calibration", default="residual_gate")
    p.add_argument("--n-items", type=int, default=40)
    p.add_argument("--heldout", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--json-out", default=None)
    p.set_defaults(f=cmd_transfer)

    pg = sub.add_parser("pattern-genome",
                        help="M4 pattern-genome: registry summary (or --run)")
    pg.add_argument("--run", action="store_true",
                    help="execute the canonical 8-stage discovery run")
    pg.add_argument("--quick", action="store_true",
                    help="with --run: fast weakened run (labeled as such)")
    pg.set_defaults(f=cmd_pattern_genome)

    e4p = sub.add_parser("e4", help="EQUYLAPTA4 transfer-sweep summary")
    e4p.set_defaults(f=cmd_e4)

    e5p = sub.add_parser("e5", help="EQUYLAPTA5 functional-identity summary")
    e5p.set_defaults(f=cmd_e5)

    e6p = sub.add_parser("e6", help="EQUYLAPTA6 cross-architecture functional translation summary")
    e6p.set_defaults(f=cmd_e6)

    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
