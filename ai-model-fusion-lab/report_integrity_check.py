"""report_integrity_check.py — Automated Report Integrity Enforcement for EQUYLAPTA7.

Verifies that every headline numerical claim in EQUYLAPTA7_REPORT.md maps
directly to an authoritative machine-readable source in results.json.
Fails with a non-zero exit code if any value cannot be verified.
"""
from __future__ import annotations

import json
import os
import sys

CURR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURR) == "ai-model-fusion-lab":
    ROOT = CURR
    WORKSPACE = os.path.abspath(os.path.join(ROOT, os.pardir))
else:
    WORKSPACE = CURR
    ROOT = os.path.join(WORKSPACE, "ai-model-fusion-lab")


def verify_report_integrity(results: dict) -> bool:
    print("Running Automated Report Integrity Audit for EQUYLAPTA7...")
    report_path = os.path.join(WORKSPACE, "EQUYLAPTA7_REPORT.md")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"EQUYLAPTA7_REPORT.md missing from {WORKSPACE}")

    with open(report_path) as f:
        report_text = f.read()

    hier = results["hierarchical_discovery"]
    causal = results["causal_discovery"]
    mini = results["minimality"]
    sig = results["functional_signature"]
    recon = results["three_reconstruction_methods"]
    ctrls = results["target_controls"]
    indep = results["independent_reconstructions"]
    ds = results["depth_sweep"]
    budget = results["information_budget"]

    assertions = [
        # Source discovery
        ("Source Baseline Acc", f"{hier['baseline']['mean']:.2f}%", "hierarchical_discovery.baseline.mean"),
        ("Best Causal Unit", f"`{hier['best_causal_unit']['unit']}`", "hierarchical_discovery.best_causal_unit.unit"),
        ("Source Causal Drop", f"+{hier['best_causal_unit']['causal_drop']:.2f} percentage point", "hierarchical_discovery.best_causal_unit.causal_drop"),
        ("Restoration Acc", f"`{causal['restoration']['mean']:.2f}%`", "causal_discovery.restoration.mean"),
        ("Specificity Ratio", f"`{causal['specificity_ratio']}x`", "causal_discovery.specificity_ratio"),

        # Minimality
        ("Minimality Fraction", f"`{int(mini['minimal_fraction']*100)}%`", "minimality.minimal_fraction"),

        # Signature & Budget
        ("Signature Bytes", f"`{sig['information_budget_bytes']} bytes`", "functional_signature.information_budget_bytes"),
        ("Covariance Trace", f"`{sig['covariance_trace']:.2f}`", "functional_signature.covariance_trace"),
        ("Compression Ratio", f"`{budget['compression_ratio_model_to_sig']}`", "information_budget.compression_ratio_model_to_sig"),
        ("Head Pct of Layer", f"{budget['functional_head_pct_of_layer']:.1f}%", "information_budget.functional_head_pct_of_layer"),

        # Target Reconstruction
        ("Target Baseline Acc", f"`{recon['target_baseline']['mean']:.2f}%`", "three_reconstruction_methods.target_baseline.mean"),
        ("Method A Acc", f"{recon['method_a_structural']['accuracy']['mean']:.2f}%", "three_reconstruction_methods.method_a_structural.accuracy.mean"),
        ("Method B Acc", f"**{recon['method_b_subspace']['accuracy']['mean']:.2f}%**", "three_reconstruction_methods.method_b_subspace.accuracy.mean"),
        ("Method C Acc", f"{recon['method_c_distillation']['accuracy']['mean']:.2f}%", "three_reconstruction_methods.method_c_distillation.accuracy.mean"),
        ("Method B Target Causal Drop", f"+{recon['method_b_subspace']['causal_drop']:.2f} pp", "three_reconstruction_methods.method_b_subspace.causal_drop"),

        # Independent Reconstructions
        ("Independent Mean Acc", f"**`{indep['mean_accuracy']:.2f}%`**", "independent_reconstructions.mean_accuracy"),
        ("Representational CKA", f"`{indep['representational_linear_cka']:.4f}`", "independent_reconstructions.representational_linear_cka"),
        ("Behavioral Agreement", f"`{indep['behavioral_agreement_pct']:.2f}%`", "independent_reconstructions.behavioral_agreement_pct"),

        # Status
        ("Evidence Level", f"LEVEL {results['evidence_level']}", "evidence_level"),
        ("Final Status", f"**{results['final_status']}**", "final_status"),
    ]

    # Controls
    for ck, cv in ctrls.items():
        assertions.append((f"Control {ck}", f"{cv['mean']:.2f}%", f"target_controls.{ck}.mean"))

    # Depth Sweep
    for d in ["2L", "4L", "6L", "8L"]:
        rw = ds["sweep"][d]
        assertions.append((f"{d} Target Base", f"{rw['target_baseline']['mean']:.2f}%", f"depth_sweep.sweep.{d}.target_baseline.mean"))
        assertions.append((f"{d} Target Recon", f"{rw['target_reconstructed']['mean']:.2f}%", f"depth_sweep.sweep.{d}.target_reconstructed.mean"))

    passed = 0
    failed = []
    for label, val_str, json_path in assertions:
        if val_str in report_text:
            passed += 1
        else:
            failed.append((label, val_str, json_path))

    print(f"  [Integrity Audit] Checked {len(assertions)} headline metrics: {passed} passed, {len(failed)} failed.")
    if failed:
        err_msg = "\n".join([f"  FAIL: Metric '{lbl}' expected string '{val}' from key '{k}' not found in report!" for lbl, val, k in failed])
        print(err_msg, file=sys.stderr)
        raise AssertionError(f"Report integrity check FAILED on {len(failed)} metrics!\n{err_msg}")

    print("  [Integrity Audit] All headline numbers verified against machine-readable results.")
    return True


if __name__ == "__main__":
    results_path = os.path.join(WORKSPACE, "results.json")
    if not os.path.exists(results_path):
        results_path = os.path.join(ROOT, "results.json")
    with open(results_path) as f:
        res = json.load(f)
    verify_report_integrity(res)
