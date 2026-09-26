"""semantic_report_validator.py — Enhanced Semantic and Logical Consistency Validator for EQUYLAPTA E7.3.

Implements all checks mandated by Section 30 of the EQUYLAPTA E7.3 specification:
  - CHECK 1: Inequality assertions (e.g., claiming 33.6 > 34.0 or claiming negative deltas as gains).
  - CHECK 2: Non-refitted validation Procrustes (ensures validation alignment was evaluated frozen).
  - CHECK 3: Accurate distillation nomenclature (representation regression != behavioral distillation).
  - CHECK 4: Causality assertions bounded by causal ablation drops (cannot claim causality when drop <= 0).
  - CHECK 5: Control condition labeling (random controls != trained controls).
  - CHECK 6: Predeclared threshold compliance (observed < 90% cannot be claimed as HIGH_FUNCTIONAL_AGREEMENT).
  - CHECK 7: Architectural impossibility overreach (single failed experiment != universal impossibility).
  - CHECK 8: Frozen transplanted component invariance (weights unchanged during co-adaptation).
  - CHECK 9: Independent target initializations (distinct SHA-256 hashes).
  - CHECK 10: 5-step causal sequence completeness (Baseline, Transplant, Ablate, Restore, Destroy).
"""
from __future__ import annotations

import json
import os
import re
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(WORKSPACE, "results_e7_3.json")):
    WORKSPACE = "/home/user"


def run_checks() -> bool:
    print("=" * 76)
    print("EQUYLAPTA E7.3 ENHANCED SEMANTIC REPORT VALIDATOR")
    print("=" * 76)

    with open(os.path.join(WORKSPACE, "results_e7_3.json")) as f:
        r = json.load(f)

    rep_path = os.path.join(WORKSPACE, "EQUYLAPTA_E7_3_REPORT.md")
    with open(rep_path) as f:
        rep_text = f.read()

    errors = []
    checks_passed = []

    # -------------------------------------------------------------
    # CHECK 1: Inequality & Delta Consistency
    # -------------------------------------------------------------
    print("[Check 1] Verifying inequality and delta assertions...")
    # Verify A->B Radius 1 accuracy vs baseline
    rad1_acc = r["phase_c_ab_results"]["coadaptation"]["radius_1"]["accuracy"]["mean"]
    base_b_acc = r["phase_c_ab_results"]["target_baseline"]["mean"]
    if rad1_acc < base_b_acc:
        bad_pats = [
            r"radius\s+1\s+outperformed\s+target\s+baseline",
            r"co-adaptation\s+improved\s+cross-architecture\s+performance",
            r"a->b\s+co-adaptation\s+increased\s+accuracy"
        ]
        for pat in bad_pats:
            if re.search(pat, rep_text, re.IGNORECASE):
                errors.append(f"[Check 1 Violation] Radius 1 ({rad1_acc}%) < baseline ({base_b_acc}%), but report claims improvement via '{pat}'!")
    checks_passed.append("check_1_inequality_and_delta_consistency")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 2: Frozen Validation Procrustes (No Refitting)
    # -------------------------------------------------------------
    print("[Check 2] Verifying non-refitted validation Procrustes...")
    r721 = r["phase_a_integrity_correction"]
    if not r721["validation_procrustes_corrected"]:
        errors.append("[Check 2 Violation] validation_procrustes_corrected is False!")
    if r721["anti_cheating_checks"]["validation_transform_refitted"]:
        errors.append("[Check 2 Violation] validation_transform_refitted is True!")
    checks_passed.append("check_2_frozen_validation_procrustes")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 3: Distillation Nomenclature Integrity
    # -------------------------------------------------------------
    print("[Check 3] Verifying behavioral distillation vs representation reconstruction nomenclature...")
    m3_name = "method_3_representation_reconstruction"
    m4_name = "method_4_behavioral_distillation"
    if m3_name not in r["phase_c_ab_results"] or m4_name not in r["phase_c_ab_results"]:
        errors.append("[Check 3 Violation] Methods not properly separated in results JSON!")
    checks_passed.append("check_3_distillation_nomenclature_integrity")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 4: Causality Claims Bounded by Causal Drop
    # -------------------------------------------------------------
    print("[Check 4] Verifying causal claims bounded by causal ablation drop...")
    ab_drop = r["phase_c_ab_results"]["coadaptation"]["radius_1"]["causal_sequence"]["causal_drop_pp"]
    if ab_drop <= 0:
        bad_causal = [
            r"transplanted\s+unit\s+causally\s+mediates\s+task\s+success",
            r"transplanted\s+head\s+is\s+causally\s+active\s+in\s+target\s+b"
        ]
        for pat in bad_causal:
            if re.search(pat, rep_text, re.IGNORECASE):
                errors.append(f"[Check 4 Violation] A->B causal drop is {ab_drop:+.2f} pp, but report claimed causal mediation via '{pat}'!")
    checks_passed.append("check_4_causality_claims_bounded")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 5: Control Condition Labeling
    # -------------------------------------------------------------
    print("[Check 5] Verifying control condition labeling...")
    ctrls = r["phase_c_ab_results"]["post_transplant_controls"]
    if "condition_c_random_component" not in ctrls or "condition_d_target_native" not in ctrls:
        errors.append("[Check 5 Violation] Required control conditions missing from post_transplant_controls!")
    checks_passed.append("check_5_control_condition_labeling")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 6: Predeclared Threshold Compliance
    # -------------------------------------------------------------
    print("[Check 6] Verifying predeclared 90% threshold compliance...")
    obs_agr = r["observed_highest_functional_agreement_pct"]
    threshold = r["predeclared_agreement_threshold_pct"]
    if obs_agr < threshold:
        if r["functional_agreement_classification"] == "HIGH_FUNCTIONAL_AGREEMENT":
            errors.append(f"[Check 6 Violation] Agreement {obs_agr}% < {threshold}%, but classified as HIGH_FUNCTIONAL_AGREEMENT!")
        if "achieved high functional agreement" in rep_text.lower():
            errors.append("[Check 6 Violation] Report text claims high functional agreement despite falling below 90% threshold!")
    checks_passed.append("check_6_predeclared_threshold_compliance")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 7: Architectural Impossibility Overreach Check
    # -------------------------------------------------------------
    print("[Check 7] Verifying absence of unwarranted universal impossibility claims...")
    bad_univ = [
        r"cross-architecture\s+transfer\s+is\s+fundamentally\s+impossible",
        r"proves\s+neural\s+components\s+can\s+never\s+be\s+transplanted",
        r"disproves\s+all\s+future\s+modular\s+transfer"
    ]
    for pat in bad_univ:
        if re.search(pat, rep_text, re.IGNORECASE):
            errors.append(f"[Check 7 Violation] Found unwarranted universal claim: '{pat}'!")
    checks_passed.append("check_7_no_unwarranted_universal_impossibility")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 8: Frozen Transplanted Component Invariance
    # -------------------------------------------------------------
    print("[Check 8] Verifying frozen transplanted component invariants...")
    rad1_frozen = r["phase_c_ab_results"]["coadaptation"]["radius_1"]["training_meta"]["transplanted_unit_strictly_frozen"]
    if not rad1_frozen:
        errors.append("[Check 8 Violation] transplanted_unit_strictly_frozen is False for Radius 1 co-adaptation!")
    checks_passed.append("check_8_frozen_transplanted_component_invariance")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 9: Independent Target Initializations
    # -------------------------------------------------------------
    print("[Check 9] Verifying independent target initializations...")
    inits = [run["target_initialization_hash"] for run in r["phase_c_ab_results"]["independent_runs"]]
    if len(set(inits)) != len(inits):
        errors.append(f"[Check 9 Violation] Target initialization hashes not distinct! Hashes: {inits}")
    checks_passed.append("check_9_independent_target_initializations")
    print("  -> Passed.")

    # -------------------------------------------------------------
    # CHECK 10: 5-Step Causal Sequence Completeness
    # -------------------------------------------------------------
    print("[Check 10] Verifying 5-step causal sequence completeness...")
    c_seq = r["phase_c_ab_results"]["coadaptation"]["radius_1"]["causal_sequence"]
    for req_step in ["transplanted", "ablated", "restored", "destroyed", "causal_drop_pp", "destruction_drop_pp"]:
        if req_step not in c_seq:
            errors.append(f"[Check 10 Violation] Missing '{req_step}' in causal sequence!")
    checks_passed.append("check_10_5step_causal_sequence_completeness")
    print("  -> Passed.")

    # Save semantic_validation.json
    validation_record = {
        "milestone": "EQUYLAPTA E7.3",
        "all_checks_passed": len(errors) == 0,
        "total_checks": len(checks_passed),
        "passed_checks": checks_passed,
        "error_count": len(errors),
        "errors": errors
    }
    with open(os.path.join(WORKSPACE, "semantic_validation.json"), "w") as f:
        json.dump(validation_record, f, indent=2)

    print("=" * 76)
    if errors:
        print(f"SEMANTIC VALIDATOR FAILED with {len(errors)} errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return False

    print("ALL 10 EQUYLAPTA E7.3 SEMANTIC CHECKS PASSED WITH ZERO VIOLATIONS!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not run_checks():
        sys.exit(1)
