"""src/classification_engine.py

Automated deterministic classification engine for EQUYLAPTA 7.8.2.
Ingests results.json (and adapter_update_audit.json) and outputs final_classification.json.
Enforces strict deterministic mapping rules with zero manual intervention:
- If functional agreement < 0.90 -> FAIL
- If any declared parameter update norm == 0 -> INVALID_OPTIMIZATION
- If causal drop <= random control -> INCONCLUSIVE / ARTIFACT
- If causal drop > random control AND held-out capability retained (lift > 0 over baseline) -> SUCCESS
- If causal drop > random control BUT held-out capability NOT retained -> PARTIAL_TRANSFER_LOCALIZED_ONLY
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any


def classify_results(results_path: str,
                     audit_path: str,
                     output_path: str = "final_classification.json") -> Dict[str, Any]:
    with open(results_path, "r") as f:
        results = json.load(f)

    # Load adapter update audit
    with open(audit_path, "r") as f:
        audit = json.load(f)

    heldout = results["heldout_seed_999"]["Conditions"]
    cond_e = heldout["Condition_E_Recipient_Reconstructed_FDC"]
    cond_f = heldout["Condition_F_Random_Matched"]
    cond_a = heldout["Condition_A_Host_Baseline"]

    func_agreement = cond_e["functional_agreement_pct"] / 100.0  # ratio
    cond_e_causal_drop = cond_e["causal_ablation_drop_pp"]
    cond_f_causal_drop = cond_f["causal_ablation_drop_pp"]
    cond_e_acc = cond_e["active_accuracy"]
    cond_a_acc = cond_a["active_accuracy"]
    cond_f_acc = cond_f["active_accuracy"]
    heldout_lift = cond_e_acc - cond_a_acc

    # Check parameter update norms
    declared_params = audit.get("declared_parameters", [])
    update_norms = audit.get("update_norms", {})
    zero_update_params = [p for p in declared_params if update_norms.get(p, 0.0) <= 1e-7]

    # Evaluate decision logic
    rules_triggered = []
    verdict = None
    detailed_justification = ""

    if len(zero_update_params) > 0:
        verdict = "INVALID_OPTIMIZATION"
        rules_triggered.append("RULE_ZERO_PARAMETER_UPDATE")
        detailed_justification = f"Declared parameters had zero update norm: {zero_update_params}"
    elif func_agreement < 0.90:
        verdict = "FAIL"
        rules_triggered.append("RULE_FUNCTIONAL_AGREEMENT_BELOW_90")
        detailed_justification = f"Functional agreement {func_agreement*100:.1f}% fell below the pre-registered 90.0% threshold."
    elif cond_e_causal_drop <= cond_f_causal_drop:
        verdict = "INCONCLUSIVE_ARTIFACT"
        rules_triggered.append("RULE_CAUSAL_DROP_LE_RANDOM")
        detailed_justification = f"Transplanted circuit causal drop ({cond_e_causal_drop:.1f} pp) did not exceed random control ({cond_f_causal_drop:.1f} pp)."
    elif heldout_lift > 0.0:
        verdict = "SUCCESS"
        rules_triggered.append("RULE_FULL_TRANSFER_SUCCESS")
        detailed_justification = (
            f"Transplanted circuit exhibited genuine causal mediation (drop {cond_e_causal_drop:.1f} pp vs random {cond_f_causal_drop:.1f} pp) "
            f"and positive capability transfer (+{heldout_lift:.1f} pp over baseline)."
        )
    else:
        verdict = "PARTIAL_TRANSFER_LOCALIZED_ONLY"
        rules_triggered.append("RULE_LOCALIZED_CAUSAL_MEDIATION_WITHOUT_CAPABILITY_LIFT")
        detailed_justification = (
            f"Transplanted circuit demonstrated localized causal mediation (ablation drop {cond_e_causal_drop:.1f} pp vs random {cond_f_causal_drop:.1f} pp), "
            f"satisfying functional effect alignment, but capability transfer was not retained on held-out test data "
            f"(active accuracy {cond_e_acc:.1f}% vs baseline {cond_a_acc:.1f}%, net lift {heldout_lift:.1f} pp; random control active acc {cond_f_acc:.1f}%)."
        )

    # 5-seed replication classification
    rep_summary = results.get("replication_5_seeds", {}).get("Condition_E_Recipient_Reconstructed_FDC", {})
    rep_status = rep_summary.get("sign_consistency_status", "UNKNOWN")

    # Activation patching claim consistency
    act_val_lift = results["activation_patching"]["validation_seed_888"]["donor_lift_over_baseline_pp"]
    act_held_lift = results["activation_patching"]["heldout_seed_999"]["donor_lift_over_baseline_pp"]
    if act_val_lift > 0 and act_held_lift < 0:
        act_claim = "Validation improvement was not retained on held-out data"
    elif act_val_lift > 0 and act_held_lift > 0:
        act_claim = "Donor activation patching improved accuracy across both validation and held-out data"
    else:
        act_claim = "Donor activation patching failed to produce positive capability lift"

    classification = {
        "milestone": "EQUYLAPTA_7.8.2",
        "verdict": verdict,
        "rules_triggered": rules_triggered,
        "detailed_justification": detailed_justification,
        "evidence_provenance": {
            "functional_agreement_ratio": round(func_agreement, 4),
            "functional_agreement_threshold": 0.90,
            "transplanted_causal_drop_pp": round(cond_e_causal_drop, 2),
            "random_control_causal_drop_pp": round(cond_f_causal_drop, 2),
            "transplanted_active_acc_pct": round(cond_e_acc, 2),
            "recipient_baseline_acc_pct": round(cond_a_acc, 2),
            "random_control_active_acc_pct": round(cond_f_acc, 2),
            "heldout_capability_lift_pp": round(heldout_lift, 2),
            "declared_parameters_count": len(declared_params),
            "zero_update_parameters": zero_update_params,
            "all_declared_parameters_updated": len(zero_update_params) == 0
        },
        "replication_classification": {
            "status": rep_status,
            "seed_values": rep_summary.get("seed_values", []),
            "mean": rep_summary.get("mean"),
            "median": rep_summary.get("median"),
            "positive_seed_count": rep_summary.get("positive_seed_count"),
            "negative_seed_count": rep_summary.get("negative_seed_count"),
            "zero_seed_count": rep_summary.get("zero_seed_count")
        },
        "activation_patching_provenance": {
            "validation_donor_lift_pp": round(act_val_lift, 2),
            "heldout_donor_lift_pp": round(act_held_lift, 2),
            "mandated_narrative_claim": act_claim
        }
    }

    with open(output_path, "w") as f:
        json.dump(classification, f, indent=2)

    return classification


if __name__ == "__main__":
    r_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    a_path = sys.argv[2] if len(sys.argv) > 2 else "adapter_update_audit.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "final_classification.json"
    c = classify_results(r_path, a_path, out_path)
    print("Classification Engine Output:")
    print("  Verdict:", c["verdict"])
    print("  Justification:", c["detailed_justification"])
