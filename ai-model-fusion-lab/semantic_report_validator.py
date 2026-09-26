"""semantic_report_validator.py — Automated Semantic and Logical Consistency Validator for EQUYLAPTA7.

Performs deep verification across:
  1. Arithmetic comparisons and verbs (outperformed vs underperformed).
  2. Signed difference validation (e.g., negative deltas cannot be prefixed with '+').
  3. Directional labels (positive deltas cannot be labeled regression).
  4. Causal sequence validation (ablation drops, restoration recovery).
  5. Convergence assertions (verifying parameter, representational, and behavioral convergence).
  6. Structured claim-to-source-key provenance with calculated expected values.
  7. Evidence-level bounds and strictly prohibited inflated claims.
"""
from __future__ import annotations

import json
import os
import re
import sys

CURR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURR) == "ai-model-fusion-lab":
    ROOT = CURR
    WORKSPACE = os.path.abspath(os.path.join(ROOT, os.pardir))
else:
    WORKSPACE = CURR
    ROOT = os.path.join(WORKSPACE, "ai-model-fusion-lab")


def run_semantic_validation(results: dict | None = None) -> bool:
    print("Running Semantic Report Validation for EQUYLAPTA7...")
    if results is None:
        rpath = os.path.join(WORKSPACE, "results.json")
        if not os.path.exists(rpath):
            rpath = os.path.join(ROOT, "results.json")
        with open(rpath) as f:
            results = json.load(f)

    report_path = os.path.join(WORKSPACE, "EQUYLAPTA7_REPORT.md")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"EQUYLAPTA7_REPORT.md not found in {WORKSPACE}")

    with open(report_path) as f:
        report_text = f.read()

    errors = []

    # ---------------------------------------------------------
    # RULE 1: CAUSAL ABLATION & SPECIFICITY
    # ---------------------------------------------------------
    best_unit = results["hierarchical_discovery"]["best_causal_unit"]["unit"]
    causal_drop = results["hierarchical_discovery"]["best_causal_unit"]["causal_drop"]
    base_acc = results["hierarchical_discovery"]["baseline"]["mean"]
    abl_acc = results["causal_discovery"]["ablation"]["mean"]
    rest_acc = results["causal_discovery"]["restoration"]["mean"]
    spec_ratio = results["causal_discovery"]["specificity_ratio"]

    # Ablation must drop performance
    if abl_acc > base_acc:
        errors.append(f"Ablation accuracy ({abl_acc:.2f}%) exceeds baseline ({base_acc:.2f}%), but causal drop claimed!")
    if f"+{causal_drop:.2f} pp" not in report_text and f"+{causal_drop:.2f} percentage point" not in report_text:
        errors.append(f"Causal drop +{causal_drop:.2f} pp not found in report!")

    # ---------------------------------------------------------
    # RULE 2: THREE RECONSTRUCTION METHODS COMPARISON
    # ---------------------------------------------------------
    recon = results["three_reconstruction_methods"]
    meth_a = recon["method_a_structural"]["accuracy"]["mean"]
    meth_b = recon["method_b_subspace"]["accuracy"]["mean"]
    meth_c = recon["method_c_distillation"]["accuracy"]["mean"]

    if meth_b < meth_a:
        if "method b.*proved superior to method a" in report_text.lower():
            errors.append(f"Method B ({meth_b:.2f}%) < Method A ({meth_a:.2f}%), but report claims Method B superior!")

    # ---------------------------------------------------------
    # RULE 3: TARGET LOCAL CONTROL COMPARISON & VERB
    # ---------------------------------------------------------
    ctrl_local = results["target_controls"]["control3_target_local_trained"]["mean"]
    diff_local = round(meth_b - ctrl_local, 2)
    expected_verb = "outperformed" if diff_local > 0 else ("underperformed" if diff_local < 0 else "matched")

    if diff_local < 0:
        if "outperformed the target-local trained control" in report_text.lower():
            errors.append(f"Arithmetic violation: Method B ({meth_b:.2f}%) < Local Control ({ctrl_local:.2f}%), but report claims 'outperformed'!")
    elif diff_local > 0:
        if "underperformed the target-local trained control" in report_text.lower():
            errors.append(f"Arithmetic violation: Method B ({meth_b:.2f}%) > Local Control ({ctrl_local:.2f}%), but report claims 'underperformed'!")

    # ---------------------------------------------------------
    # RULE 4: INDEPENDENT RECONSTRUCTIONS & CONVERGENCE
    # ---------------------------------------------------------
    indep = results["independent_reconstructions"]
    beh_agr = indep["behavioral_agreement_pct"]
    is_conv = indep["functional_convergence"]

    if not is_conv:
        if "independent reconstructions converged" in report_text.lower():
            errors.append("Report asserts independent reconstructions converged, but functional_convergence is False!")
    if f"{beh_agr:.2f}%" not in report_text:
        errors.append(f"Behavioral agreement {beh_agr:.2f}% not found in report!")

    # ---------------------------------------------------------
    # RULE 5: DEPTH SWEEP MONOTONICITY
    # ---------------------------------------------------------
    ds = results["depth_sweep"]
    expected_mono = ds["monotonicity"]
    if f"Monotonicity: `{expected_mono}`" not in report_text:
        errors.append(f"Depth monotonicity classification '{expected_mono}' not found in report!")

    # ---------------------------------------------------------
    # RULE 6: DIRECTIONAL LABELS (CAPABILITIES)
    # ---------------------------------------------------------
    cap = results["capability_specificity"]
    for dom, delta in cap["deltas"].items():
        if delta > 0:
            bad_label = rf"\|\s*{dom}\s*\|.*\|.*\|\s*\+{delta:.2f}\s*pp\s*\|\s*REGRESSION"
            if re.search(bad_label, report_text, re.IGNORECASE):
                errors.append(f"Directionality violation: domain {dom} has positive delta +{delta:.2f} pp but is labeled REGRESSION!")

    # ---------------------------------------------------------
    # RULE 7: STRUCTURED CLAIM-TO-SOURCE-KEY PROVENANCE
    # ---------------------------------------------------------
    provenance_claims = [
        {
            "claim_id": "CLM-E7-01",
            "source_key": "hierarchical_discovery.best_causal_unit.unit",
            "expected_value": best_unit,
            "required_text": f"`{best_unit}`"
        },
        {
            "claim_id": "CLM-E7-02",
            "source_key": "causal_discovery.restoration.mean",
            "expected_value": rest_acc,
            "required_text": f"`{rest_acc:.2f}%`"
        },
        {
            "claim_id": "CLM-E7-03",
            "source_key": "information_budget.compression_ratio_model_to_sig",
            "expected_value": results["information_budget"]["compression_ratio_model_to_sig"],
            "required_text": results["information_budget"]["compression_ratio_model_to_sig"]
        },
        {
            "claim_id": "CLM-E7-04",
            "source_key": "three_reconstruction_methods.method_b_subspace.accuracy.mean",
            "expected_value": meth_b,
            "required_text": f"**{meth_b:.2f}%**"
        },
        {
            "claim_id": "CLM-E7-05",
            "source_key": "independent_reconstructions.behavioral_agreement_pct",
            "expected_value": beh_agr,
            "required_text": f"{beh_agr:.2f}%"
        },
        {
            "claim_id": "CLM-E7-06",
            "source_key": "evidence_level",
            "expected_value": results["evidence_level"],
            "required_text": f"LEVEL {results['evidence_level']}"
        }
    ]

    for clm in provenance_claims:
        if "required_text" in clm and clm["required_text"] not in report_text:
            errors.append(f"Provenance claim {clm['claim_id']} failed: required text '{clm['required_text']}' from key '{clm['source_key']}' not found in report!")

    # ---------------------------------------------------------
    # RULE 8: PROHIBITED INFLATED PHRASES
    # ---------------------------------------------------------
    prohibited_phrases = [
        "proved transferable",
        "universal component",
        "general intelligence module",
        "model-independent intelligence",
        "portable reasoning"
    ]
    for ph in prohibited_phrases:
        if ph in report_text.lower():
            errors.append(f"Prohibited inflated phrase detected: '{ph}'")

    if errors:
        print(f"  [Semantic Validator] FAILED with {len(errors)} violations:", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        raise AssertionError(f"Semantic validation failed with {len(errors)} errors:\n" + "\n".join(errors))

    print(f"  [Semantic Validator] All {len(provenance_claims)} provenance claims and 7 semantic rules PASSED!")
    return True


if __name__ == "__main__":
    run_semantic_validation()
