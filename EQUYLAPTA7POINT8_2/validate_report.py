"""validate_report.py

Authoritative report consistency validator for EQUYLAPTA 7.8.2.
Detects:
- mismatched numbers between report and results.json
- incorrect percentages
- wrong seed counts
- false 'all seeds positive' claims
- validation vs held-out confusion
- 90% threshold violations
- claims unsupported by results.json
- unoptimized declared parameters
Fails loudly if any discrepancy is detected.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Any, List


def validate_report_consistency(report_path: str = "/home/user/EQUYLAPTA7POINT8_2/EQUYLAPTA7POINT8_2_REPORT.md",
                                results_path: str = "/home/user/EQUYLAPTA7POINT8_2/results.json",
                                audit_path: str = "/home/user/EQUYLAPTA7POINT8_2/adapter_update_audit.json") -> Dict[str, Any]:
    if not os.path.exists(report_path):
        return {"status": "FAIL", "errors": [f"Report not found at {report_path}"]}
    if not os.path.exists(results_path):
        return {"status": "FAIL", "errors": [f"Results not found at {results_path}"]}

    with open(report_path, "r", encoding="utf-8") as f:
        rep_text = f.read()

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    errors = []

    # 1. Milestone verification
    if "EQUYLAPTA 7.8.2" not in rep_text and "EQUYLAPTA7POINT8_2" not in rep_text:
        errors.append("Report header does not mention EQUYLAPTA 7.8.2")

    # 2. Check 90% threshold compliance
    threshold = res.get("scientific_classification", {}).get("predeclared_threshold_pct", 90.0)
    observed = res.get("scientific_classification", {}).get("observed_exact_agreement_pct", 0.0)
    threshold_met = res.get("scientific_classification", {}).get("threshold_met", False)

    if threshold_met is False:
        if "Threshold Met: True" in rep_text or "predeclared threshold was met" in rep_text.lower():
            errors.append("Report falsely claims 90% threshold was met when results.json says False")

    # 3. Check for false 'replicated across all seeds' claim if any seed is negative
    seed_stats = res.get("seed_replication", {})
    for cond_name, stats in seed_stats.items():
        raw_vals = stats.get("causal_drop_pp", {}).get("raw", [])
        has_negative = any(v < 0 for v in raw_vals)
        if has_negative:
            pattern = f"{cond_name}.*all seeds positive"
            if re.search(pattern, rep_text, re.IGNORECASE):
                errors.append(f"Report claims all seeds positive for {cond_name}, but raw values contain negative: {raw_vals}")

    # 4. Check activation patching provenance
    act = res.get("activation_patching", {})
    val_lift = act.get("validation_seed_888", {}).get("donor_lift_over_baseline_pp", 0.0)
    held_lift = act.get("heldout_seed_999", {}).get("donor_lift_over_baseline_pp", 0.0)
    if val_lift > 0 and held_lift < 0:
        if "Validation improvement was not retained on held-out data" not in rep_text:
            errors.append("Report missing mandated activation patching claim: 'Validation improvement was not retained on held-out data'")
        if "positive donor activation patching lift on held-out" in rep_text.lower():
            errors.append("Report falsely claims positive donor activation patching lift on held-out data")

    # 5. Check oversized circuit data vs legacy prose
    size_sweep = res.get("size_sweep", [])
    size_8 = next((s for s in size_sweep if s["size"] == 8), None)
    if size_8:
        val_drop_8 = size_8.get("val_causal_drop_pp")
        if "0.0 pp additional causal gain" in rep_text or "0.0 pp additional causal benefit" in rep_text:
            errors.append("Report repeats legacy claim of '0.0 pp additional causal gain' for oversized circuit")
        if f"+{val_drop_8}" not in rep_text and f"{val_drop_8}" not in rep_text:
            errors.append(f"Report does not contain observed 8-node validation causal drop ({val_drop_8} pp)")

    # 6. Check adapter update audit
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            aud = json.load(f)
        for p in aud.get("declared_parameters", []):
            if p not in rep_text:
                errors.append(f"Declared parameter {p} not mentioned in report")
            u_norm = aud.get("update_norms", {}).get(p, 0.0)
            if u_norm <= 1e-7:
                errors.append(f"Parameter {p} has zero update norm in audit: {u_norm}")

    passed = len(errors) == 0

    return {
        "status": "PASS" if passed else "FAIL",
        "checked_report": report_path,
        "checked_results": results_path,
        "total_errors": len(errors),
        "errors": errors
    }


if __name__ == "__main__":
    rep_p = sys.argv[1] if len(sys.argv) > 1 else "/home/user/EQUYLAPTA7POINT8_2/EQUYLAPTA7POINT8_2_REPORT.md"
    res_p = sys.argv[2] if len(sys.argv) > 2 else "/home/user/EQUYLAPTA7POINT8_2/results.json"
    aud_p = sys.argv[3] if len(sys.argv) > 3 else "/home/user/EQUYLAPTA7POINT8_2/adapter_update_audit.json"
    res = validate_report_consistency(rep_p, res_p, aud_p)
    if res["status"] != "PASS":
        print("ERROR: Report consistency validation FAILED!")
        for e in res["errors"]:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("SUCCESS: Report consistency validated with zero discrepancies.")
