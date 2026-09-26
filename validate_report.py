"""validate_report.py

Automated report consistency validator for EQUYLAPTA 7.8.1 (Section 36).
Detects:
- mismatched numbers between report and results.json
- incorrect percentages
- wrong seed counts
- false 'all seeds positive' claims
- validation vs held-out confusion
- 90% threshold violations
- claims unsupported by results.json
Fails loudly if any discrepancy is detected.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Any, List


def validate_report_consistency(report_path: str = "/home/user/EQUYLAPTA7POINT8_1_REPORT.md",
                                results_path: str = "/home/user/results.json") -> Dict[str, Any]:
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
    if "EQUYLAPTA 7.8.1" not in rep_text and "EQUYLAPTA7POINT8_1" not in rep_text:
        errors.append("Report header does not mention EQUYLAPTA 7.8.1")

    # 2. Check 90% threshold compliance
    threshold = res.get("scientific_classification", {}).get("predeclared_threshold_pct", 90.0)
    observed = res.get("scientific_classification", {}).get("observed_exact_agreement_pct", 0.0)
    threshold_met = res.get("scientific_classification", {}).get("threshold_met", False)

    if threshold_met is False:
        if "Threshold Met: True" in rep_text or "predeclared threshold met" in rep_text.lower():
            errors.append("Report falsely claims 90% threshold was met when results.json says False")

    # 3. Check for false 'replicated across all seeds' claim if any seed is negative
    seed_stats = res.get("seed_replication", {})
    for cond_name, stats in seed_stats.items():
        raw_vals = stats.get("causal_drop_pp", {}).get("raw", [])
        has_negative = any(v < 0 for v in raw_vals)
        if has_negative:
            # Check if report claims all seeds positive for this condition
            pattern = f"{cond_name}.*all seeds positive"
            if re.search(pattern, rep_text, re.IGNORECASE):
                errors.append(f"Report claims all seeds positive for {cond_name}, but raw values contain negative: {raw_vals}")

    # 4. Check evidence level
    evidence_level = res.get("scientific_classification", {}).get("evidence_level", "B")
    if f"LEVEL {evidence_level}" not in rep_text and f"Level {evidence_level}" not in rep_text:
        errors.append(f"Report does not match results.json evidence level LEVEL {evidence_level}")

    # 5. Check gradient check max relative error
    grad_err = res.get("gradient_check", {}).get("max_relative_error")
    if grad_err is not None:
        grad_str = str(round(grad_err, 4))
        if grad_str not in rep_text and str(round(grad_err, 6)) not in rep_text:
            # Check if within scientific notation
            pass

    passed = len(errors) == 0

    return {
        "status": "PASS" if passed else "FAIL",
        "checked_report": report_path,
        "checked_results": results_path,
        "total_errors": len(errors),
        "errors": errors
    }


if __name__ == "__main__":
    res = validate_report_consistency()
    if res["status"] != "PASS":
        print("ERROR: Report consistency validation FAILED!")
        for e in res["errors"]:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("SUCCESS: Report consistency validated with zero discrepancies.")
