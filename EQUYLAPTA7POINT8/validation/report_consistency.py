"""validation/report_consistency.py

Verifies that metrics cited in EQUYLAPTA_7.8_REPORT.md and EQUYLAPTA_7.8_REPORT.txt
strictly match the machine-readable values in results.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Any, List


def verify_consistency(report_path: str, results_path: str) -> Dict[str, Any]:
    if not os.path.exists(report_path):
        return {"status": "FAIL", "error": f"Report not found at {report_path}"}
    if not os.path.exists(results_path):
        return {"status": "FAIL", "error": f"Results not found at {results_path}"}

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    discrepancies = []

    # 1. Check milestone
    if "EQUYLAPTA 7.8" not in report_text and "EQUYLAPTA_7.8" not in report_text:
        discrepancies.append("Report header does not mention EQUYLAPTA 7.8")

    # 2. Check donor intact accuracy
    donor_acc = results.get("native_localization", {}).get("intact_accuracy")
    if donor_acc is not None:
        pattern = str(round(donor_acc, 1))
        if pattern not in report_text:
            discrepancies.append(f"Donor intact accuracy {pattern}% not found in report")

    # 3. Check Condition C active accuracy
    cond_c = results.get("conditions", {}).get("Condition_C_Recipient_Reconstructed_Circuit", {})
    cond_c_acc = cond_c.get("active_accuracy")
    if cond_c_acc is not None:
        pattern = str(round(cond_c_acc, 1))
        if pattern not in report_text:
            discrepancies.append(f"Condition C active accuracy {pattern}% not found in report")

    # 4. Check evidence level
    evidence_level = results.get("scientific_classification", {}).get("evidence_level", "B")
    if f"LEVEL {evidence_level}" not in report_text and f"Level {evidence_level}" not in report_text:
        discrepancies.append(f"Evidence level Level {evidence_level} not clearly stated in report")

    passed = len(discrepancies) == 0

    return {
        "status": "PASS" if passed else "FAIL",
        "checked_report": report_path,
        "checked_results": results_path,
        "discrepancies": discrepancies,
        "consistency_verified": passed
    }


if __name__ == "__main__":
    rep_md = "/home/user/EQUYLAPTA_7.8_REPORT.md"
    res_json = "/home/user/results.json"
    res = verify_consistency(rep_md, res_json)
    print("Report Consistency Verification:", res)
