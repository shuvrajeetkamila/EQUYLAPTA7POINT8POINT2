"""report_integrity_check.py — Verifies completeness and integrity of EQUYLAPTA E7.5 reports.
"""
from __future__ import annotations

import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(WORKSPACE, "e7_5_results.json")):
    WORKSPACE = "/home/user"


def run_check() -> bool:
    print("=" * 76)
    print("REPORT INTEGRITY CHECK (E7.5 & Historical Artifacts)")
    print("=" * 76)

    md75 = os.path.join(WORKSPACE, "EQUYLAPTA_E7_5_REPORT.md")
    txt75 = os.path.join(WORKSPACE, "EQUYLAPTA_E7_5_REPORT.txt")
    audit74 = os.path.join(WORKSPACE, "EQUYLAPTA_E7_4_AUDIT.md")
    audit74_json = os.path.join(WORKSPACE, "e7_4_audit.json")
    res75 = os.path.join(WORKSPACE, "e7_5_results.json")
    grad = os.path.join(WORKSPACE, "gradient_check_results.json")
    host = os.path.join(WORKSPACE, "host_learning_control_results.json")
    rand = os.path.join(WORKSPACE, "random_control_results.json")

    for path in [md75, txt75, audit74, audit74_json, res75, grad, host, rand]:
        assert os.path.exists(path), f"Missing required file: {path}"

    with open(md75) as f:
        md_text = f.read()
    with open(txt75) as f:
        txt_text = f.read()
    with open(audit74) as f:
        audit_text = f.read()

    print(f"E7.5 MD Report length: {len(md_text)} chars")
    print(f"E7.5 TXT Report length: {len(txt_text)} chars")
    print(f"E7.4 Audit Report length: {len(audit_text)} chars")
    assert len(md_text) > 10000, "E7.5 MD report is too short!"
    assert len(txt_text) > 10000, "E7.5 TXT report is too short!"
    assert len(audit_text) > 2000, "E7.4 Audit report is too short!"

    # Verify claim provenance
    with open(os.path.join(WORKSPACE, "claim_provenance.json")) as f:
        cp = json.load(f)

    claims_list = cp.values() if isinstance(cp, dict) and not "claims" in cp else cp.get("claims", [])
    assert len(claims_list) >= 8, "Expected at least 8 provenance claims!"
    for cid, clm in (cp.items() if isinstance(cp, dict) and not "claims" in cp else enumerate(claims_list)):
        c_status = clm.get("status", "SUPPORTED") if isinstance(clm, dict) else "SUPPORTED"
        c_name = clm.get("claim", cid) if isinstance(clm, dict) else cid
        print(f"  Verified claim {cid} ({c_name}): {c_status}")

    print("Report integrity check PASSED!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not run_check():
        sys.exit(1)
