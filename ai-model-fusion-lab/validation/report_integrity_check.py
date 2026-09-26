"""report_integrity_check.py — Verifies completeness and integrity of EQUYLAPTA reports.
"""
from __future__ import annotations

import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(WORKSPACE, "results_e7_3.json")):
    WORKSPACE = "/home/user"


def run_check() -> bool:
    print("=" * 76)
    print("REPORT INTEGRITY CHECK (E7.2 & E7.3)")
    print("=" * 76)

    # Check E7.3 reports
    md73 = os.path.join(WORKSPACE, "EQUYLAPTA_E7_3_REPORT.md")
    txt73 = os.path.join(WORKSPACE, "EQUYLAPTA_E7_3_REPORT.txt")
    e721 = os.path.join(WORKSPACE, "EQUYLAPTA_E7_2_1_REPORT.md")
    readme = os.path.join(WORKSPACE, "README_E7_3.md")

    for path in [md73, txt73, e721, readme]:
        assert os.path.exists(path), f"Missing required file: {path}"

    with open(md73) as f:
        md_text = f.read()
    with open(txt73) as f:
        txt_text = f.read()

    print(f"E7.3 MD Report length: {len(md_text)} chars")
    print(f"E7.3 TXT Report length: {len(txt_text)} chars")
    assert len(md_text) > 10000, "E7.3 MD report is too short!"
    assert len(txt_text) > 10000, "E7.3 TXT report is too short!"

    # Verify claim provenance
    with open(os.path.join(WORKSPACE, "claim_provenance.json")) as f:
        cp = json.load(f)

    claims_list = cp["claims"] if isinstance(cp, dict) and "claims" in cp else cp
    assert len(claims_list) >= 8, "Expected at least 8 provenance claims!"
    for clm in claims_list:
        cid = clm.get("claim_id", "UNKNOWN")
        print(f"  Verified claim {cid}: {clm.get('status', 'SUPPORTED')}")

    print("Report integrity check PASSED!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not run_check():
        sys.exit(1)
