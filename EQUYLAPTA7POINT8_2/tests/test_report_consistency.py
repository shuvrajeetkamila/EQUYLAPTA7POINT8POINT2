"""tests/test_report_consistency.py

Unit test asserting zero discrepancies between report text and results.json / adapter_update_audit.json.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from validate_report import validate_report_consistency


def test_report_consistency():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rep_p = os.path.join(base_dir, "EQUYLAPTA7POINT8_2_REPORT.md")
    res_p = os.path.join(base_dir, "results.json")
    aud_p = os.path.join(base_dir, "adapter_update_audit.json")

    res = validate_report_consistency(rep_p, res_p, aud_p)
    assert res["status"] == "PASS", f"Report consistency validation failed: {res['errors']}"


if __name__ == "__main__":
    test_report_consistency()
    print("test_report_consistency passed successfully!")
