"""tests/test_report_consistency.py

Unit test for report consistency validation against results.json.
"""
from __future__ import annotations

import unittest
from validate_report import validate_report_consistency


class TestReportConsistency(unittest.TestCase):
    def test_report_validation(self):
        # Only runs if report and results exist
        import os
        if os.path.exists("/home/user/EQUYLAPTA7POINT8_1_REPORT.md") and os.path.exists("/home/user/results.json"):
            res = validate_report_consistency(
                "/home/user/EQUYLAPTA7POINT8_1_REPORT.md",
                "/home/user/results.json"
            )
            self.assertEqual(res["status"], "PASS", f"Report validation errors: {res.get('errors')}")


if __name__ == "__main__":
    unittest.main()
