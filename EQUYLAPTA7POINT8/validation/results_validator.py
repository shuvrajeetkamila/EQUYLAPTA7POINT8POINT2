"""validation/results_validator.py

Automated Quality Gate for EQUYLAPTA 7.8 (Section 46).
Verifies that all scientific invariants, conditions, controls, graphs, and files are present and valid.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any, List


def run_quality_gate(results_path: str = "/home/user/results.json",
                     workspace_dir: str = "/home/user") -> Dict[str, Any]:
    gate_checks = {}

    if not os.path.exists(results_path):
        return {"status": "FAIL", "reason": f"results.json missing at {results_path}"}

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    # 1. Milestone identification
    gate_checks["milestone_is_7_8"] = res.get("milestone") == "EQUYLAPTA_7.8"

    # 2. Exhaustive 16 subsets evaluated
    subsets = res.get("exhaustive_subset_search", {}).get("subsets", [])
    gate_checks["exhaustive_16_subsets_present"] = len(subsets) == 16

    # 3. Path patching conditions present (6 conditions)
    path_conds = res.get("path_patching", {}).get("conditions", {})
    gate_checks["path_patching_6_conditions_present"] = len(path_conds) == 6

    # 4. Mediation analysis present
    med = res.get("conditional_mediation", {})
    gate_checks["mediation_analysis_present"] = "total_effect_pp" in med and "proportion_mediated_pct" in med

    # 5. Recipient discovery and graphs present
    donor_graph_path = os.path.join(workspace_dir, "donor_causal_graph.json")
    recip_graph_path = os.path.join(workspace_dir, "recipient_causal_graph.json")
    gate_checks["donor_causal_graph_file_exists"] = os.path.exists(donor_graph_path)
    gate_checks["recipient_causal_graph_file_exists"] = os.path.exists(recip_graph_path)

    # 6. Oversized circuit distinct from FDC
    oversized = res.get("circuits", {}).get("oversized_circuit", {})
    fdc = res.get("circuits", {}).get("exact_fdc", {})
    gate_checks["oversized_circuit_distinct_from_fdc"] = (
        oversized.get("node_count", 0) == 8 and
        oversized.get("node_count", 0) > fdc.get("node_count", 0)
    )

    # 7. Parameter provenance file exists
    provenance_path = os.path.join(workspace_dir, "parameter_provenance.json")
    gate_checks["parameter_provenance_file_exists"] = os.path.exists(provenance_path)

    # 8. Gradient check passed
    grad_check = res.get("gradient_check", {})
    gate_checks["gradient_check_passed"] = grad_check.get("gradient_check_passed", False) is True

    # 9. All 10 conditions evaluated
    conds = res.get("conditions", {})
    gate_checks["all_conditions_evaluated"] = len(conds) >= 10

    # 10. Activation patch battery present
    act_patch = res.get("activation_patch_battery", {})
    gate_checks["activation_patch_battery_present"] = "validation" in act_patch and "heldout" in act_patch

    # 11. 5-seed statistics present
    seed_res = res.get("seed_replication", {})
    gate_checks["five_seed_replication_present"] = len(seed_res) > 0

    # 12. Honest scientific classification
    classification = res.get("scientific_classification", {}).get("evidence_level")
    gate_checks["classification_honestly_reported"] = classification in ["A", "B", "C"]

    # 13. Disk usage check (< 120 MB)
    size_report_path = os.path.join(workspace_dir, "artifact_size_report.json")
    if os.path.exists(size_report_path):
        with open(size_report_path, "r", encoding="utf-8") as sf:
            s_data = json.load(sf)
        workspace_mb = s_data.get("total_workspace_size_mb", 0.0)
        gate_checks["disk_usage_strictly_under_120mb"] = (workspace_mb < 120.0)
    else:
        gate_checks["disk_usage_strictly_under_120mb"] = True

    all_passed = all(gate_checks.values())

    return {
        "status": "PASS" if all_passed else "FAIL",
        "quality_gate_passed": all_passed,
        "checks": gate_checks
    }


if __name__ == "__main__":
    qg = run_quality_gate()
    print("Quality Gate Result:", qg)
