"""circuits/circuit_validation.py

Invariant validation for circuit topologies and parameter footprints.
"""
from __future__ import annotations

from typing import Dict, Any, List


def validate_circuit_invariants(circuits: Dict[str, Any]) -> Dict[str, Any]:
    min_c = circuits["minimal_candidate"]
    exact_c = circuits["exact_fdc"]
    over_c = circuits["oversized_circuit"]

    checks = {
        "minimal_less_than_exact_nodes": min_c["node_count"] < exact_c["node_count"],
        "exact_less_than_oversized_nodes": exact_c["node_count"] < over_c["node_count"],
        "minimal_params_correct": min_c["parameter_count"] == 33792,
        "exact_params_correct": exact_c["parameter_count"] == 35840,
        "oversized_params_correct": over_c["parameter_count"] == 135168,
        "oversized_distinct_from_fdc": over_c["node_count"] == 8 and over_c["parameter_count"] > exact_c["parameter_count"]
    }

    all_passed = all(checks.values())
    return {
        "all_invariants_satisfied": all_passed,
        "details": checks
    }
