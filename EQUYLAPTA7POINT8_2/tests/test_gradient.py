"""tests/test_gradient.py

Verifies analytical gradients against central finite differences for all declared adapter branches:
- head_bridges.L0_head_2.W_in
- head_bridges.L0_head_2.W_out
- mlp_bridges.L0_mlp.W_in
- mlp_bridges.L0_mlp.W_out
Pass criteria: max_relative_error <= 0.05, mean_relative_error <= 0.02, >= 10 coordinates each.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gradient_validation import run_gradient_validation


def test_multi_branch_gradients():
    res = run_gradient_validation(output_path=None, n_coords=12, eps=1e-2)
    assert res["overall_gradient_verification_passed"] is True, "Overall gradient verification failed!"

    for b_name, b_data in res["branches"].items():
        assert b_data["num_coordinates_tested"] >= 10, f"Too few coordinates tested for {b_name}"
        assert b_data["max_relative_error"] <= 0.05, f"Max relative error exceeded for {b_name}: {b_data['max_relative_error']}"
        assert b_data["mean_relative_error"] <= 0.02, f"Mean relative error exceeded for {b_name}: {b_data['mean_relative_error']}"
        assert b_data["pass_criteria_met"] is True, f"Pass criteria not met for {b_name}"


if __name__ == "__main__":
    test_multi_branch_gradients()
    print("test_gradient passed successfully!")
