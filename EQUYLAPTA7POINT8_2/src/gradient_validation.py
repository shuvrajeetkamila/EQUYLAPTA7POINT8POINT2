"""src/gradient_validation.py

Rigorous finite-difference gradient validation across >=10 coordinates per trainable adapter branch.
Evaluated on total loss: L_total = L_functional + lambda_reg * L_reg.
Pass criteria: max_relative_error <= 0.05, mean_relative_error <= 0.02.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any, List
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.synthetic import load_model, gen_suite
from src.transfer import ArchitectureTranslator
from src.corrected_functional_objective import (
    compute_donor_causal_signature,
    compute_functional_loss,
    compute_multi_branch_gradients,
    compute_head_adapter_gradients,
    compute_mlp_adapter_gradients
)


def run_gradient_validation(output_path: str = "gradient_validation_results.json",
                           n_coords: int = 12,
                           eps: float = 1e-2,
                           lambda_reg: float = 0.0001) -> Dict[str, Any]:
    donor = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e4-math-4L"))
    recip = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e6-base-4L"))

    # Condition E has both head and mlp bridges
    translator = ArchitectureTranslator(
        donor, recip,
        active_elements=["L0_head_2", "L0_mlp"],
        condition_name="Condition_E_Recipient_Reconstructed_FDC",
        seed=42
    )

    items = gen_suite("math", n=1, seed=888)
    item = items[0]
    p_ids = recip.tokenizer.encode(item.prompt)
    ids = np.array([p_ids], dtype=np.int64)

    delta_target = compute_donor_causal_signature(donor, ids, source_head=(0, 2))

    head_bridge = translator.head_bridges["L0_head_2"]
    mlp_bridge = translator.mlp_bridges["L0_mlp"]

    def compute_total_loss() -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        translator.apply_to_recipient()
        l_func, diff, l_int, l_abl = compute_functional_loss(recip, ids, delta_target)
        l_reg = 0.5 * lambda_reg * (
            np.sum((head_bridge.W_in - head_bridge.W_in_0) ** 2) +
            np.sum((head_bridge.W_out - head_bridge.W_out_0) ** 2) +
            np.sum((mlp_bridge.W_in - mlp_bridge.W_in_0) ** 2) +
            np.sum((mlp_bridge.W_out - mlp_bridge.W_out_0) ** 2)
        )
        return float(l_func + l_reg), diff, l_int, l_abl

    # Analytical gradients at base parameters
    base_loss, diff, l_int, l_abl = compute_total_loss()
    g_head, g_mlp1, g_mlp2 = compute_multi_branch_gradients(recip, ids, diff, l_int, l_abl, head_slot=0, mlp_d_sub=256, alpha_mlp=0.5)

    dW_in_head, dW_out_head = compute_head_adapter_gradients(g_head, head_bridge, lambda_reg=lambda_reg)
    dW_in_mlp, dW_out_mlp = compute_mlp_adapter_gradients(g_mlp1, g_mlp2, mlp_bridge, lambda_reg=lambda_reg)

    targets = {
        "head_bridges.L0_head_2.W_in": (head_bridge.W_in, dW_in_head),
        "head_bridges.L0_head_2.W_out": (head_bridge.W_out, dW_out_head),
        "mlp_bridges.L0_mlp.W_in": (mlp_bridge.W_in, dW_in_mlp),
        "mlp_bridges.L0_mlp.W_out": (mlp_bridge.W_out, dW_out_mlp)
    }

    results = {}
    rng = np.random.default_rng(12345)

    for name, (param, ana_grad) in targets.items():
        shape = param.shape
        coords_checked = []
        rel_errors = []

        # Find significant coordinates (|ana_grad| >= 0.005) to avoid float32 quantization floor
        valid_indices = np.argwhere(np.abs(ana_grad) >= 0.005)
        if len(valid_indices) < n_coords:
            valid_indices = np.argwhere(np.abs(ana_grad) >= 0.001)

        rng.shuffle(valid_indices)
        chosen_indices = [tuple(int(x) for x in idx) for idx in valid_indices[:n_coords]]

        for idx in chosen_indices:
            orig_val = float(param[idx])

            # Perturb +eps
            param[idx] = orig_val + eps
            lp, _, _, _ = compute_total_loss()

            # Perturb -eps
            param[idx] = orig_val - eps
            lm, _, _, _ = compute_total_loss()

            # Restore original
            param[idx] = orig_val

            num_g = (lp - lm) / (2.0 * eps)
            ana_g = float(ana_grad[idx])
            rel_err = abs(num_g - ana_g) / (abs(num_g) + abs(ana_g) + 1e-12)

            rel_errors.append(rel_err)
            coords_checked.append({
                "coordinate": list(idx),
                "numerical_gradient": float(num_g),
                "analytical_gradient": float(ana_g),
                "relative_error": float(rel_err)
            })

        max_err = float(np.max(rel_errors))
        mean_err = float(np.mean(rel_errors))
        passed = (max_err <= 0.05) and (mean_err <= 0.02)

        results[name] = {
            "parameter_shape": list(shape),
            "num_coordinates_tested": len(coords_checked),
            "max_relative_error": max_err,
            "mean_relative_error": mean_err,
            "pass_criteria_met": passed,
            "coordinates": coords_checked
        }

    overall_pass = all(r["pass_criteria_met"] for r in results.values())
    summary = {
        "milestone": "EQUYLAPTA_7.8.2",
        "verification_type": "EXACT_TWO_BRANCH_FINITE_DIFFERENCES",
        "total_adapter_branches_tested": len(results),
        "overall_gradient_verification_passed": overall_pass,
        "branches": results
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "gradient_validation_results.json")
    res = run_gradient_validation(out_file)
    print("Gradient Validation Result:")
    for b_name, b_data in res["branches"].items():
        print(f"  {b_name}: max_err={b_data['max_relative_error']:.6f}, mean_err={b_data['mean_relative_error']:.6f}, passed={b_data['pass_criteria_met']}")
    print(f"Overall Passed: {res['overall_gradient_verification_passed']}")
