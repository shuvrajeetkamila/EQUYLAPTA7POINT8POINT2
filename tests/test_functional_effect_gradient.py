"""tests/test_functional_effect_gradient.py

Mandatory Finite-Difference Gradient Test for EQUYLAPTA E7.5.
Verifies that the analytical two-branch gradient of the functional effect objective:
    L_func = 0.5 * mean( ((Y_intact(x, theta) - Y_ablated(x, theta)) - F_source(x))^2 )
differentiates through both the intact branch (+dlogits) and ablated branch (-dlogits)
and matches central finite differences with relative error < 0.05.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ai-model-fusion-lab"))
from models.synthetic import load_model


def run_gradient_check(model_name: str = "e4-base-4L",
                       epsilon: float = 1e-4,
                       tolerance: float = 0.05) -> dict:
    print("=" * 76)
    print(f"ANALYTICAL VS. FINITE-DIFFERENCE GRADIENT CHECK [MODEL: {model_name}]")
    print("=" * 76)

    model = load_model(model_name)
    spec = model.spec
    head_dim = spec.hidden // spec.heads
    d_comp = 64
    d_tgt = spec.hidden

    np.random.seed(42)
    # Parameter groups
    W_in = np.random.randn(head_dim, head_dim).astype(np.float32) * 0.1
    W_comp = np.random.randn(head_dim, d_comp).astype(np.float32) * 0.1  # FROZEN
    W_out = np.random.randn(d_comp, d_tgt).astype(np.float32) * 0.1

    F_src = np.random.randn(spec.vocab).astype(np.float32)
    arr = np.array([[1, 2, 3, 4]], dtype=np.int64)

    def compute_loss(w_in: np.ndarray, w_out: np.ndarray, w_rec: np.ndarray | None = None) -> float:
        w_eff = (w_in @ W_comp @ w_out).astype(np.float32)
        model.params["L0.attn.o.W"][:head_dim, :] = w_eff
        if w_rec is not None:
            model.params["L0.mlp.1.W"] = w_rec.astype(np.float32)

        # 1. Intact forward pass
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        logits_int, _ = model.forward(arr)
        y_int = logits_int[0, -1, :].astype(np.float64)

        # 2. Ablated forward pass (component disabled in head 0)
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        model.head_mask[0, 0] = 0.0
        logits_abl, _ = model.forward(arr)
        y_abl = logits_abl[0, -1, :].astype(np.float64)

        diff = (y_int - y_abl) - F_src
        return 0.5 * float(np.mean(diff ** 2))

    def compute_analytical_gradients(w_in: np.ndarray, w_out: np.ndarray, w_rec: np.ndarray | None = None):
        w_eff = (w_in @ W_comp @ w_out).astype(np.float32)
        model.params["L0.attn.o.W"][:head_dim, :] = w_eff
        if w_rec is not None:
            model.params["L0.mlp.1.W"] = w_rec.astype(np.float32)

        # 1. Intact pass
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        logits_int, _ = model.forward(arr)
        y_int = logits_int[0, -1, :].astype(np.float64)

        # 2. Ablated pass
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        model.head_mask[0, 0] = 0.0
        logits_abl, _ = model.forward(arr)
        y_abl = logits_abl[0, -1, :].astype(np.float64)

        diff = (y_int - y_abl) - F_src
        dlogits = (diff / spec.vocab).astype(np.float32)

        # Branch 1: Intact backward (+dlogits)
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        dlogits_int = np.zeros_like(logits_int)
        dlogits_int[0, -1, :] = dlogits
        grads_int = model.backward(arr, dlogits_int)

        # Branch 2: Ablated backward (-dlogits)
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        model.head_mask[0, 0] = 0.0
        dlogits_abl = np.zeros_like(logits_abl)
        dlogits_abl[0, -1, :] = -dlogits
        grads_abl = model.backward(arr, dlogits_abl)

        # Combined effective gradient for head projection
        G_eff = grads_int["L0.attn.o.W"][:head_dim, :] + grads_abl["L0.attn.o.W"][:head_dim, :]

        # Bridge gradients via chain rule
        dW_in = G_eff @ (W_comp @ w_out).T
        dW_out = (w_in @ W_comp).T @ G_eff

        # Receiver gradient combining both branches
        dRec = grads_int["L0.mlp.1.W"] + grads_abl["L0.mlp.1.W"]

        return dW_in, dW_out, dRec

    orig_rec = model.params["L0.mlp.1.W"].copy()
    dW_in, dW_out, dRec = compute_analytical_gradients(W_in, W_out, orig_rec)

    records = []
    rel_errors = []
    abs_errors = []

    # 1. Test W_in parameters
    print("\n[Group 1: Input Translator W_in]")
    test_coords_in = [(0, 0), (1, 2), (2, 3), (3, 1), (0, 3)]
    for (i, j) in test_coords_in:
        w_p = W_in.copy(); w_p[i, j] += epsilon
        w_m = W_in.copy(); w_m[i, j] -= epsilon
        num = (compute_loss(w_p, W_out, orig_rec) - compute_loss(w_m, W_out, orig_rec)) / (2 * epsilon)
        ana = float(dW_in[i, j])
        abs_err = abs(num - ana)
        denom = abs(num) + abs(ana)
        rel_err = abs_err / denom if denom > 1e-12 else 0.0
        rel_errors.append(rel_err)
        abs_errors.append(abs_err)
        records.append({
            "parameter_group": "INPUT_TRANSLATOR",
            "name": f"W_in[{i},{j}]",
            "analytical_gradient": ana,
            "numerical_gradient": num,
            "absolute_error": abs_err,
            "relative_error": rel_err
        })
        print(f"  W_in[{i},{j}]: Ana={ana:+.6e}, Num={num:+.6e}, RelErr={rel_err:.2e}")

    # 2. Test W_out parameters
    print("\n[Group 2: Output Translator W_out]")
    test_coords_out = [(0, 0), (5, 8), (12, 15), (20, 2), (30, 10)]
    for (i, j) in test_coords_out:
        w_p = W_out.copy(); w_p[i, j] += epsilon
        w_m = W_out.copy(); w_m[i, j] -= epsilon
        num = (compute_loss(W_in, w_p, orig_rec) - compute_loss(W_in, w_m, orig_rec)) / (2 * epsilon)
        ana = float(dW_out[i, j])
        abs_err = abs(num - ana)
        denom = abs(num) + abs(ana)
        rel_err = abs_err / denom if denom > 1e-12 else 0.0
        rel_errors.append(rel_err)
        abs_errors.append(abs_err)
        records.append({
            "parameter_group": "OUTPUT_TRANSLATOR",
            "name": f"W_out[{i},{j}]",
            "analytical_gradient": ana,
            "numerical_gradient": num,
            "absolute_error": abs_err,
            "relative_error": rel_err
        })
        print(f"  W_out[{i},{j}]: Ana={ana:+.6e}, Num={num:+.6e}, RelErr={rel_err:.2e}")

    # 3. Test Receiver parameters (Layer 0 MLP)
    print("\n[Group 3: Recipient Receiver L0.mlp.1.W (Two-Branch Verification)]")
    test_coords_rec = [(0, 0), (2, 5), (8, 10), (14, 3), (15, 20)]
    for (i, j) in test_coords_rec:
        r_p = orig_rec.copy(); r_p[i, j] += epsilon
        r_m = orig_rec.copy(); r_m[i, j] -= epsilon
        num = (compute_loss(W_in, W_out, r_p) - compute_loss(W_in, W_out, r_m)) / (2 * epsilon)
        ana = float(dRec[i, j])
        abs_err = abs(num - ana)
        denom = abs(num) + abs(ana)
        rel_err = abs_err / denom if denom > 1e-12 else 0.0
        rel_errors.append(rel_err)
        abs_errors.append(abs_err)
        records.append({
            "parameter_group": "RECIPIENT_RECEIVER",
            "name": f"L0.mlp.1.W[{i},{j}]",
            "analytical_gradient": ana,
            "numerical_gradient": num,
            "absolute_error": abs_err,
            "relative_error": rel_err
        })
        print(f"  L0.mlp.1.W[{i},{j}]: Ana={ana:+.6e}, Num={num:+.6e}, RelErr={rel_err:.2e}")

    max_rel_error = float(max(rel_errors))
    mean_rel_error = float(np.mean(rel_errors))
    max_abs_error = float(max(abs_errors))
    passed = bool(max_rel_error < tolerance)

    print("\n" + "=" * 76)
    print(f"GRADIENT CHECK SUMMARY:")
    print(f"  Epsilon: {epsilon:.1e}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Max Relative Error:  {max_rel_error:.6e}")
    print(f"  Mean Relative Error: {mean_rel_error:.6e}")
    print(f"  Max Absolute Error:  {max_abs_error:.6e}")
    print(f"  Test Status:         {'PASS' if passed else 'FAIL'}")
    print("=" * 76)

    results = {
        "model": model_name,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "max_relative_error": max_rel_error,
        "mean_relative_error": mean_rel_error,
        "max_absolute_error": max_abs_error,
        "gradient_check_passed": passed,
        "status": "PASS" if passed else "FAIL",
        "detailed_evaluations": records
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "gradient_check_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved canonical gradient check results to: {out_path}")
    return results


if __name__ == "__main__":
    res = run_gradient_check()
    if not res["gradient_check_passed"]:
        print("CRITICAL ERROR: Gradient check failed!", file=sys.stderr)
        sys.exit(1)
