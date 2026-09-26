"""transfer/activation_patch_transfer.py

Implements direct donor-to-recipient surgical activation patching (Section 24).
Tests whether donor capability activations can directly drive recipient predictions.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import load_model, MicroTransformer
from benchmarking.suites import micro_suite
from benchmarking.harness import eval_suite


def evaluate_donor_to_recipient_patch(donor_model: MicroTransformer,
                                      recipient_model: MicroTransformer,
                                      suite: Any,
                                      patch_mode: str = "donor",
                                      seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    d_src = donor_model.spec.hidden
    d_tgt = recipient_model.spec.hidden

    correct = 0
    for it in suite.items:
        p_ids = recipient_model.tokenizer.encode(it.prompt)
        scores = []
        for o in it.options:
            o_ids = recipient_model.tokenizer.encode(" " + o, max_len=8)
            ids = np.concatenate([p_ids, o_ids])[None, :]

            if patch_mode == "donor":
                _, c_src = donor_model.forward(ids, collect=True)
                act_src = c_src["hiddens"][0]  # [1, T, d_src]
                act_tgt = np.zeros((1, ids.shape[1], d_tgt), dtype=np.float32)
                act_tgt[:, :, :min(d_src, d_tgt)] = act_src[:, :, :min(d_src, d_tgt)]

            elif patch_mode == "random":
                act_tgt = rng.standard_normal((1, ids.shape[1], d_tgt)).astype(np.float32) * 0.5

            elif patch_mode == "shuffled":
                _, c_src = donor_model.forward(ids, collect=True)
                act_src = c_src["hiddens"][0]
                act_tgt = np.zeros((1, ids.shape[1], d_tgt), dtype=np.float32)
                perm = rng.permutation(ids.shape[1])
                act_tgt[:, :, :min(d_src, d_tgt)] = act_src[:, perm, :min(d_src, d_tgt)]

            elif patch_mode == "baseline":
                act_tgt = None

            # Patch recipient at Layer 0 residual output
            if act_tgt is not None:
                recipient_model.patch = {("resid", 1): act_tgt}
            else:
                recipient_model.patch = {}

            logits, _ = recipient_model.forward(ids)
            recipient_model.patch = {}

            # Option log-prob
            lp = 0.0
            n = len(o_ids)
            for i in range(n):
                pos = ids.shape[1] - n + i - 1
                row = logits[0, pos] - logits[0, pos].max()
                logz = np.log(np.exp(row).sum())
                lp += float(row[o_ids[i]] - logz)
            scores.append(lp / n)

        if int(np.argmax(scores)) == it.answer:
            correct += 1

    return (correct / len(suite.items)) * 100.0


def run_activation_patch_battery(donor_model_name: str = "e4-math-4L",
                                 recipient_model_name: str = "e6-base-4L",
                                 val_seed: int = 888,
                                 heldout_seed: int = 999) -> Dict[str, Any]:
    donor = load_model(donor_model_name)
    recipient = load_model(recipient_model_name)

    suite_val = micro_suite("math", n=20, seed=val_seed)
    suite_held = micro_suite("math", n=20, seed=heldout_seed)

    # Validation split
    acc_base_val = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "baseline")
    acc_donor_val = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "donor")
    acc_rand_val = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "random")
    acc_shuf_val = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "shuffled")

    # Held-out split
    acc_base_held = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "baseline")
    acc_donor_held = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "donor")
    acc_rand_held = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "random")
    acc_shuf_held = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "shuffled")

    donor_lift_val = acc_donor_val - acc_base_val
    donor_lift_held = acc_donor_held - acc_base_held

    return {
        "validation": {
            "baseline_accuracy": round(acc_base_val, 2),
            "donor_patch_accuracy": round(acc_donor_val, 2),
            "random_patch_accuracy": round(acc_rand_val, 2),
            "shuffled_patch_accuracy": round(acc_shuf_val, 2),
            "donor_lift_over_baseline_pp": round(donor_lift_val, 2),
            "donor_lift_over_random_pp": round(acc_donor_val - acc_rand_val, 2),
            "donor_lift_over_shuffled_pp": round(acc_donor_val - acc_shuf_val, 2)
        },
        "heldout": {
            "baseline_accuracy": round(acc_base_held, 2),
            "donor_patch_accuracy": round(acc_donor_held, 2),
            "random_patch_accuracy": round(acc_rand_held, 2),
            "shuffled_patch_accuracy": round(acc_shuf_held, 2),
            "donor_lift_over_baseline_pp": round(donor_lift_held, 2),
            "donor_lift_over_random_pp": round(acc_donor_held - acc_rand_held, 2),
            "donor_lift_over_shuffled_pp": round(acc_donor_held - acc_shuf_held, 2)
        },
        "conclusion": "DIRECT_ACTIVATION_LIFT_CONFIRMED" if donor_lift_val > 0 else "NO_ACTIVATION_LIFT"
    }


if __name__ == "__main__":
    res = run_activation_patch_battery()
    print("Activation patch battery results:", res["validation"])
