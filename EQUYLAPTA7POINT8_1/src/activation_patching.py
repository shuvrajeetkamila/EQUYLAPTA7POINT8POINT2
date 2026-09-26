"""src/activation_patching.py

Direct surgical activation patching from donor to recipient residual stream.
Strictly validates on BOTH validation split (seed=888) and held-out test split (seed=999).
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
                act_src = c_src["hiddens"][0]
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

            elif patch_mode == "zero":
                act_tgt = np.zeros((1, ids.shape[1], d_tgt), dtype=np.float32)

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

    # Validation
    val_base = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "baseline")
    val_donor = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "donor")
    val_rand = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "random")
    val_shuf = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "shuffled")
    val_zero = evaluate_donor_to_recipient_patch(donor, recipient, suite_val, "zero")

    # Held-out
    held_base = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "baseline")
    held_donor = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "donor")
    held_rand = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "random")
    held_shuf = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "shuffled")
    held_zero = evaluate_donor_to_recipient_patch(donor, recipient, suite_held, "zero")

    val_lift = val_donor - val_base
    held_lift = held_donor - held_base

    return {
        "validation_seed_888": {
            "baseline": round(val_base, 2),
            "donor_patch": round(val_donor, 2),
            "random_patch": round(val_rand, 2),
            "shuffled_patch": round(val_shuf, 2),
            "zero_patch": round(val_zero, 2),
            "donor_lift_over_baseline_pp": round(val_lift, 2),
            "donor_lift_over_random_pp": round(val_donor - val_rand, 2),
            "donor_lift_over_shuffled_pp": round(val_donor - val_shuf, 2)
        },
        "heldout_seed_999": {
            "baseline": round(held_base, 2),
            "donor_patch": round(held_donor, 2),
            "random_patch": round(held_rand, 2),
            "shuffled_patch": round(held_shuf, 2),
            "zero_patch": round(held_zero, 2),
            "donor_lift_over_baseline_pp": round(held_lift, 2),
            "donor_lift_over_random_pp": round(held_donor - held_rand, 2),
            "donor_lift_over_shuffled_pp": round(held_donor - held_shuf, 2)
        },
        "heldout_effect_retained": held_lift > 0 and (held_donor >= held_rand)
    }
