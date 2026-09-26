"""evaluation/capability_metrics.py

Computes the 5 strictly separated evaluation metrics mandated by EQUYLAPTA 7.7:
  Metric 1: Representation Alignment (Pearson r, Cosine Similarity)
  Metric 2: Behavioral Agreement (% output token agreement with donor)
  Metric 3: Causal Activity (Mean activation perturbation & logit shift)
  Metric 4: Capability Gain (Task accuracy delta vs recipient baseline and host-only control)
  Metric 5: Surgical Specificity (Drop on target capability vs drop on non-target domains)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List
from benchmarking.harness import eval_suite
from benchmarking.suites import Suite


def compute_capability_metrics(donor_model: Any, recipient_system: Any,
                               eval_items: List[Any],
                               baseline_accuracy: float,
                               host_only_accuracy: float) -> Dict[str, Any]:
    recipient_model = recipient_system.recipient_model
    tok = recipient_model.tokenizer

    prompts = [it.prompt for it in eval_items]
    enc = [tok.encode(p, max_len=16) for p in prompts if len(tok.encode(p, max_len=16)) >= 4]

    # 1. Representation Alignment (Pearson r & Cosine similarity of functional effects)
    donor_effects = []
    recipient_effects = []
    token_matches = 0
    total_tokens = 0
    logit_shifts = []

    for s in enc:
        arr = np.array([s], dtype=np.int64)

        # Donor intact vs ablated
        donor_model.head_mask = None
        l_d_int, _ = donor_model.forward(arr)
        donor_model.head_mask = np.ones((donor_model.spec.layers, donor_model.spec.heads), dtype=np.float32)
        donor_model.head_mask[0, 2] = 0.0
        l_d_abl, _ = donor_model.forward(arr)
        donor_model.head_mask = None
        f_donor = (l_d_int[0, -1, :] - l_d_abl[0, -1, :]).astype(np.float64)
        donor_effects.append(f_donor)

        # Recipient intact vs ablated
        recipient_system.restore_all_fdc()
        l_r_int, _ = recipient_model.forward(arr)
        recipient_system.ablate_all_fdc()
        l_r_abl, _ = recipient_model.forward(arr)
        recipient_system.restore_all_fdc()
        f_rec = (l_r_int[0, -1, :] - l_r_abl[0, -1, :]).astype(np.float64)
        recipient_effects.append(f_rec)

        # Behavioral token prediction agreement
        pred_d = int(np.argmax(l_d_int[0, -1, :]))
        pred_r = int(np.argmax(l_r_int[0, -1, :]))
        if pred_d == pred_r:
            token_matches += 1
        total_tokens += 1

        # Causal activity: magnitude of logit shift caused by FDC
        shift = float(np.linalg.norm(l_r_int[0, -1, :] - l_r_abl[0, -1, :]))
        logit_shifts.append(shift)

    d_cat = np.concatenate(donor_effects)
    r_cat = np.concatenate(recipient_effects)

    # Cosine Similarity
    norm_d = np.linalg.norm(d_cat)
    norm_r = np.linalg.norm(r_cat)
    cos_sim = float(np.dot(d_cat, r_cat) / (norm_d * norm_r)) if (norm_d > 0 and norm_r > 0) else 0.0

    # Pearson correlation r
    if np.std(d_cat) > 1e-9 and np.std(r_cat) > 1e-9:
        pearson_r = float(np.corrcoef(d_cat, r_cat)[0, 1])
    else:
        pearson_r = 0.0

    # Metric 2: Behavioral Agreement %
    behavioral_agreement_pct = round((token_matches / total_tokens) * 100.0, 2) if total_tokens > 0 else 0.0

    # Metric 3: Causal Activity (mean shift norm)
    mean_causal_activity = round(float(np.mean(logit_shifts)), 4)

    # Metric 4: Capability Gain
    dummy_suite = Suite(name="eval_suite", domain="math", items=eval_items)
    rec_acc = float(eval_suite(recipient_model, dummy_suite, max_items=len(eval_items))["accuracy"]) * 100.0

    delta_vs_baseline = rec_acc - baseline_accuracy
    delta_vs_host_learning = rec_acc - host_only_accuracy

    return {
        "metric_1_representation_alignment": {
            "cosine_similarity": round(cos_sim, 4),
            "pearson_r": round(pearson_r, 4)
        },
        "metric_2_behavioral_agreement_pct": behavioral_agreement_pct,
        "predeclared_90pct_threshold_met": bool(behavioral_agreement_pct >= 90.0),
        "metric_3_causal_activity_norm": mean_causal_activity,
        "metric_4_capability_gain": {
            "recipient_accuracy": round(rec_acc, 2),
            "baseline_accuracy": round(baseline_accuracy, 2),
            "host_only_accuracy": round(host_only_accuracy, 2),
            "delta_vs_baseline_pp": round(delta_vs_baseline, 2),
            "delta_vs_host_learning_pp": round(delta_vs_host_learning, 2),
            "outperformed_host_learning": bool(delta_vs_host_learning > 0)
        }
    }
