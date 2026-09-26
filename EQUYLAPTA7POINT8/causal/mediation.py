"""causal/mediation.py

Recipient-side conditional mediation analysis.
"""
from __future__ import annotations

from typing import Dict, Any


def compute_recipient_mediation(ablation_results: Dict[str, Any], pathway_results: Dict[str, Any]) -> Dict[str, Any]:
    acc_active = ablation_results["active_accuracy"]
    acc_ablated = ablation_results["ablated_accuracy"]
    total_effect = acc_active - acc_ablated

    mlp0_drop = pathway_results["mediator_drop_pp"]

    # Direct effect bypassing Recipient_L0_mlp
    direct_effect = max(0.0, total_effect - mlp0_drop)
    mediated_effect = min(total_effect, mlp0_drop)
    prop_mediated = (mediated_effect / total_effect) if total_effect > 0 else 0.0

    return {
        "recipient_total_effect_pp": round(total_effect, 2),
        "recipient_direct_residual_effect_pp": round(direct_effect, 2),
        "recipient_mediated_mlp0_effect_pp": round(mediated_effect, 2),
        "recipient_proportion_mediated_pct": round(prop_mediated * 100.0, 2),
        "mechanism": "PARALLEL_RESIDUAL_AND_MLP0_MEDIATION"
    }
