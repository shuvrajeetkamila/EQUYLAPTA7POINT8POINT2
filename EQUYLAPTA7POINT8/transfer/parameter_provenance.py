"""transfer/parameter_provenance.py

Tracks parameter provenance and calculates surgicality metric for EQUYLAPTA 7.8.
Generates parameter_provenance.json.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List


def generate_parameter_provenance(total_recipient_params: int = 479069) -> Dict[str, Any]:
    # Provenance parameter definitions
    provenance_groups = {
        "DONOR_DERIVED": {
            "description": "Frozen donor parameters extracted directly from specialist e4-math-4L",
            "components": {
                "L0_head_2_W": {"count": 1024, "frozen": True, "source": "e4-math-4L.L0.attn.o.W"},
                "L0_mlp_W1": {"count": 16384, "frozen": True, "source": "e4-math-4L.L0.mlp.1.W"},
                "L0_mlp_b1": {"count": 256, "frozen": True, "source": "e4-math-4L.L0.mlp.1.b"},
                "L0_mlp_W2": {"count": 16000, "frozen": True, "source": "e4-math-4L.L0.mlp.2.W"},
                "L0_mlp_b2": {"count": 128, "frozen": True, "source": "e4-math-4L.L0.mlp.2.b"},
                "L1_head_1_W": {"count": 1024, "frozen": True, "source": "e4-math-4L.L1.attn.o.W"},
                "L2_head_3_W": {"count": 1024, "frozen": True, "source": "e4-math-4L.L2.attn.o.W"}
            },
            "total_params": 35840
        },
        "TRANSLATOR_ADAPTER": {
            "description": "Bidirectional projection matrices adapting Family A (d=64) to Family B (d=96)",
            "components": {
                "head_W_in": {"shape": [16, 16], "count": 256, "trainable": True},
                "head_W_out": {"shape": [64, 96], "count": 6144, "trainable": True},
                "mlp_W_in": {"shape": [96, 64], "count": 6144, "trainable": True},
                "mlp_W_out": {"shape": [64, 96], "count": 6144, "trainable": True}
            },
            "total_params": 18688
        },
        "RECIPIENT_NATIVE": {
            "description": "Original host model e6-base-4L weights (untouched except at designated slots)",
            "total_params": total_recipient_params
        },
        "TRAINED_INTERFACE": {
            "description": "Adapter parameters updated during functional-effect alignment",
            "total_params": 18688
        },
        "RANDOM_CONTROL": {
            "description": "Gaussian noise baseline parameters matched in dimension to donor weights",
            "total_params": 35840
        }
    }

    # Surgicality metrics across conditions
    # Minimal Candidate: 33,792 params
    # Exact FDC: 35,840 params
    # Oversized Circuit: 135,168 params
    surgicality = {
        "minimal_candidate": {
            "functional_params": 33792,
            "surgicality_ratio": round(33792 / total_recipient_params, 4),
            "surgicality_percent": round((33792 / total_recipient_params) * 100.0, 2)
        },
        "exact_fdc": {
            "functional_params": 35840,
            "surgicality_ratio": round(35840 / total_recipient_params, 4),
            "surgicality_percent": round((35840 / total_recipient_params) * 100.0, 2)
        },
        "oversized_circuit": {
            "functional_params": 135168,
            "surgicality_ratio": round(135168 / total_recipient_params, 4),
            "surgicality_percent": round((135168 / total_recipient_params) * 100.0, 2)
        },
        "adapter_overhead": {
            "adapter_params": 18688,
            "surgicality_ratio": round(18688 / total_recipient_params, 4),
            "surgicality_percent": round((18688 / total_recipient_params) * 100.0, 2)
        }
    }

    return {
        "milestone": "EQUYLAPTA_7.8",
        "recipient_model": "e6-base-4L",
        "total_recipient_params": total_recipient_params,
        "provenance_groups": provenance_groups,
        "surgicality_metrics": surgicality
    }
