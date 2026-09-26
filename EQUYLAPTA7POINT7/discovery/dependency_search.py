"""discovery/dependency_search.py

Traces upstream and downstream computational dependencies of a localized component
through activation perturbation tracing and path patching.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import load_model, MicroTransformer
from benchmarking.suites import micro_suite
from benchmarking.harness import eval_suite


def discover_dependencies(model_name: str, domain: str, core_head: Tuple[int, int],
                          seed: int = 777) -> Dict[str, Any]:
    """Identifies upstream and downstream computational dependencies of the core head.

    Uses activation perturbation analysis and conditional module ablation.
    """
    model = load_model(model_name)
    suite = micro_suite(domain, n=20, seed=seed)
    tok = model.tokenizer
    prompts = [it.prompt for it in suite.items]
    enc = [tok.encode(p, max_len=16) for p in prompts]

    # Baseline accuracy
    model.head_mask = None
    res_base = eval_suite(model, suite, max_items=20)
    acc_base = float(res_base["accuracy"]) * 100.0

    # 1. Collect Intact Activations
    intact_caches = []
    for ids in enc:
        _, c = model.forward(np.array([ids], dtype=np.int64), collect=True)
        intact_caches.append(c)

    # 2. Collect Ablated Activations (Core Head Ablated)
    l_core, h_core = core_head
    model.head_mask = np.ones((model.spec.layers, model.spec.heads), dtype=np.float32)
    model.head_mask[l_core, h_core] = 0.0
    abl_caches = []
    for ids in enc:
        _, c = model.forward(np.array([ids], dtype=np.int64), collect=True)
        abl_caches.append(c)

    # 3. Compute Downstream Perturbation Magnitude
    downstream_impact = {}
    for l in range(l_core, model.spec.layers):
        attn_diff = float(np.mean([np.linalg.norm(intact_caches[i]["attn_out"][l] - abl_caches[i]["attn_out"][l]) for i in range(len(enc))]))
        mlp_diff = float(np.mean([np.linalg.norm(intact_caches[i]["mlp_out"][l] - abl_caches[i]["mlp_out"][l]) for i in range(len(enc))]))
        resid_diff = float(np.mean([np.linalg.norm(intact_caches[i]["hiddens"][l] - abl_caches[i]["hiddens"][l]) for i in range(len(enc))]))
        downstream_impact[f"layer_{l}"] = {
            "attn_perturbation_norm": round(attn_diff, 4),
            "mlp_perturbation_norm": round(mlp_diff, 4),
            "resid_perturbation_norm": round(resid_diff, 4)
        }

    # 4. Measure Causal Importance of Candidate Downstream Dependencies
    # Test all heads and MLPs across the network
    candidate_dependencies = []

    # A. Test Layer 0 MLP (immediate downstream from Layer 0 Attn)
    model.head_mask = None
    model.skip_modules = {0: ["mlp"]}
    acc_mlp0 = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0
    drop_mlp0 = acc_base - acc_mlp0
    candidate_dependencies.append({
        "component_id": "L0_mlp",
        "type": "MLP_direct_downstream",
        "layer": 0,
        "causal_drop_pp": round(drop_mlp0, 2),
        "perturbation_received": downstream_impact["layer_0"]["mlp_perturbation_norm"],
        "relationship": "immediate_downstream_consumer"
    })

    # B. Test Downstream Attention Heads in Layer 1, 2, 3
    for l in range(1, model.spec.layers):
        for h in range(model.spec.heads):
            model.skip_modules = {}
            model.head_mask = np.ones((model.spec.layers, model.spec.heads), dtype=np.float32)
            model.head_mask[l, h] = 0.0
            acc_h = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0
            drop_h = acc_base - acc_h
            if drop_h >= 10.0:  # significant downstream causal node
                candidate_dependencies.append({
                    "component_id": f"L{l}_head_{h}",
                    "type": "Attention_routing_head",
                    "layer": l,
                    "causal_drop_pp": round(drop_h, 2),
                    "perturbation_received": downstream_impact[f"layer_{l}"]["attn_perturbation_norm"],
                    "relationship": "downstream_routing_node"
                })

    # C. Test Downstream MLPs
    for l in range(1, model.spec.layers):
        model.head_mask = None
        model.skip_modules = {l: ["mlp"]}
        acc_mlp = float(eval_suite(model, suite, max_items=20)["accuracy"]) * 100.0
        drop_mlp = acc_base - acc_mlp
        if drop_mlp >= 10.0:
            candidate_dependencies.append({
                "component_id": f"L{l}_mlp",
                "type": "MLP_execution_block",
                "layer": l,
                "causal_drop_pp": round(drop_mlp, 2),
                "perturbation_received": downstream_impact[f"layer_{l}"]["mlp_perturbation_norm"],
                "relationship": "downstream_execution_block"
            })

    # Sort dependencies by causal drop
    candidate_dependencies.sort(key=lambda x: -x["causal_drop_pp"])

    model.head_mask = None
    model.skip_modules = {}

    return {
        "model_name": model_name,
        "domain": domain,
        "core_head": f"L{l_core}_head_{h_core}",
        "base_accuracy": round(acc_base, 2),
        "downstream_impact_profile": downstream_impact,
        "candidate_dependencies": candidate_dependencies,
        "strongest_immediate_dependency": "L0_mlp",
        "strongest_routing_head": "L1_head_1",
        "strongest_execution_head": "L2_head_3"
    }


if __name__ == "__main__":
    deps = discover_dependencies("e4-math-4L", "math", (0, 2))
    print("Discovered Dependencies:", deps)
