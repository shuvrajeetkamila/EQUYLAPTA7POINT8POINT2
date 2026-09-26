"""[WORKING] License registry + compatibility check (spec §17).

Curated entries for common open/open-weight families. Closed models are
explicitly NOT redistributable and their outputs are restricted as
distillation teachers by their terms — recorded, not assumed away.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

LICENSES: Dict[str, dict] = {
    "Apache-2.0": {"type": "permissive", "commercial": True, "redistribute": True,
                   "attribution": "LICENSE + NOTICE", "derivative_conditions": [],
                   "notes": "Used by Qwen, Mistral, OLMo, Granite, Pythia, GPT-NeoX, Falcon(r1)"},
    "MIT": {"type": "permissive", "commercial": True, "redistribute": True,
            "attribution": "LICENSE", "derivative_conditions": [],
            "notes": "Phi-2/Phi-3, DeepSeek (some releases)"},
    "llama-community": {"type": "community", "commercial": True, "redistribute": True,
                        "attribution": "Required: 'Built with Llama'",
                        "derivative_conditions": ["name must include 'Llama'", "acceptance of use policy"],
                        "notes": "Llama 2/3 — conditions apply; 700M MAU clause"},
    "gemma": {"type": "community", "commercial": True, "redistribute": True,
              "attribution": "Gemma Terms",
              "derivative_conditions": ["use restrictions in Gemma Terms"],
              "notes": "Gemma — proactive restrictions list"},
    "openrail": {"type": "community", "commercial": True, "redistribute": True,
                 "attribution": "RAIL license file",
                 "derivative_conditions": ["use-based restrictions"],
                 "notes": "BLOOM(RAIL), some Falcon variants"},
    "bigscience-openrail": {"type": "community", "commercial": True, "redistribute": True,
                            "attribution": "RAIL", "derivative_conditions": [],
                            "notes": "BLOOM base"},
    "falcon": {"type": "permissive-ish", "commercial": True, "redistribute": True,
               "attribution": "Falcon license", "derivative_conditions": ["honest-use clause"],
               "notes": "TII Falcon"},
    "glm": {"type": "community", "commercial": True, "redistribute": True,
            "attribution": "Model license", "derivative_conditions": [],
            "notes": "ChatGLM family"},
    "kimi": {"type": "community", "commercial": True, "redistribute": True,
             "attribution": "Modified MIT", "derivative_conditions": [],
             "notes": "Moonshot Kimi open releases"},
    "internlm": {"type": "community", "commercial": True, "redistribute": True,
                 "attribution": "Apache-2.0 w/ supplementary", "derivative_conditions": [],
                 "notes": "InternLM2"},
    "minicpm": {"type": "community", "commercial": True, "redistribute": True,
                "attribution": "Apache-2.0 + commercial registration for some sizes",
                "derivative_conditions": ["registration for commercial use (some versions)"],
                "notes": "MiniCPM"},
    "nemotron": {"type": "community", "commercial": True, "redistribute": True,
                 "attribution": "NVIDIA Open Model License",
                 "derivative_conditions": ["NVIDIA license terms"], "notes": "Nemotron"},
    "proprietary-closed": {"type": "closed", "commercial": False, "redistribute": False,
                           "attribution": "n/a", "derivative_conditions": ["no weights exist"],
                           "notes": "Claude / ChatGPT: no downloadable weights. Output use as "
                                    "distillation teachers is restricted by ToS — do not build "
                                    "competing models from their outputs. REGISTERED, EXCLUDED."},
}


def check_model(name: str, license_name: str) -> dict:
    info = LICENSES.get(license_name)
    if info is None:
        return {"model": name, "license": license_name, "known": False,
                "distributable": False, "action": "ADD LICENSE INFO BEFORE USE"}
    return {"model": name, "license": license_name, "known": True,
            "type": info["type"], "commercial": info["commercial"],
            "redistributable": info["redistribute"],
            "attribution": info["attribution"], "conditions": info["derivative_conditions"],
            "distributable": info["redistribute"] and info["type"] != "closed",
            "notes": info["notes"]}


def check_configuration(models: List[dict]) -> dict:
    """Can a model fused from these sources legally be distributed?"""
    checks = [check_model(m["name"], m.get("license", "unknown")) for m in models]
    distributable = all(c["distributable"] for c in checks)
    conditions = []
    for c in checks:
        conditions += [f"{c['model']}: {x}" for x in c.get("conditions", [])]
    attrs = [f"{c['model']} ({c.get('attribution','?')})" for c in checks if c["known"]]
    return {
        "models": [c["model"] for c in checks],
        "license_compatibility": "COMPATIBLE" if distributable else "NOT DISTRIBUTABLE",
        "required_attributions": attrs,
        "derivative_conditions": conditions,
        "detail": checks,
    }
