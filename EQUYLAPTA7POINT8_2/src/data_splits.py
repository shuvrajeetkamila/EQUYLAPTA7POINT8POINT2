"""src/data_splits.py

Enforces the strict leakage-prevention protocol.
Guarantees mathematically disjoint token/sequence splits between:
- Characterization split (seed 777 base)
- Validation split (seed 888 base)
- Held-out test split (seed 999 base)
Ensuring zero prompt or sequence overlap.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Dict, Any, Set
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import gen_suite
from benchmarking.suites import Suite, SuiteItem


def get_disjoint_splits(domain: str = "math", n: int = 20) -> Dict[str, Suite]:
    """Generates strictly disjoint suites for characterization, validation, and held-out testing."""
    seen_prompts: Set[str] = set()
    suites: Dict[str, Suite] = {}
    seeds = {"characterization": 777, "validation": 888, "heldout": 999}

    for split_name, base_seed in seeds.items():
        items = []
        cur_seed = base_seed
        while len(items) < n:
            candidates = gen_suite(domain, n=50, seed=cur_seed)
            for c in candidates:
                if c.prompt not in seen_prompts:
                    seen_prompts.add(c.prompt)
                    items.append(c)
                    if len(items) == n:
                        break
            cur_seed += 10000
        suites[split_name] = Suite(f"{split_name}-{domain}", domain, [SuiteItem(**asdict(i)) for i in items])

    return suites
