"""Pattern null models and randomization controls (Milestone 4 spec §12).

The question is always: *is this pattern more predictive of capability than
chance?* Every detection statistic is compared against its mandated null and
reported with an empirical p-value (never assumed distributions).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def p_value(stat: float, null_stats: Sequence[float], tail: str = "greater") -> float:
    """Empirical Monte-Carlo p: (1 + #nulls as extreme) / (K + 1)."""
    n = np.asarray(null_stats, float)
    if tail == "greater":
        c = int(np.sum(n >= stat))
    elif tail == "less":
        c = int(np.sum(n <= stat))
    else:
        c = int(np.sum(np.abs(n) >= abs(stat)))
    return float((1 + c) / (len(n) + 1))


def random_position_sets(span: int, k: int, K: int = 200, seed: int = 0) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(K):
        out.append(sorted(rng.choice(span, size=min(k, span), replace=False).tolist()))
    return out


def random_same_size_node_sets(members: List[str], K: int = 200, seed: int = 0) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    pool = [m for m in members]
    while len(out) < K:
        k = rng.integers(max(1, len(pool) - 1), len(pool) + 1)
        s = tuple(sorted(rng.choice(pool, size=k, replace=False).tolist()))
        if s not in seen:
            seen.add(s)
            out.append(list(s))
    return out


def degree_preserving_random_graph(edges: List[Tuple[str, str]], nodes: List[str],
                                   depth_of: Dict[str, int], K: int = 200,
                                   seed: int = 0) -> List[List[Tuple[str, str]]]:
    """§12 (hub/graph nulls): rewire while keeping each node's degree sequence
    AND a valid DAG ordering (edges only from shallower/equal depth to deeper)."""
    rng = np.random.default_rng(seed)
    outs: Dict[str, int] = {}
    ins: Dict[str, int] = {}
    for u, v in edges:
        outs[u] = outs.get(u, 0) + 1
        ins[v] = ins.get(v, 0) + 1
    out_pool: List[str] = sum(([u] * c for u, c in outs.items()), [])
    in_pool: List[str] = sum(([v] * c for v, c in ins.items()), [])
    graphs = []
    for _ in range(K):
        rng.shuffle(out_pool)
        rng.shuffle(in_pool)
        es = []
        for u, v in zip(out_pool, in_pool):
            if u == v or depth_of.get(u, 0) > depth_of.get(v, 0):
                continue
            if (u, v) in es:
                continue
            es.append((u, v))
        graphs.append(es)
    return graphs


def random_ratio_samples(n: int, K: int = 200, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=n)) for _ in range(K)]


def shuffled_labels(y: List[int], K: int = 200, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    yarr = np.asarray(y)
    out = []
    for _ in range(K):
        p = rng.permutation(len(yarr))
        out.append(yarr[p])
    return out


def random_mask_circuit(model_spec, members: List[dict], K: int = 8,
                        seed: int = 0) -> List[List[dict]]:
    """Random-structure control circuits: same member LEVELS/layers shuffled
    onto random valid indices (same param budget, random 'structure')."""
    rng = np.random.default_rng(seed)
    L, H = model_spec.layers, model_spec.heads
    out = []
    for _ in range(K):
        rm = []
        for m in members:
            mm = dict(m)
            if mm["level"] == "HEAD":
                mm["layer"] = int(rng.integers(0, L))
                mm["index"] = int(rng.integers(0, H))
            elif mm["level"] == "MODULE":
                mm["layer"] = int(rng.integers(0, L))
            elif mm["level"] == "CHANNEL_GROUP":
                d = mm.get("dims") or {"lo": 0, "hi": 32}
                w = d.get("hi", 32) - d.get("lo", 0)
                lo = int(rng.integers(0, max(1, 256 - w)))
                mm["dims"] = {"lo": lo, "hi": lo + w}
            rm.append(mm)
        out.append(rm)
    return out
