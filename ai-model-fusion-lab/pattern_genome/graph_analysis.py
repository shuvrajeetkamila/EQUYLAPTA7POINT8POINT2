"""Universal discovery primitives (Milestone 4 spec §11).

A. Activation Graph Builder   -> build_activation_graph
B. Motif Detector             -> detect_* functions on the graph
C. Sequence Detector          -> sequence/curve/ratio fits with nulls
D. Subspace Detector          -> sparsity / participation / selectivity
E. Interaction Detector       -> factorial A/B/A+B causal test
F. Path Detector              -> ranked paths through the dependency DAG

Edges follow §11A allowed measures: intervention effect (causal dependence)
and activation correlation. All null comparisons live in controls.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# A. Activation graph builder
# --------------------------------------------------------------------------
@dataclass
class PATNode:
    node_id: str          # "embed","attn0","mlp1","resid1","final","L0.h3"
    kind: str             # embed|attn|mlp|resid|final|head
    layer: int = -1
    index: Optional[int] = None      # head index
    ablatable: bool = False          # can be zeroed via forward masks
    task_relevance: float = 0.0      # calib acc drop when ablated (points)
    selectivity: float = 0.0         # task drop - mean other-suite drop
    act: Optional[np.ndarray] = None  # pooled activations [N, D]


@dataclass
class Graph:
    nodes: Dict[str, PATNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], float] = field(default_factory=dict)  # u->v causal dep
    corr: Dict[Tuple[str, str], float] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    def add_node(self, n: PATNode):
        self.nodes[n.node_id] = n

    def add_edge(self, u: str, v: str, w: float):
        self.edges[(u, v)] = float(w)

    def downstream(self, u: str) -> List[str]:
        return [v for (a, v) in self.edges if a == u]

    def upstream(self, v: str) -> List[str]:
        return [a for (a, b) in self.edges if b == v]

    def depth_order(self) -> List[str]:
        order = ["embed"] + [n for n, nd in self.nodes.items() if nd.kind in ("attn", "mlp", "head")]
        order = [n for n in order if n in self.nodes]
        # interleave by layer then final/resid
        res = [n for n, nd in sorted(self.nodes.items(), key=lambda kv: (kv[1].layer, kv[0]))
               if nd.kind in ("attn", "mlp", "head")]
        return ["embed"] + res + ["final"]


def _pool(act: np.ndarray) -> np.ndarray:
    """[B, T, D] -> [B, D] mean over positions."""
    return act.reshape(act.shape[0], -1, act.shape[-1]).mean(1)


def _set_mask(model, node_id: str, off: bool):
    """Zero one graph node during forward (attn/mlp modules or single heads)."""
    import re
    if node_id.startswith("attn") or node_id.startswith("mlp"):
        l = int(re.match(r"^(attn|mlp)(\d+)$", node_id).group(2))
        mods = model.skip_modules.setdefault(l, [])
        name = node_id[:4] if node_id.startswith("attn") else "mlp"
        if off and name not in mods:
            mods.append(name)
        if not off:
            model.skip_modules.pop(l, None)
    elif ".h" in node_id:
        l, h = map(int, node_id.replace("L", "").split(".h"))
        if off:
            if model.head_mask is None:
                model.head_mask = np.ones((model.spec.layers, model.spec.heads), np.float32)
            model.head_mask[l, h] = 0.0
        else:
            model.head_mask = None


def collect_nodes(model, texts: List[str], tokenizer_cap: int = 32,
                  batch: int = 16) -> Dict[str, np.ndarray]:
    """Collect pooled activations at every collection point (§3 step 1)."""
    from benchmarking.suites import Suite
    enc = []
    for t in texts:
        ids = model.tokenizer.encode(t, max_len=tokenizer_cap)
        if len(ids) >= 4:
            enc.append(ids)
    L = model.spec.layers
    bags: Dict[str, List[np.ndarray]] = {k: [] for k in (
        ["embed", "final"] + [f"attn{l}" for l in range(L)]
        + [f"mlp{l}" for l in range(L)] + [f"resid{l}" for l in range(L)])}
    for i in range(0, len(enc), batch):
        chunk = enc[i:i + batch]
        M = max(len(s) for s in chunk)
        arr = np.zeros((len(chunk), M), np.int64)
        for r, s in enumerate(chunk):
            arr[r, :len(s)] = s
        _, cache = model.forward(arr, collect=True)
        bags["embed"].append(_pool(cache["embed"]))
        bags["final"].append(_pool(cache["final_hidden"]))
        for l in range(L):
            bags[f"attn{l}"].append(_pool(cache["attn_out"][l]))
            bags[f"mlp{l}"].append(_pool(cache["mlp_out"][l]))
            bags[f"resid{l}"].append(_pool(cache["hiddens"][l]))
    return {k: np.concatenate(v, 0) for k, v in bags.items() if v}


def _rel_change(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity of pooled activations (causal dependence proxy)."""
    va, vb = a.reshape(-1), b.reshape(-1)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(1.0 - float(np.dot(va, vb)) / (na * nb))


def build_activation_graph(model, task_texts: List[str], gen_texts: List[str],
                           relevance_fn=None, include_heads: bool = True,
                           log=None) -> Graph:
    """Nodes = collection points (+ heads); edges = causal dependence
    (downstream activation change when the upstream node is zeroed) for
    ablatable upstreams, correlation otherwise. Node task relevance and
    selectivity come from relevance_fn(node)->(relevance, selectivity)."""
    g = Graph()
    task_acts = collect_nodes(model, task_texts)
    gen_acts = collect_nodes(model, gen_texts)
    L, H = model.spec.layers, model.spec.heads
    g.add_node(PATNode("embed", "embed", -1, ablatable=False,
                       act=task_acts["embed"]))
    for l in range(L):
        g.add_node(PATNode(f"attn{l}", "attn", l, ablatable=True, act=task_acts[f"attn{l}"]))
        g.add_node(PATNode(f"mlp{l}", "mlp", l, ablatable=True, act=task_acts[f"mlp{l}"]))
        g.add_node(PATNode(f"resid{l}", "resid", l, ablatable=False, act=task_acts[f"resid{l}"]))
        if include_heads:
            for h in range(H):
                g.add_node(PATNode(f"L{l}.h{h}", "head", l, index=h, ablatable=True))
    g.add_node(PATNode("final", "final", L, ablatable=False, act=task_acts["final"]))

    # causal edges: zero upstream, measure downstream change (task condition)
    ablatable = [n for n, nd in g.nodes.items() if nd.ablatable]
    downstream_of = {}
    for l in range(L):
        downs = [f"mlp{l}", f"resid{l}"] + ([f"attn{l+1}", f"mlp{l+1}", f"resid{l+1}"]
                                            if l + 1 < L else ["final"])
        downstream_of[f"attn{l}"] = [f"mlp{l}", f"resid{l}"] + \
            ([f"attn{l+1}", f"resid{l+1}"] if l + 1 < L else ["final"])
        downstream_of[f"mlp{l}"] = [f"resid{l}"] + \
            ([f"attn{l+1}", f"resid{l+1}"] if l + 1 < L else ["final"])
        if include_heads:
            for h in range(H):
                downstream_of[f"L{l}.h{h}"] = [f"attn{l}", f"mlp{l}", f"resid{l}"]
    clean = task_acts
    for u in ablatable:
        _set_mask(model, u, True)
        try:
            masked = collect_nodes(model, task_texts)
        finally:
            _set_mask(model, u, False)
        for v in downstream_of.get(u, []):
            if v in masked:
                g.add_edge(u, v, _rel_change(clean[v], masked[v]))
    # structural correlation edges along the dataflow skeleton
    chain = ["embed"] + sum([[f"attn{l}", f"mlp{l}", f"resid{l}"] for l in range(L)], []) + ["final"]
    for a, b in zip(chain, chain[1:]):
        if a in clean and b in clean and (a, b) not in g.edges:
            g.corr[(a, b)] = float(abs(np.corrcoef(
                clean[a].reshape(len(clean[a]), -1).ravel(),
                clean[b].reshape(len(clean[b]), -1).ravel())[0, 1]))
    g.meta["task_texts_n"] = len(task_texts)
    g.meta["gen_texts_n"] = len(gen_texts)
    g.meta["gen_task_contrast"] = {
        n: float(np.linalg.norm(task_acts.get(n, np.zeros((1, 1))).mean(0)
                                - gen_acts[n].mean(0)) / (np.linalg.norm(gen_acts[n].mean(0)) + 1e-9))
        for n in list(task_acts)[:2 * L + 2]}
    if relevance_fn:
        for n in ablatable:
            rel, sel = relevance_fn(n)
            g.nodes[n].task_relevance = rel
            g.nodes[n].selectivity = sel
    if log:
        log(f"  graph: {len(g.nodes)} nodes, {len(g.edges)} causal edges, "
            f"{len(g.corr)} structural edges")
    return g


# --------------------------------------------------------------------------
# B. Motif detectors (§11B)
# --------------------------------------------------------------------------
def _paths(g: Graph, src: str = "embed", dst: str = "final") -> List[List[str]]:
    """All directed paths src->dst over causal + structural edges (tiny DAG)."""
    adj: Dict[str, List[str]] = {}
    for (u, v) in list(g.edges) + list(g.corr):
        adj.setdefault(u, []).append(v)
    out, stack = [], [(src, [src])]
    while stack:
        u, path = stack.pop()
        if u == dst:
            out.append(path)
            continue
        for v in adj.get(u, []):
            if v not in path:
                stack.append((v, path + [v]))
    return out


def detect_chain(g: Graph, tau: float = 0.02) -> List[List[str]]:
    """P021/P028: task-dependent directed chains through the graph."""
    paths = _paths(g)
    scored = []
    for p in paths:
        ws = [g.edges.get((a, b), 0.0) for a, b in zip(p, p[1:])]
        if ws and min(ws) >= tau:
            scored.append(p)
    return sorted(scored, key=lambda p: -min(g.edges.get((a, b), 0.0)
                                             for a, b in zip(p, p[1:])))


def detect_fan_in(g: Graph, k: int = 2) -> Dict[str, List[str]]:
    """P024/P026: nodes with >=k ablatable upstreams (convergence points)."""
    out = {}
    for n, nd in g.nodes.items():
        ups = [u for u in g.upstream(n) if g.nodes[u].ablatable
               and g.edges.get((u, n), 0) > 0]
        if len(ups) >= k:
            out[n] = ups
    return out


def detect_fan_out(g: Graph, k: int = 2) -> Dict[str, List[str]]:
    """P025: nodes with >=k ablatable downstreams (branching points)."""
    out = {}
    for n, nd in g.nodes.items():
        downs = [v for v in g.downstream(n) if g.nodes[v].kind in ("attn", "mlp")
                 and g.edges.get((n, v), 0) > 0]
        if len(downs) >= k:
            out[n] = downs
    return out


def detect_hub(g: Graph) -> Optional[str]:
    """P030: max weighted-degree node (causal mass in+out)."""
    wd = {n: sum(w for (u, v), w in g.edges.items() if u == n or v == n)
          for n in g.nodes}
    if not wd:
        return None
    return max(wd, key=wd.get)


def edge_betweenness(g: Graph) -> Dict[Tuple[str, str], float]:
    """P038: fraction of embed->final paths using each edge."""
    paths = _paths(g)
    if not paths:
        return {}
    cnt: Dict[Tuple[str, str], float] = {}
    for p in paths:
        for a, b in zip(p, p[1:]):
            cnt[(a, b)] = cnt.get((a, b), 0) + 1.0
    return {e: c / len(paths) for e, c in cnt.items()}


def detect_bottleneck(g: Graph) -> Optional[str]:
    """P037: the node whose removal destroys the most embed->final paths
    (min-cut flavored); ties broken by path traffic."""
    paths = _paths(g)
    if not paths:
        return None
    traffic: Dict[str, float] = {}
    for p in paths:
        for n in p[1:-1]:
            traffic[n] = traffic.get(n, 0) + 1.0
    if not traffic:
        return None
    def criticality(n):
        surviving = sum(1 for p in paths if n not in p)
        return (surviving, -traffic[n])   # minimize surviving, maximize traffic
    return min(traffic, key=criticality)


def detect_redundant_paths(g: Graph, top: int = 3) -> List[List[List[str]]]:
    """P040/P093: groups of edge-disjoint high-scoring paths embed->final."""
    paths = detect_chain(g, tau=0.0)
    used: set = set()
    groups: List[List[List[str]]] = []
    for p in sorted(paths, key=lambda p: -min([g.edges.get((a, b), 0)
                                               for a, b in zip(p, p[1:])] or [0])):
        es = set(zip(p, p[1:]))
        if es & used:
            continue
        used |= es
        groups.append([p])
        if len(groups) >= top:
            break
    return groups


def detect_communities(g: Graph) -> Dict[str, int]:
    """P036/P048: label-propagation communities on the undirected version."""
    labels = {n: i for i, n in enumerate(g.nodes)}
    und: Dict[str, set] = {n: set() for n in g.nodes}
    for (u, v) in list(g.edges) + list(g.corr):
        und[u].add(v)
        und[v].add(u)
    for _ in range(32):
        new = dict(labels)
        for n in g.nodes:
            if not und[n]:
                continue
            counts: Dict[int, int] = {}
            for m in und[n]:
                counts[labels[m]] = counts.get(labels[m], 0) + 1
            new[n] = max(counts, key=counts.get)
        if new == labels:
            break
        labels = new
    return labels


def detect_diamonds(g: Graph) -> List[Tuple[str, str, List[str]]]:
    """P091/P098: fork->process->recombine motifs (u->m->v via >=2 mids)."""
    out = []
    for u in g.nodes:
        mids = [m for m in g.downstream(u)
                if any(m in g.downstream(x) for x in g.downstream(u)) ]
        for v in set(b for m in g.downstream(u) for b in g.downstream(m)):
            if v == u:
                continue
            shared = [m for m in g.downstream(u) if v in g.downstream(m)]
            if len(shared) >= 2:
                out.append((u, v, shared))
    uniq, seen = [], set()
    for u, v, ms in out:
        k = (u, v, tuple(sorted(ms)))
        if k not in seen:
            seen.add(k)
            uniq.append((u, v, ms))
    return uniq


def conditional_edges(g: Graph, task_acts: Dict[str, np.ndarray],
                      gen_acts: Dict[str, np.ndarray],
                      model, task_texts, gen_texts) -> Dict[Tuple[str, str], float]:
    """P079/P080: routing = |causal weight(task condition) - causal weight(general)|."""
    ablatable = [n for n, nd in g.nodes.items() if nd.ablatable]
    gen_change: Dict[Tuple[str, str], float] = {}
    clean_gen = gen_acts
    for u in ablatable:
        _set_mask(model, u, True)
        try:
            masked = collect_nodes(model, gen_texts)
        finally:
            _set_mask(model, u, False)
        for v in g.downstream(u):
            if v in masked:
                gen_change[(u, v)] = _rel_change(clean_gen[v], masked[v])
    routed = {}
    for e, wt in g.edges.items():
        wg = gen_change.get(e, 0.0)
        routed[e] = abs(wt - wg)
    return routed


# --------------------------------------------------------------------------
# C. Sequence / curve / ratio detectors (§11C) — Phase B hypotheses
# --------------------------------------------------------------------------
def _norm_dist(seq: List[float], target: List[float]) -> float:
    n, m = len(seq), len(target)
    if n == 0 or m == 0:
        return 1e9
    scale = (np.mean(seq) / np.mean(target)) if np.mean(target) else 1.0
    t = np.array(target) * scale
    k = min(n, m)
    return float(np.sqrt(np.mean((np.array(seq[:k]) - t[:k]) ** 2)) / (np.mean(np.abs(seq[:k])) + 1e-9))


def fib_like(n: int) -> List[float]:
    a, b, out = 1.0, 1.0, []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


def lucas_like(n: int) -> List[float]:
    a, b, out = 2.0, 1.0, []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


def sequence_gap_test(positions: List[int], K: int = 200, seed: int = 0,
                      span: Optional[int] = None) -> Dict:
    """P001/P004/P010/P011/P013: compare gap structure of significant-region
    positions against named sequences; null = random position sets (§12)."""
    pos = sorted(set(int(p) for p in positions))
    if len(pos) < 2:
        return {"significant": False, "reason": "fewer than 2 positions"}
    span = span or (max(pos) + 1)
    gaps = np.diff(pos).astype(float).tolist()
    rng = np.random.default_rng(seed)
    targets = {
        "fibonacci": fib_like(len(gaps)), "lucas": lucas_like(len(gaps)),
        "uniform": [float(np.mean(gaps))] * len(gaps),
        "prime": [float(p) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:len(gaps)]],
        "geometric": [float(2 ** i) for i in range(len(gaps))],
    }
    dists = {k: _norm_dist(gaps, v) for k, v in targets.items()}
    best_named = min(dists, key=dists.get)
    # null: random position sets of same size over same span
    null_d = []
    for _ in range(K):
        rp = sorted(rng.choice(span, size=len(pos), replace=False).tolist())
        null_d.append(_norm_dist(np.diff(rp).astype(float).tolist(), targets[best_named]))
    stat = dists[best_named]
    p = float((1 + sum(1 for d in null_d if d <= stat)) / (K + 1))
    return {"significant": p < 0.05, "p": round(p, 4), "best_fit": best_named,
            "distances": {k: round(v, 3) for k, v in dists.items()},
            "null_median": round(float(np.median(null_d)), 3),
            "gaps": gaps, "positions": pos}


def ratio_enrichment(ratios: List[float], target: float = 1.6180339887,
                     eps: float = 0.06, K: int = 200, seed: int = 0) -> Dict:
    """P002/P065: fraction of ratio pairs within target±eps vs random ratios."""
    r = np.asarray([x for x in ratios if 0.01 < x < 100.0])
    if len(r) < 2:
        return {"significant": False, "reason": "too few ratios"}
    hits = int(np.sum(np.abs(r - target) < eps * target))
    frac = hits / len(r)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(K):
        rr = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=len(r)))
        null.append(float(np.mean(np.abs(rr - target) < eps * target)))
    p = float((1 + sum(1 for x in null if x >= frac)) / (K + 1))
    return {"significant": p < 0.05, "p": round(p, 4), "phi_fraction": round(frac, 4),
            "null_median": round(float(np.median(null)), 4), "n_ratios": int(len(r))}


def magnitude_family_fit(mags: np.ndarray) -> Dict:
    """P005/P012: power-law vs exponential vs lognormal vs uniform (sorted mags)."""
    m = np.sort(np.asarray(mags, float))[::-1]
    m = m[m > 0]
    if len(m) < 4:
        return {"best": "insufficient", "fits": {}}
    x = np.arange(1, len(m) + 1)
    fits = {}
    for name, fx in [("power_law", np.log(x)), ("exponential", x)]:
        A = np.vstack([fx, np.ones_like(fx)]).T
        coef, res_, *_ = np.linalg.lstsq(A, np.log(m), rcond=None)
        pred = A @ coef
        ss = 1 - np.sum((np.log(m) - pred) ** 2) / (np.sum((np.log(m) - np.log(m).mean()) ** 2) + 1e-12)
        fits[name] = {"r2": round(float(ss), 4), "exponent": round(float(coef[0]), 4)}
    mu = float(np.mean(np.log(m[m > 0])))
    fits["lognormal"] = {"sigma": round(float(np.std(np.log(m))), 4)}
    fits["uniform"] = {"cv": round(float(np.std(m) / (np.mean(m) + 1e-12)), 4)}
    best = max(["power_law", "exponential"], key=lambda k: fits[k]["r2"])
    return {"best": best if fits[best]["r2"] > 0.6 else "weak_fit",
            "fits": fits, "top20pct_share": round(float(m[:max(1, len(m)//5)].sum() / m.sum()), 4)}


def depth_curve_fit(depths: np.ndarray, vals: np.ndarray) -> Dict:
    """P006/P007/P008: exp/log/sqrt/linear contribution-vs-depth fits."""
    d = np.asarray(depths, float); v = np.asarray(vals, float)
    if len(d) < 3:
        return {"best": "insufficient", "fits": {}}
    out = {}
    for name, tx in [("linear", d), ("log", np.log1p(d)), ("sqrt", np.sqrt(d)),
                     ("exp", d)]:
        A = np.vstack([tx, np.ones_like(tx)]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        pred = A @ coef
        r2 = 1 - np.sum((v - pred) ** 2) / (np.sum((v - v.mean()) ** 2) + 1e-12)
        out[name] = {"r2": round(float(r2), 4), "slope": round(float(coef[0]), 5)}
    if abs(v.std()) < 1e-9:
        return {"best": "flat", "fits": out}
    best = max(out, key=lambda k: out[k]["r2"])
    return {"best": best if out[best]["r2"] > 0.5 else "weak_fit", "fits": out}


# --------------------------------------------------------------------------
# D. Subspace detector (§11D)
# --------------------------------------------------------------------------
def subspace_stats(acts: np.ndarray) -> Dict:
    X = acts - acts.mean(0, keepdims=True)
    if X.shape[0] < 4:
        return {"effective_rank": 0.0, "participation_ratio": 0.0}
    C = (X.T @ X) / (X.shape[0] - 1)
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum() + 1e-12
    pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))
    erank = float(np.exp(-np.sum((ev / tot) * np.log(ev / tot + 1e-12))))
    top5 = float(ev[:5].sum() / tot)
    return {"effective_rank": round(erank, 2), "participation_ratio": round(pr, 2),
            "top5_energy": round(top5, 4), "dim": int(X.shape[1])}


def hoyer_sparsity(v: np.ndarray) -> float:
    x = np.abs(np.asarray(v, float).ravel())
    n = len(x)
    if n == 0 or np.max(x) < 1e-12:
        return 0.0
    l1, l2 = x.sum(), np.sqrt((x ** 2).sum())
    return float((math.sqrt(n) - l1 / (l2 + 1e-12)) / (math.sqrt(n) - 1 + 1e-12))


# --------------------------------------------------------------------------
# E. Interaction detector (§11E) — factorial causal test
# --------------------------------------------------------------------------
def interaction_factorial(model, mask_a, mask_b, eval_fn) -> Dict:
    """interaction = perf(A+B removed) - perf(A) - perf(B) + baseline.
    mask_* are callables that set/clear masks on the model."""
    model.skip_layers, model.skip_modules, model.head_mask = [], {}, None
    model.mlp_channel_mask = None
    try:
        base = eval_fn()
        mask_a(True)
        a = eval_fn()
        mask_a(False)
        mask_b(True)
        b = eval_fn()
        mask_b(False)
        mask_a(True); mask_b(True)
        ab = eval_fn()
        mask_a(False); mask_b(False)
    finally:
        model.skip_layers, model.skip_modules, model.head_mask = [], {}, None
        model.mlp_channel_mask = None
    return {"baseline": round(float(base), 2), "a_alone": round(float(a), 2),
            "b_alone": round(float(b), 2), "ab": round(float(ab), 2),
            "interaction": round(float(ab - a - b + base), 2)}


# --------------------------------------------------------------------------
# F. Path detector (§11F)
# --------------------------------------------------------------------------
def rank_paths(g: Graph, top: int = 6) -> List[Dict]:
    paths = _paths(g)
    scored = []
    for p in paths:
        ws = [g.edges.get((a, b), 0.0) for a, b in zip(p, p[1:])]
        if not ws:
            continue
        sel = np.mean([g.nodes[n].selectivity for n in p[1:-1]]) if len(p) > 2 else 0.0
        scored.append({"path": p, "min_edge": round(float(min(ws)), 4),
                       "mean_edge": round(float(np.mean(ws)), 4),
                       "mean_selectivity": round(float(sel), 2),
                       "score": round(float(min(ws) * (1 + max(sel, 0))), 5)})
    return sorted(scored, key=lambda d: -d["score"])[:top]
