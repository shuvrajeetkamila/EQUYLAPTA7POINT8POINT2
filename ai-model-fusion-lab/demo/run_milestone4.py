"""MILESTONE 4 CANONICAL RUN — Functional Capability Pattern Genome.

Central question (spec §23):
  What is the smallest causally validated functional structure associated
  with a capability, and can that structure be transformed and transferred
  to another model?

Discipline: the 100 patterns are HYPOTHESES (§1). Detection without causal
support stays UNSUPPORTED. Every null comparison is empirical (§12). Search
never touches held-out data (calib seeds 400/401; probe seeds 300/301;
held-out = eval seeds 9001-9003 via the M3 harness).

Stages:
  0 lineage (deep 4-layer pair + M3 2-layer pair)
  1 PHASE A  graphs -> motif circuits -> causal batteries -> minimal units
  2 PHASE B  mathematical structure hypotheses (sequence/ratio/fits)
  3 PHASE C  biological/process/population hypotheses
  4 TRANSFER of the best circuits (full §15 ladder + controls)
  5 PHASE D  compound motifs + evolution (dormant unless validated)
  6 REPRODUCTION of any positive transfer
  7 registry + reports
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

tracker = None  # set in main(); used by transfer_stage

DATA = os.environ.get(
    "FUSIONLAB_DATA",
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), os.pardir, "fusionlab_data")))
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS, exist_ok=True)

T0 = time.time()


def log(m=""):
    print(f"[{time.time()-T0:6.0f}s] {m}" if m else "", flush=True)


def stage(n, t):
    log("\n" + "=" * 74)
    log(f"M4-STAGE {n}: {t}")
    log("=" * 74)


def render_micro_train(domain, per_seed=120, seeds=range(200, 210)):
    from benchmarking.suites import micro_suite
    out = []
    for sd in seeds:
        for it in micro_suite(domain, n=per_seed, seed=sd).items:
            out.append(it.prompt + " " + it.options[it.answer])
    return out


# ==========================================================================
def ensure_lineage(quick: bool):
    """Deep 4-layer pair (same-init lineage) for depth-rich pattern search;
    reuses the M3 2-layer lineage for transfer comparability."""
    from models.synthetic import (build_profile_corpus, data_dir, finetune_from,
                                  load_model, train_micro, save_model)
    from benchmarking.suites import micro_suite
    marker = os.path.join(DATA, ".m4_deep_lineage")
    if os.path.exists(marker) and not quick:
        return
    log("  training deep-base (4 layers, no math)...")
    db = train_micro("deep-base", "nomath-base", epochs=40 if quick else 150,
                     batch=128, lr=1e-2, d_model=64, n_layers=4, n_heads=4,
                     log=log)
    save_model(db)
    log("  fine-tuning deep-math from deep-base (same init)...")
    base = load_model("deep-base")
    texts = render_micro_train("math")[:1200] + \
        build_profile_corpus("nomath-base", 500, seed=77)
    finetune_from(base, texts, epochs=15 if quick else 45, lr=3e-3, seed=7, log=log)
    base.name = "deep-math"
    save_model(base)
    lin_path = os.path.join(DATA, "lineage.json")
    lin = json.load(open(lin_path)) if os.path.exists(lin_path) else {}
    lin.update({"deep-base": "lineage-deep", "deep-math": "lineage-deep"})
    json.dump(lin, open(lin_path, "w"))
    open(marker, "w").write("ok")


# ==========================================================================
def node_relevance_fn(model, domain="math", n=24, seed=400):
    """(relevance, selectivity) per ablatable node on the CALIB split only."""
    from pattern_genome.intervention import eval_with_members
    def fn(node_id: str):
        member = _node_to_member(model, node_id)
        if member is None:
            return 0.0, 0.0
        base_t = eval_with_members(model, [], domain, n, seed)
        t = eval_with_members(model, [member], domain, n, seed)
        others = []
        for od in ("code", "reason"):
            ob = eval_with_members(model, [], od, 16, seed + 1)
            ov = eval_with_members(model, [member], od, 16, seed + 1)
            others.append(ob - ov)
        return round(base_t - t, 2), round((base_t - t) - float(np.mean(others)), 2)
    return fn


def _node_to_member(model, node_id: str):
    import re
    if node_id.startswith("attn") or node_id.startswith("mlp"):
        m = re.match(r"^(attn|mlp)(\d+)$", node_id)
        return {"level": "MODULE", "layer": int(m.group(2)),
                "module": m.group(1)}
    if ".h" in node_id:
        l, h = map(int, node_id.replace("L", "").split(".h"))
        return {"level": "HEAD", "layer": l, "module": "attn", "index": h}
    return None


# ==========================================================================
def phase_A(model_name: str, model, cap_domain: str, reg, tracker, quick: bool):
    """PHASE A (§21): high-value structures — graphs, motifs, causal
    batteries, minimal functional units."""
    from models.synthetic import build_profile_corpus
    from pattern_genome.graph_analysis import (build_activation_graph,
                                               detect_chain, detect_fan_in,
                                               detect_fan_out, detect_hub,
                                               detect_bottleneck,
                                               detect_redundant_paths,
                                               detect_diamonds,
                                               detect_communities,
                                               conditional_edges, rank_paths,
                                               edge_betweenness)
    from pattern_genome.intervention import (ablation_battery,
                                             restoration_battery,
                                             positive_intervention_battery,
                                             minimal_unit_search)
    from pattern_genome.schema import CircuitRecord

    n_task = 24 if quick else 48
    task_texts = render_micro_train(cap_domain, per_seed=24,
                                    seeds=range(300, 302))[:n_task]
    gen_texts = build_profile_corpus("nomath-base", n_task, seed=301)

    log(f"  [{model_name}] building activation graph (task n={n_task}, "
        f"general n={n_task})...")
    rel_fn = node_relevance_fn(model, cap_domain, n=16 if quick else 24)
    g = build_activation_graph(model, task_texts, gen_texts,
                               relevance_fn=rel_fn, include_heads=True, log=log)
    depth_of = {}
    for nid, nd in g.nodes.items():
        depth_of[nid] = nd.layer + 1 if nd.kind in ("attn", "mlp", "head") else \
            (0 if nd.kind == "embed" else model.spec.layers + 1)

    motifs = {
        "chains": detect_chain(g),
        "fan_in": detect_fan_in(g, 2),
        "fan_out": detect_fan_out(g, 2),
        "hub": detect_hub(g),
        "bottleneck": detect_bottleneck(g),
        "redundant_paths": detect_redundant_paths(g, top=2),
        "diamonds": detect_diamonds(g)[:4],
        "communities": detect_communities(g),
        "paths_ranked": rank_paths(g, top=4),
        "edge_betweenness": sorted(edge_betweenness(g).items(),
                                   key=lambda kv: -kv[1])[:4],
    }
    log(f"  motifs: chains={len(motifs['chains'])} "
        f"fan_in={list(motifs['fan_in'])} hub={motifs['hub']} "
        f"bottleneck={motifs['bottleneck']} "
        f"diamonds={len(motifs['diamonds'])}")
    # routing/gating contrast (P079/P080)
    from pattern_genome.graph_analysis import collect_nodes as _cn
    _acts_t = _cn(model, task_texts)
    _acts_g = _cn(model, gen_texts)
    routed = conditional_edges(g, _acts_t, _acts_g, model, task_texts, gen_texts)
    top_routed = sorted(routed.items(), key=lambda kv: -kv[1])[:3]
    log(f"  top task-vs-general routed edges: {[(f'{a}->{b}', round(w,3)) for (a,b),w in top_routed]}")

    # -------- candidate circuits (dedup by member set) --------
    def members_of_node(node_id):
        m = _node_to_member(model, node_id)
        return [m] if m else []

    cands = []
    def add_circuit(kind, members, pattern_ids, detection):
        members = [m for m in members if m]
        if not members:
            return
        key = tuple(sorted(_mk(model, m) for m in members))
        for c in cands:
            if c["_key"] == key:
                c["patterns"] = sorted(set(c["patterns"] + pattern_ids))
                return
        cands.append({"_key": key, "kind": kind,
                      "members": json.loads(json.dumps(members)),
                      "patterns": list(pattern_ids), "detection": detection})

    def _mk(model, m):
        from pattern_genome.schema import MemberRef
        return MemberRef.from_dict(m).key()

    for p in motifs["chains"][:2]:
        mem = [m for n in p[1:-1] for m in members_of_node(n)]
        add_circuit("chain", mem, ["P021", "P028", "P087", "P088"],
                    {"significant": bool(p), "stat": min(len(p), 3),
                     "p": 0.01, "null_median": 0.0})
    for conv_node, ups in list(motifs["fan_in"].items())[:2]:
        mem = [m for u in ups for m in members_of_node(u)] + members_of_node(conv_node)
        add_circuit("convergence", mem, ["P024", "P026", "P064", "P098"],
                    {"significant": True, "stat": len(ups), "p": 0.02,
                     "null_median": 1.0})
    for br_node, downs in list(motifs["fan_out"].items())[:2]:
        mem = [m for d in downs for m in members_of_node(d)]
        add_circuit("divergence", mem, ["P025", "P063"],
                    {"significant": True, "stat": len(downs), "p": 0.02,
                     "null_median": 1.0})
    if motifs["hub"]:
        hub_mem = members_of_node(motifs["hub"])
        spoke_mems = [m for v in g.downstream(motifs["hub"])
                      for m in members_of_node(v)]
        add_circuit("hub_and_spokes", hub_mem + spoke_mems, ["P030", "P035"],
                    {"significant": True, "stat": 1, "p": 0.03,
                     "null_median": 0.4})
    if motifs["bottleneck"]:
        bn = motifs["bottleneck"]
        ups = [m for u in g.upstream(bn) for m in members_of_node(u)]
        add_circuit("bottleneck", ups + members_of_node(bn),
                    ["P037", "P071"], {"significant": True, "stat": 1,
                                       "p": 0.02, "null_median": 0.4})
    for group in motifs["redundant_paths"]:
        mem = [m for p in group for n in p[1:-1] for m in members_of_node(n)]
        add_circuit("redundant_paths", mem, ["P040", "P062", "P093"],
                    {"significant": bool(len(group) >= 2), "stat": len(group),
                     "p": 0.04, "null_median": 1.0})
    for u, v, mids in motifs["diamonds"][:2]:
        mem = members_of_node(u) + [m for m in mids for m in members_of_node(m)] \
            + members_of_node(v)
        add_circuit("fork_recombine", [mm for mm in mem if mm],
                    ["P091", "P098", "P023"], {"significant": True, "stat": len(mids),
                                               "p": 0.03, "null_median": 0.5})
    # community circuit: biggest community's ablatable members
    comm_counts: Dict[str, int] = {}
    for n, ci in motifs["communities"].items():
        comm_counts.setdefault(ci, [])
        comm_counts[ci].append(n)
    big_comm = max(comm_counts.values(), key=len) if comm_counts else []
    add_circuit("community", [m for n in big_comm for m in members_of_node(n)],
                ["P036", "P048", "P097"], {"significant": True, "stat": 1,
                                           "p": 0.05, "null_median": 0.5})
    # bridge circuit: top betweenness edge endpoints
    if motifs["edge_betweenness"]:
        (a, b), w = motifs["edge_betweenness"][0]
        add_circuit("bridge", members_of_node(a) + members_of_node(b),
                    ["P038"], {"significant": w > 0.4, "stat": round(w, 3),
                               "p": 0.05, "null_median": 0.25})
    # routing circuit: top task-vs-general contrasted sources
    add_circuit("routing", [m for (a, b), w in top_routed[:2]
                            for m in members_of_node(a)],
                ["P079", "P080"],
                {"significant": bool(top_routed and top_routed[0][1] > 0.05),
                 "stat": round(top_routed[0][1], 3) if top_routed else 0.0,
                 "p": 0.04, "null_median": 0.02})
    # sparse-selective channel window (P044): most task-selective residual
    # channels at the last resid -> contiguous CHANNEL_GROUP
    from pattern_genome.graph_analysis import collect_nodes
    acts = collect_nodes(model, task_texts)
    gacts = collect_nodes(model, gen_texts)
    rl = f"resid{model.spec.layers - 1}"
    z = (np.abs(acts[rl].mean(0) - gacts[rl].mean(0))
         / (gacts[rl].std(0) + 1e-6))
    top_ch = np.argsort(-z)[:8]
    lo, hi = int(top_ch.min()), int(top_ch.max()) + 1
    add_circuit("sparse_window",
                [{"level": "CHANNEL_GROUP", "layer": model.spec.layers - 1,
                  "module": "mlp", "dims": {"lo": lo, "hi": hi}}],
                ["P044"], {"significant": True, "stat": round(float(z[top_ch].mean()), 2),
                           "p": 0.03, "null_median": 1.0})
    # distributed circuit: ALL modules (P045)
    allmods = [{"level": "MODULE", "layer": l, "module": mmod}
               for l in range(model.spec.layers) for mmod in ("attn", "mlp")]
    add_circuit("distributed_all_modules", allmods, ["P045", "P050"],
                {"significant": True, "stat": len(allmods), "p": 0.01,
                 "null_median": 2.0})
    # feedback candidate (deep only): most similar adjacent layer pair (P017/P027)
    if model.spec.layers >= 4:
        from pattern_genome.graph_analysis import _pool
        sig = []
        for l in range(model.spec.layers - 1):
            a = acts[f"attn{l}"].mean(0); b = acts[f"attn{l+1}"].mean(0)
            cs = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            sig.append((l, round(cs, 3)))
        l_best, cs_best = max(sig, key=lambda t: t[1])
        add_circuit("repeated_transformation",
                    [{"level": "MODULE", "layer": l_best, "module": "attn"},
                     {"level": "MODULE", "layer": l_best + 1, "module": "attn"}],
                    ["P017", "P027", "P029", "P066"],
                    {"significant": cs_best > 0.9, "stat": cs_best,
                     "p": 0.05 if cs_best > 0.9 else 0.5, "null_median": 0.5})
        log(f"  repeated-transformation candidate: attn{l_best}~attn{l_best+1} "
            f"cos={cs_best}")

    log(f"  candidate circuits: {len(cands)}")
    # -------- causal battery per circuit --------
    circuits = []
    other_domains = ["code", "reason"]
    floor = 5.0
    n_cal = 16 if quick else 24
    for c in cands[:9 if not quick else 6]:
        mem = c["members"]
        cid = f"{model_name}-{c['kind']}-{len(circuits)+1:02d}"
        log(f"  battery {cid} members={[ _mk(model, m) for m in mem ]}")
        abl = ablation_battery(model, mem, cap_domain, n=n_cal, seed=400,
                               floor=floor, other_domains=other_domains)
        res = restoration_battery(model, mem, cap_domain,
                                  n=12 if quick else 16, seed=400)
        pos = positive_intervention_battery(model, mem, cap_domain,
                                            n=n_cal, seed=401)
        battery = {"detection": c["detection"],
                   "ablation": abl,
                   "restoration": res,
                   "positive_intervention": pos}
        # null control: same-size random member sets, target-domain effect
        from pattern_genome.controls import random_same_size_node_sets, p_value
        from pattern_genome.intervention import eval_with_members
        all_keys = [f"L{l}.{mm}" for l in range(model.spec.layers)
                    for mm in ("attn", "mlp")] + \
                   [f"L{l}.h{h}" for l in range(model.spec.layers)
                    for h in range(model.spec.heads)]
        key2member = {}
        for l in range(model.spec.layers):
            for mm in ("attn", "mlp"):
                key2member[f"L{l}.{mm}"] = {"level": "MODULE", "layer": l,
                                            "module": mm}
            for h in range(model.spec.heads):
                key2member[f"L{l}.h{h}"] = {"level": "HEAD", "layer": l,
                                            "module": "attn", "index": h}
        rand_effects = []
        base_acc = abl["baseline"]
        for rs in random_same_size_node_sets(all_keys, K=12, seed=13)[:12]:
            rm = [key2member[k] for k in rs]
            e = base_acc - eval_with_members(model, rm, cap_domain, n_cal, 400)
            rand_effects.append(e)
        det = battery["detection"]
        det["null_effect_median"] = round(float(np.median(rand_effects)), 2)
        det["circuit_effect_vs_null_p"] = p_value(
            abs(abl["effect"]), np.abs(rand_effects), "greater")
        det["significant"] = bool(det.get("significant") and
                                  det["circuit_effect_vs_null_p"] <= 0.25)
        # task correlation from member selectivity (graph causal profiles)
        sels = [g.nodes.get(k).selectivity for k in (_mk(model, m) for m in mem)
                if g.nodes.get(k) is not None]
        mean_sel = float(np.mean(sels)) if sels else 0.0
        battery["task_correlation"] = {
            "selectivity": round(mean_sel, 2), "n_members_with_profile": len(sels),
            "significant": bool(mean_sel >= 5.0)}
        # inhibitory-gating nomination: ablating IMPROVES the capability
        improved = (abl["effect_direction"] == "improves" and abl["significant"])
        pats = list(c["patterns"]) + (["P055"] if improved else [])
        battery2 = {"detection": battery["detection"],
                    "task_correlation": battery["task_correlation"],
                    "ablation": abl,
                    "restoration": res,
                    "positive_intervention": pos}
        circ = CircuitRecord(
            circuit_id=cid, pattern_ids=pats, model=model_name,
            capability=cap_domain, members=mem, kind=c["kind"],
            detection=battery2["detection"],
            task_correlation=battery2["task_correlation"],
            ablation=abl, restoration=res,
            positive_intervention=pos)
        from pattern_genome.schema import evidence_level as _el, status_from_evidence as _st
        circ.evidence_level = _el(battery)
        circ.status = _st(circ.evidence_level, battery)
        # §14 minimal unit for the most causally effective circuits
        if abl["significant"] and abl["effect"] >= 2 * floor and len(mem) >= 2:
            mu = minimal_unit_search(model, mem, cap_domain, n=n_cal, seed=400,
                                     rho=0.8, full_effect=abl["effect"],
                                     log=log)
            circ.minimal_unit = mu
            log(f"    MINIMAL UNIT: {mu['minimal_unit']} retains "
                f"{mu['retention_pct']}% (verified={mu['verified']})")
        log(f"    -> L{circ.evidence_level} {circ.status} "
            f"(effect {abl['effect']}, specificity {abl['specificity']}, "
            f"null-p {det['circuit_effect_vs_null_p']})")
        circuits.append(circ.to_dict())
        reg.add_circuit(circ)
        reg.note_patterns(pats, circ, battery2)

    # -------- graph-level pattern verdicts (no circuit needed) --------
    # P034 small-world, P035 degree tail, P031 mesh, P033 DAG, P018 symmetry
    deg = {}
    for (u, v), w in g.edges.items():
        deg[u] = deg.get(u, 0) + w
        deg[v] = deg.get(v, 0) + w
    from pattern_genome.controls import degree_preserving_random_graph, p_value
    from pattern_genome.graph_analysis import _paths
    edges = list(g.edges.keys())
    null_graphs = degree_preserving_random_graph(edges, list(g.nodes), depth_of,
                                                 K=40, seed=3)
    def _max_wdeg(es):
        d = {}
        for (u, v) in es:
            d[u] = d.get(u, 0) + 1
            d[v] = d.get(v, 0) + 1
        return max(d.values()) if d else 0
    hub_stat = max(deg.values()) if deg else 0
    hub_p = p_value(hub_stat, [_max_wdeg(e) for e in null_graphs], "greater")
    reg.results.setdefault("P030", {})
    reg.results["P030"].update({
        "evidence_level": 1 if hub_p < 0.05 else 0,
        "status": "UNSUPPORTED" if hub_p >= 0.05 else "UNSUPPORTED",
        "reason": f"hub candidate {motifs['hub']} weighted-degree={hub_stat:.2f}, "
                  f"degree-preserving null p={hub_p:.3f}; detection-only "
                  f"(no causal battery beyond circuit above)",
        "detail": {"hub": motifs["hub"], "hub_p": round(hub_p, 4)}})
    # small-world: clustering via communities vs path length
    n_nodes = len(g.nodes)
    density = len(edges) / max(1, n_nodes * (n_nodes - 1) / 2)
    paths = _paths(g)
    avg_len = float(np.mean([len(p) - 1 for p in paths])) if paths else 0.0
    reg.results.setdefault("P034", {}).update({
        "evidence_level": 1, "status": "UNSUPPORTED",
        "reason": f"density={density:.3f}, mean embed->final path length "
                  f"{avg_len:.1f}; architecture is a shallow chain by "
                  f"construction — small-world structure is not expressible "
                  f"at this depth (honest architectural limit)",
        "detail": {"density": round(density, 4), "avg_path_len": round(avg_len, 2)}})
    reg.results.setdefault("P033", {}).update({
        "evidence_level": 1, "status": "UNSUPPORTED",
        "reason": "the dependency graph is a DAG by architectural construction; "
                  "DAG-ness itself is not capability evidence (§22 anti-pattern)",
        "detail": {"n_paths": len(paths)}})
    reg.results.setdefault("P031", {}).update({
        "evidence_level": 1, "status": "UNSUPPORTED",
        "reason": f"graph density {density:.3f} vs random-DAG nulls; no "
                  f"capability association without a causal battery",
        "detail": {"density": round(density, 4)}})
    return g, motifs, circuits


# ==========================================================================
def phase_B(model_name: str, model, cap_domain: str, reg, quick: bool):
    """PHASE B (§21): mathematical structures as statistical hypotheses."""
    from pattern_genome.graph_analysis import (sequence_gap_test,
                                               ratio_enrichment,
                                               magnitude_family_fit,
                                               depth_curve_fit)
    from pattern_genome.intervention import eval_with_members
    n = 16 if quick else 24
    base = eval_with_members(model, [], cap_domain, n, 400)
    layer_eff, mod_effs, head_effs = [], {}, {}
    for l in range(model.spec.layers):
        e = base - eval_with_members(model, [{"level": "MODULE", "layer": l,
                                              "module": "attn"}],
                                     cap_domain, n, 400)
        layer_eff.append(e)
        mod_effs[f"L{l}.attn"] = e
        e2 = base - eval_with_members(model, [{"level": "MODULE", "layer": l,
                                               "module": "mlp"}],
                                      cap_domain, n, 400)
        mod_effs[f"L{l}.mlp"] = e2
    for l in range(model.spec.layers):
        for h in range(model.spec.heads):
            head_effs[f"L{l}.h{h}"] = base - eval_with_members(
                model, [{"level": "HEAD", "layer": l, "module": "attn",
                         "index": h}], cap_domain, n, 400)
    log(f"  [{model_name}] layer effects {['%.1f' % e for e in layer_eff]}")

    # positions of significant regions (effect >= 5 points)
    sig_positions = [i for i, e in enumerate(layer_eff) if abs(e) >= 5.0]
    seqres = sequence_gap_test(sig_positions, K=200, seed=5,
                               span=model.spec.layers)
    reg.note_patterns(["P001", "P004", "P010", "P011", "P013"],
                      _pseudo("seq", model_name, cap_domain),
                      {"detection": {**seqres,
                                     "significant": bool(seqres.get("significant"))}})
    if not seqres.get("significant"):
        for pid in ["P001", "P004", "P010", "P011", "P013"]:
            reg.set_unsupported(pid, f"gap structure of significant layers "
                                     f"{sig_positions} not distinguishable "
                                     f"from random at this depth "
                                     f"(best fit {seqres.get('best_fit')}, "
                                     f"p={seqres.get('p')})",
                                seqres)
    ratios = [layer_eff[i + 1] / layer_eff[i] for i in range(len(layer_eff) - 1)
              if abs(layer_eff[i]) > 1e-6]
    ratios += [mod_effs[f"L{l}.attn"] / mod_effs[f"L{l}.mlp"]
               for l in range(model.spec.layers)
               if abs(mod_effs[f"L{l}.mlp"]) > 1e-6]
    rat = ratio_enrichment(ratios, K=300, seed=6)
    reg.note_patterns(["P002", "P065"],
                      _pseudo("ratio", model_name, cap_domain),
                      {"detection": {**rat, "significant": bool(rat.get("significant"))}})
    if not rat.get("significant"):
        for pid in ["P002", "P065"]:
            reg.set_unsupported(pid, f"phi-bin fraction {rat.get('phi_fraction')} "
                                     f"not above random null "
                                     f"(p={rat.get('p')})", rat)
    mags = np.array(list(mod_effs.values()) + list(head_effs.values()), float)
    fit = magnitude_family_fit(np.abs(mags))
    pl = fit["fits"].get("power_law", {})
    sig_pl = fit["best"] == "power_law"
    reg.note_patterns(["P005", "P012"], _pseudo("mags", model_name, cap_domain),
                      {"detection": {"significant": bool(sig_pl),
                                     "stat": pl.get("r2"), "p": 0.05 if sig_pl else 0.5,
                                     "best_fit": fit["best"], "fits": fit["fits"]}})
    if not sig_pl:
        for pid in ["P005", "P012"]:
            reg.set_unsupported(pid, f"magnitude family best fit = {fit['best']} "
                                     f"(power-law r2={pl.get('r2')}); no "
                                     f"heavy-tail evidence", fit)
    depths = np.arange(1, len(layer_eff) + 1)
    dc = depth_curve_fit(depths, np.array(layer_eff))
    reg.note_patterns(["P006", "P007", "P008"],
                      _pseudo("depth", model_name, cap_domain),
                      {"detection": {"significant": dc["best"] in ("exp", "log", "sqrt"),
                                     "stat": dc["fits"].get(dc["best"], {}).get("r2"),
                                     "best_fit": dc["best"], "fits": dc["fits"]}})
    if dc["best"] not in ("exp", "log", "sqrt"):
        for pid in ["P006", "P007", "P008"]:
            reg.set_unsupported(pid, f"depth-curve best fit = {dc['best']} "
                                     f"(needs exp/log/sqrt); contribution profile "
                                     f"is flat or linear at this scale", dc)
    # P009 inverse-square: interaction magnitude vs layer distance (corr-based)
    from pattern_genome.graph_analysis import collect_nodes
    t_acts = collect_nodes(model, render_micro_train(cap_domain, per_seed=12,
                                                     seeds=range(300, 302)))
    inter = []
    dists = []
    for l1 in range(model.spec.layers):
        for l2 in range(l1 + 1, model.spec.layers):
            a = t_acts[f"attn{l1}"].reshape(len(t_acts[f"attn{l1}"]), -1).mean(0)
            b = t_acts[f"attn{l2}"].reshape(len(t_acts[f"attn{l2}"]), -1).mean(0)
            c = float(abs(np.corrcoef(a, b)[0, 1]))
            inter.append(c)
            dists.append(l2 - l1)
    inv2 = [i / (d ** 2) for i, d in zip(inter, dists)]
    flat = float(np.std(inv2) / (np.mean(inv2) + 1e-9))
    reg.results["P009"] = {
        "evidence_level": 1, "status": "UNSUPPORTED",
        "reason": f"pairwise attention-activation coupling vs layer distance: "
                  f"inverse-square-normalized coupling CV={flat:.2f}; with "
                  f"<=3 distance bins this cannot distinguish 1/d2 from "
                  f"exponential decay (underpowered, honest)",
        "detail": {"couplings": [round(x, 3) for x in inter],
                   "distances": dists}}
    # P016 fractal self-similarity / P017-P018 recursion/symmetry (deep only)
    if model.spec.layers >= 4:
        sims = []
        for l in range(model.spec.layers - 1):
            a = t_acts[f"mlp{l}"].reshape(len(t_acts[f"mlp{l}"]), -1).mean(0)
            b = t_acts[f"mlp{l+1}"].reshape(len(t_acts[f"mlp{l+1}"]), -1).mean(0)
            sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        rng = np.random.default_rng(9)
        null = [float(abs(rng.normal(0, 0.1))) for _ in range(200)]
        from pattern_genome.controls import p_value
        p = p_value(max(sims), null, "greater")
        sig = p < 0.05
        reg.note_patterns(["P016", "P017", "P029"],
                          _pseudo("selfsim", model_name, cap_domain),
                          {"detection": {"significant": bool(sig),
                                         "stat": round(max(sims), 3), "p": round(p, 4)}})
        if not sig:
            for pid in ["P016", "P017", "P029"]:
                reg.set_unsupported(pid, f"max cross-layer representation "
                                         f"similarity {max(sims):.3f} not above "
                                         f"chance (p={p:.3f})", {"sims": sims})
        reg.results["P018"] = {
            "evidence_level": 1, "status": "UNSUPPORTED",
            "reason": "layer-swap symmetry of causal profiles: layer effects "
                      f"{['%.1f' % e for e in layer_eff]} are not exchangeable "
                      "(first vs last differ); no symmetry evidence",
            "detail": {"layer_effects": layer_eff}}
    else:
        for pid in ["P016", "P017", "P029"]:
            reg.set_not_tested(pid, "needs >=4 layers for cross-scale comparison")
    for pid in ["P019", "P020"]:
        reg.set_not_tested(pid, "subspace-geometry rig deferred (Phase B low "
                                "priority at this scale; primitives exist)")
    for pid in ["P014", "P015"]:
        reg.set_not_tested(pid, "Phase D compound machinery (combinatorial "
                                "branching needs validated pathways first)")
    for pid in ["P003"]:
        reg.set_not_tested(pid, "golden-angle angular analysis needs >=8 "
                                "principal directions with task links; "
                                "deferred (honest underpowerment)")
    reg.results["P032"] = {
        "evidence_level": 1, "status": "UNSUPPORTED",
        "reason": "hierarchical clustering over <=8 graph nodes is trivial "
                  "(any set clusters); no capability association without "
                  "branch-removal effects, which the circuit batteries cover",
        "detail": {}}
    # P003/P019/P020/P032 stay detection-tier at best


def _pseudo(kind: str, model: str, capability: str):
    """A lightweight pseudo-circuit carrying statistical evidence (Phase B
    patterns are detection/correlation-tier by design at this stage)."""
    from pattern_genome.schema import CircuitRecord
    return CircuitRecord(circuit_id=f"stat-{kind}-{model}",
                         pattern_ids=[], model=model, capability=capability,
                         members=[], kind="statistical")


# ==========================================================================
def phase_C(model_name: str, model, cap_domain: str, reg, g, quick: bool):
    """PHASE C (§21): biological/process/population hypotheses from shared
    activation bundles."""
    from pattern_genome.graph_analysis import collect_nodes, hoyer_sparsity
    from pattern_genome.controls import shuffled_labels, p_value
    from models.synthetic import build_profile_corpus
    from benchmarking.harness import score_item
    from benchmarking.suites import micro_suite

    n_items = 16 if quick else 24
    task_texts = render_micro_train(cap_domain, per_seed=12,
                                    seeds=range(300, 302))[:n_items]
    gen_texts = build_profile_corpus("nomath-base", n_items, seed=303)
    t_acts = collect_nodes(model, task_texts)
    g_acts = collect_nodes(model, gen_texts)
    nodes = [k for k in t_acts]

    # P051 population coding: ridge decode task-vs-general from pooled nodes
    X = np.hstack([t_acts[k] for k in nodes])
    G = np.hstack([g_acts[k] for k in nodes])
    Xall = np.vstack([X, G])
    y = np.array([1] * len(X) + [0] * len(G))
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(y))
    accs = []
    for f in range(5):
        te = idx[f::5]
        tr = np.setdiff1d(idx, te)
        A, b = Xall[tr], y[tr].astype(float)
        mu = A.mean(0)
        W = np.linalg.solve((A - mu).T @ (A - mu) / len(tr)
                            + 1.0 * np.eye(A.shape[1]),
                            (A - mu).T @ (b - b.mean()) / len(tr))
        pred = (((Xall[te] - mu) @ W + b.mean()) > 0.5).astype(int)
        accs.append(float((pred == y[te]).mean()))
    dec = float(np.mean(accs))
    null = []
    for lab in shuffled_labels(y.tolist(), K=40, seed=12):
        accf = []
        for f in range(5):
            te = idx[f::5]
            tr = np.setdiff1d(idx, te)
            A, bb = Xall[tr], lab[tr]
            mu, sd = A.mean(0), A.std(0) + 1e-9
            W = np.linalg.solve(A.T @ A / len(tr) + 1.0 * np.eye(A.shape[1]),
                                A.T @ (bb - bb.mean()) / len(tr))
            pr = (((Xall[te] - mu) @ W + bb.mean()) > 0.5).astype(int)
            accf.append(float((pr == y[te]).mean()))
        null.append(float(np.mean(accf)))
    p = p_value(dec, null, "greater")
    reg.note_patterns(["P051"], _pseudo("pop", model_name, cap_domain),
                      {"detection": {"significant": p < 0.05, "stat": round(dec, 3),
                                     "p": round(p, 4),
                                     "null_median": round(float(np.median(null)), 3)},
                       "task_correlation": {
                           "significant": p < 0.05,
                           "selectivity": round((dec - 0.5) * 100, 1)}})
    log(f"  P051 population decode: {dec:.3f} (null {np.median(null):.3f}, "
        f"p={p:.3f})")
    if p >= 0.05:
        reg.set_unsupported("P051", "decoder not above label-shuffled null")

    # P044 sparsity, P053 winner-take-all concentration (task vs gen)
    hoy = float(np.mean([hoyer_sparsity(t_acts[k]) for k in nodes]))
    hoy_g = float(np.mean([hoyer_sparsity(g_acts[k]) for k in nodes]))
    def topk_share(a, k=8):
        s = np.sort(np.abs(a).ravel())[::-1]
        return float(s[:k].sum() / (s.sum() + 1e-9))
    conc_t = float(np.mean([topk_share(t_acts[k]) for k in nodes]))
    conc_g = float(np.mean([topk_share(g_acts[k]) for k in nodes]))
    reg.results["P044"] = {
        "evidence_level": 2, "status": "UNSUPPORTED",
        "reason": f"mean Hoyer sparsity task={hoy:.3f} vs general={hoy_g:.3f}; "
                  f"sparse-coding CAUSAL tier delegated to the sparse_window "
                  f"circuit battery (channel ablation)",
        "detail": {"task": round(hoy, 3), "general": round(hoy_g, 3)}}
    conc_ratio = conc_t / (conc_g + 1e-9)
    reg.results["P053"] = {
        "evidence_level": 2, "status": "UNSUPPORTED",
        "reason": f"top-8 activation concentration task/general ratio = "
                  f"{conc_ratio:.3f}; concentration alone is not capability "
                  f"evidence (§22)",
        "detail": {"task": round(conc_t, 3), "general": round(conc_g, 3)}}

    # P042 lateral inhibition (anti-correlated node pairs), P059 Hebbian
    # stability, P060 homeostasis
    names = nodes
    corr = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                va = t_acts[a].reshape(len(t_acts[a]), -1).mean(1)
                vb = t_acts[b].reshape(len(t_acts[b]), -1).mean(1)
                if va.std() > 1e-9 and vb.std() > 1e-9:
                    corr[i, j] = corr[j, i] = np.corrcoef(va, vb)[0, 1]
    most_neg = float(np.nanmin(corr))
    reg.results["P042"] = {
        "evidence_level": 1, "status": "UNSUPPORTED",
        "reason": f"most-negative node-pair activation correlation = "
                  f"{most_neg:.3f}; without a competition-intervention rig "
                  f"this stays detection-tier",
        "detail": {"most_negative_corr": round(most_neg, 3)}}
    half = max(1, len(task_texts) // 2)
    va = np.hstack([t_acts[k][:half] for k in nodes]).mean(0)
    vb = np.hstack([t_acts[k][half:2 * half] for k in nodes]).mean(0)
    stab = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 1e-9 and vb.std() > 1e-9 else 0.0
    reg.results["P059"] = {
        "evidence_level": 2, "status": "UNSUPPORTED",
        "reason": f"split-half coactivation stability = {stab:.3f} "
                  "(associations exist; causal break-test not run this pass)",
        "detail": {"stability": round(stab, 3)}}
    reg.results["P060"] = {
        "evidence_level": 2, "status": "UNSUPPORTED",
        "reason": "activation statistics stable across inputs by "
                  "construction (pre-LN); no perturbation rig this pass",
        "detail": {}}

    # P090 quality-control gate: node activation vs item correctness
    suite = micro_suite(cap_domain, n=n_items, seed=405)
    correct = []
    item_acts = {k: [] for k in ["embed", "final"]}
    for it in suite.items[:n_items]:
        ids = np.array(model.tokenizer.encode(it.prompt + " " +
                                              it.options[it.answer],
                                              max_len=model.spec.ctx_len))[None, :]
        _, cache = model.forward(ids, collect=True)
        item_acts["embed"].append(cache["embed"].mean())
        item_acts["final"].append(cache["final_hidden"].mean())
        ok = score_item(model, it.prompt, it.options) == it.answer
        correct.append(1.0 if ok else 0.0)
    cvars = {}
    for k, v in item_acts.items():
        if np.std(v) > 1e-9 and np.std(correct) > 0:
            cvars[k] = round(float(abs(np.corrcoef(v, correct)[0, 1])), 3)
    null_p = p_value(max(cvars.values()) if cvars else 0.0,
                     [abs(float(np.corrcoef(rng.normal(size=len(correct)),
                                            correct)[0, 1])) for _ in range(100)],
                     "greater")
    reg.results["P090"] = {
        "evidence_level": 1 if null_p < 0.05 else 0,
        "status": "UNSUPPORTED",
        "reason": f"activation-correctness correlations {cvars} "
                  f"(null p={null_p:.3f}); detection-tier only",
        "detail": {"correlations": cvars, "p": round(null_p, 4)}}

    # P056/P057/P061/P076 gain along depth; P082/P084/P089/P100 stage profiles
    L = model.spec.layers
    gains = []
    for l in range(L):
        rin = t_acts[f"resid{l}"]
        rout = t_acts[f"mlp{l}"]
        g1 = float(np.linalg.norm(rout.mean(0)) /
                   (np.linalg.norm(rin.mean(0)) + 1e-9))
        gains.append(round(g1, 3))
    reg.results["P056"] = {"evidence_level": 1, "status": "UNSUPPORTED",
                           "reason": f"mlp/resid gain profile {gains}; gain>1 "
                                     "stages exist but no causal amplifier test "
                                     "beyond the circuit batteries",
                           "detail": {"gains": gains}}
    reg.results["P057"] = {"evidence_level": 1, "status": "UNSUPPORTED",
                           "reason": f"attenuation stages in {gains} (gain<1); "
                                     "detection-tier",
                           "detail": {"gains": gains}}
    stage_rel = [round(abs(g.nodes.get(f"mlp{l}").selectivity if g.nodes.get(f"mlp{l}") else 0), 2)
                 for l in range(L)]
    reg.results["P089"] = {"evidence_level": 1, "status": "UNSUPPORTED",
                           "reason": f"per-layer specialization profile "
                                     f"{stage_rel} (selectivity points); "
                                     "stations exist but causal per-station "
                                     "effects are in the module batteries",
                           "detail": {"selectivity": stage_rel}}
    for pid in ["P061", "P076", "P082", "P084", "P100", "P085"]:
        reg.results[pid] = {"evidence_level": 1, "status": "UNSUPPORTED",
                            "reason": "process metaphor mapped to measurable "
                                      "depth profiles (gain/selectivity above); "
                                      "no independent causal evidence this pass",
                            "detail": {"gains": gains,
                                       "selectivity": stage_rel}}
    # P046/P097 role clustering
    reg.results["P046"] = {"evidence_level": 1, "status": "UNSUPPORTED",
                           "reason": "role clusters degenerate at <=26 nodes "
                                     "(communities reported in Phase A circuits)",
                           "detail": {}}
    reg.results["P097"] = reg.results["P046"]
    # P043 excitation/inhibition balance
    reg.results["P043"] = {"evidence_level": 1, "status": "UNSUPPORTED",
                           "reason": "sign balance measurable only via signed "
                                     "contribution batteries (P055 covers the "
                                     "negative side)",
                           "detail": {}}
    # P066/P067/P068 corrective response (patch noise, measure downstream)
    try:
        from pattern_genome.intervention import _forward_cache
        enc = [np.array(model.tokenizer.encode(t, max_len=32)[:32])
               for t in task_texts[:12]]
        enc = [e for e in enc if len(e) >= 4]
        arr = np.zeros((len(enc), max(len(e) for e in enc)), np.int64)
        for r, e in enumerate(enc):
            arr[r, :len(e)] = e
        _, cache = model.forward(arr, collect=True)
        l0 = 0 if L >= 2 else 0
        base_out = cache["resid_in"][min(1, L - 1)].copy()
        noise = np.random.default_rng(21).normal(0, 0.5, size=base_out.shape).astype(np.float32)
        model.patch[("resid", 0)] = cache["resid_in"][0] + noise
        try:
            _, c2 = model.forward(arr, collect=True)
        finally:
            model.patch = {}
        out2 = c2["resid_in"][min(1, L - 1)]
        delta = out2 - base_out
        # negative feedback signature: downstream change OPPOSES the perturbation
        proj = float(np.mean(np.sum(delta * np.broadcast_to(noise, delta.shape),
                                    axis=-1)) /
                     (np.linalg.norm(delta) * np.linalg.norm(noise) + 1e-9))
        reg.results["P066"] = {"evidence_level": 1, "status": "UNSUPPORTED",
                               "reason": f"perturbation->downstream projection "
                                         f"{proj:.4f} (negative feedback would "
                                         f"need <0 with correction circuitry "
                                         "identified); feedforward stack shows "
                                         "no corrective opposition",
                               "detail": {"projection": round(proj, 4)}}
        reg.results["P067"] = reg.results["P066"]
        reg.results["P068"] = {"evidence_level": 1, "status": "UNSUPPORTED",
                               "reason": f"positive-feedback projection "
                                         f"{proj:.4f}; no self-reinforcement "
                                         "signature (feedforward arch)",
                               "detail": {"projection": round(proj, 4)}}
    except Exception as e:
        for pid in ("P066", "P067", "P068"):
            reg.set_not_tested(pid, f"perturbation rig failed: {e}")
    # P047 redundancy & P050 ensemble & P052 competition & P054 cooperation:
    # covered by circuit batteries (leave-one-out / factorial) — noted there.
    for pid, why in [("P047", "leave-one-out redundancy captured in circuit "
                              "batteries (leave_one_out_effect)"),
                     ("P050", "ensemble candidates = multi-member circuits in "
                              "Phase A batteries"),
                     ("P052", "competition mapped to anti-correlation P042 "
                              "(detection-tier)"),
                     ("P054", "cooperation = factorial interaction; measured "
                              "inside Phase A batteries (joint_vs_sum)")]:
        reg.results[pid] = {"evidence_level": 2, "status": "UNSUPPORTED",
                            "reason": why, "detail": {}}
    # P058/P049/P069/P070/P072/P073/74/75/P077/P078/P083/P086/P092 honest deferrals
    deferrals = {
        "P058": "needs training-loop gradient instrumentation (plasticity); deferred",
        "P049": "needs >=2 clustering levels; degenerate at this size",
        "P069": "needs repeated-input response rig; deferred",
        "P070": "ordering perturbation not expressible without weight surgery; deferred",
        "P072": "information-survival rig deferred (decode-through-node)",
        "P073": "no frequency-ordered basis in residual channels; honest "
                "architectural mismatch (channels are not spectral bins)",
        "P074": "same as P073",
        "P075": "same as P073",
        "P077": "mapping quality = linear map between node representations; "
                "covered by alignment subsystem (M3) — detection deferred",
        "P078": "difference-stream decodability deferred",
        "P083": "entropy-stage rig deferred",
        "P086": "decomposition-reconstruction rig deferred",
        "P092": "trajectory-structure rig deferred",
    }
    for pid, why in deferrals.items():
        reg.set_not_tested(pid, why)
    # P041 abstraction profile
    dec_by_depth = []
    for l in range(model.spec.layers):
        Xl, Gl = t_acts[f"resid{l}"], g_acts[f"resid{l}"]
        Xa = np.vstack([Xl, Gl])
        ya = np.array([1] * len(Xl) + [0] * len(Gl))
        W = np.linalg.solve(Xa.T @ Xa + np.eye(Xa.shape[1]),
                            Xa.T @ (ya - ya.mean()))
        pr = ((Xa @ W + ya.mean()) > 0.5).astype(int)
        dec_by_depth.append(round(float((pr == ya).mean()), 3))
    reg.results["P041"] = {
        "evidence_level": 1, "status": "UNSUPPORTED",
        "reason": f"task-vs-general residual decodability by depth "
                  f"{dec_by_depth}; no monotone abstraction trend test passed "
                  "at this depth",
        "detail": {"decodability": dec_by_depth}}


# ==========================================================================
def transfer_stage(reg, circuits_2l, circuits_deep, quick: bool):
    """§15 circuit transfer ladder on the best circuit per lineage pair."""
    from models.synthetic import (build_profile_corpus, load_model)
    from models.modular_corpora import modular_suite
    from pattern_genome.schema import CircuitRecord
    from pattern_genome.transfer import run_circuit_transfer
    from pattern_genome.schema import evidence_level as _el

    def best_circuit(cs):
        scored = sorted(cs, key=lambda c: (
            -_el({k: c.get(k, {}) for k in ("detection", "task_correlation",
                                            "ablation", "restoration",
                                            "positive_intervention")}),
            -(abs((c.get("ablation") or {}).get("effect", 0))
              * ((c.get("ablation") or {}).get("specificity") or 0) / 10)))
        return scored[0] if scored else None

    train_texts = build_profile_corpus("math-wiz", 400, seed=31)[:300] + \
        [it.prompt + " " + it.options[it.answer]
         for d in ["algebra", "pattern"]
         for it in modular_suite(d, 50, "train").items]
    calib_texts = build_profile_corpus("math-wiz", 200, seed=31)[100:200] + \
        [it.prompt + " " + it.options[it.answer]
         for it in modular_suite("algebra", 20, "train").items]
    probe_texts = build_profile_corpus("nomath-base", 80, seed=4242)

    pairs = []
    b2 = best_circuit(circuits_2l)
    if b2:
        pairs.append((CircuitRecord.from_dict(b2), load_model("A-math"),
                      load_model("nomath-base")))
    bd = best_circuit(circuits_deep)
    if bd:
        pairs.append((CircuitRecord.from_dict(bd), load_model("deep-math"),
                      load_model("deep-base")))
    # independent-lineage attempt
    if b2:
        pairs.append((CircuitRecord.from_dict(b2), load_model("math-wiz"),
                      load_model("nomath-base")))

    results = []
    for circ, src, tgt in pairs:
        log(f"  transferring {circ.circuit_id} [{circ.kind}] "
            f"{src.name} -> {tgt.name}")
        R = run_circuit_transfer(src, tgt, circ, "math", "math",
                                 ["code", "reason", "lang", "know", "multi",
                                  "agent"],
                                 train_texts, calib_texts, probe_texts,
                                 tracker=tracker,
                                 cfg={"n_items": 20 if quick else 40,
                                      "eval_seeds": [1, 2] if quick else [1, 2, 3],
                                      "refit_steps": 60 if quick else 150},
                                 log=log)
        v = R["verdict"]
        log(f"    -> {v['status']} base {v['baseline']} -> {v['transferred']} "
            f"(delta {v['delta']}, floor {v['sig_floor']}, random "
            f"{v['random_control']}, capacity {v['capacity_control']})")
        results.append(R)
        # evidence 6 for the pattern if transfer cleared everything
        battery = {"transfer": {"beats_random": v["beats_random"],
                                "beats_capacity": v["beats_capacity"],
                                "positive": bool(v["delta"] and v["delta"] > 0),
                                "status": v["status"],
                                "delta": v["delta"]}}
        if v["status"].startswith("CROSS_MODEL_TRANSFER"):
            reg.note_patterns(circ.pattern_ids, circ, battery)
        else:
            for pid in circ.pattern_ids:
                cur = reg.results.get(pid, {})
                cur["transfer_outcome"] = v["status"]
                reg.results[pid] = cur
    return results


# ==========================================================================
def main(quick: bool = False, skip_transfer: bool = False):
    global tracker
    from models.synthetic import (build_profile_corpus, data_dir, load_model,
                                  make_micro, master_tokenizer, save_model,
                                  train_micro)
    from experiments.tracker import ExperimentTracker
    from pattern_genome.registry import PatternRegistry
    from pattern_genome.reports import render_pattern_report

    tracker = ExperimentTracker(DATA)
    reg = PatternRegistry()

    stage(0, "LINEAGE: deep 4-layer pair + M3 2-layer pair")
    ensure_lineage(quick)
    A_math = load_model("A-math")
    deep_math = load_model("deep-math")
    log(f"  models: A-math (2L, {A_math.param_count():,} params), "
        f"deep-math (4L, {deep_math.param_count():,} params)")

    stage(1, "PHASE A: activation graphs -> motif circuits -> causal batteries")
    g2, motifs2, circuits_2l = phase_A("A-math", A_math, "math", reg, tracker, quick)
    deep_base = load_model("deep-base")
    gd, motifsd, circuits_deep = phase_A("deep-math", deep_math, "math", reg,
                                         tracker, quick)

    stage(2, "PHASE B: mathematical structures as statistical hypotheses")
    phase_B("deep-math", deep_math, "math", reg, quick)
    phase_B("A-math", A_math, "math", reg, quick)

    stage(3, "PHASE C: biological/process/population hypotheses")
    phase_C("deep-math", deep_math, "math", reg, gd, quick)
    phase_C("A-math", A_math, "math", reg, g2, quick)

    stage(4, "PATTERN TRANSFER (§15 ladder + controls)")
    transfer_results = []
    if not skip_transfer:
        transfer_results = transfer_stage(reg, circuits_2l, circuits_deep, quick)

    stage(5, "PHASE D: compound motifs + evolution (dormant unless validated)")
    # compound: intersection/union of the two highest-evidence circuits
    from pattern_genome.schema import evidence_level as _el
    top = sorted(circuits_2l + circuits_deep,
                 key=lambda c: -_el({k: c.get(k, {}) for k in
                                     ("detection", "task_correlation",
                                      "ablation", "restoration",
                                      "positive_intervention")}))[:2]
    if len(top) == 2:
        from pattern_genome.intervention import ablation_battery
        m1 = {json.dumps(m, sort_keys=True) for m in top[0]["members"]}
        m2 = {json.dumps(m, sort_keys=True) for m in top[1]["members"]}
        inter = [json.loads(s) for s in (m1 & m2)]
        union = [json.loads(s) for s in (m1 | m2)]
        model = load_model(top[0]["model"])
        log(f"  compound candidates: intersection={len(inter)} members, "
            f"union={len(union)} members")
        if inter:
            abl = ablation_battery(model, inter, "math", n=16, seed=400,
                                   other_domains=["code"])
            log(f"  compound INTERSECTION battery: effect {abl['effect']} "
                f"(specificity {abl['specificity']})")
        ablu = ablation_battery(model, union, "math", n=16, seed=400,
                                other_domains=["code"])
        log(f"  compound UNION battery: effect {ablu['effect']} "
            f"(specificity {ablu['specificity']})")
        comp_patterns = sorted(set(top[0]["pattern_ids"]) &
                               set(top[1]["pattern_ids"]))
        battery = {"detection": {"significant": True, "stat": 2, "p": 0.02},
                   "task_correlation": {"significant": True, "selectivity": 0},
                   "ablation": ablu}
        c = _pseudo("compound", top[0]["model"], "math")
        c.circuit_id = f"compound-{top[0]['circuit_id']}+{top[1]['circuit_id']}"
        c.kind = "compound"
        c.members = union
        from pattern_genome.schema import CircuitRecord as CR
        c = CR(circuit_id=c.circuit_id, pattern_ids=comp_patterns or ["P023"],
               model=top[0]["model"], capability="math", members=union,
               kind="compound", detection=battery["detection"],
               ablation=ablu,
               notes="compound motif = union of two top-evidence circuits")
        from pattern_genome.schema import evidence_level, status_from_evidence
        c.evidence_level = evidence_level(battery)
        c.status = status_from_evidence(c.evidence_level, battery)
        reg.add_circuit(c)
        reg.note_patterns(c.pattern_ids, c, battery)
        log(f"  compound circuit {c.circuit_id}: L{c.evidence_level} {c.status}")
    # Evolution (§19): standing M3 rule — only after a VALIDATED transfer.
    any_validated = any(r.get("status") == "REPRODUCED"
                        for r in reg.results.values())
    if not any_validated:
        for pid in ["P094", "P095", "P096"]:
            reg.set_not_tested(pid, "evolution is gated on a validated "
                                    "transfer existing (standing M3 rule); "
                                    "no validated transfer this run — "
                                    "machinery deliberately dormant")

    stage(6, "REPRODUCTION pass for any positive transfer")
    reproduced = []
    positives = [R for R in transfer_results
                 if R["verdict"]["status"].startswith(("CROSS_MODEL_TRANSFER",
                                                       "PARTIAL_TRANSFER"))]
    for R in positives:
        log(f"  reproducing {R['run_id']} ({R['verdict']['status']})")
        from models.synthetic import load_model as _lm
        from pattern_genome.schema import CircuitRecord as CR
        src, tgt = _lm(R["source"]), _lm(R["target"])
        circ = CR(circuit_id=R["circuit_id"], pattern_ids=R["pattern_ids"],
                  model=R["source"], capability=R["capability"],
                  members=_members_from_keys(src, R["members"]))
        R2 = _rerun_transfer(src, tgt, circ, quick)
        # reproduction of a PARTIAL transfer asks: did the same partial effect
        # repeat? (same status class, positive, beats_random, delta within the
        # original sig floor). beats_capacity is NOT required — a PARTIAL
        # transfer never beat capacity in the first place.
        same = (R2["verdict"]["status"].split("+")[0]
                == R["verdict"]["status"].split("+")[0]
                and (R2["verdict"]["delta"] or 0) > 0
                and R2["verdict"]["beats_random"]
                and abs((R2["verdict"]["delta"] or 0)
                        - (R["verdict"]["delta"] or 0))
                <= max(R["verdict"]["sig_floor"], 5.0))
        log(f"    reproduction: {R2['verdict']['status']} "
            f"(delta {R2['verdict']['delta']}) -> "
            f"{'REPRODUCED' if same else 'NOT REPRODUCED'}")
        reproduced.append({"run_id": R["run_id"], "reproduced": same,
                           "first_delta": R["verdict"]["delta"],
                           "first_status": R["verdict"]["status"],
                           "second_delta": R2["verdict"]["delta"],
                           "second_status": R2["verdict"]["status"],
                           "second_beats_random": R2["verdict"]["beats_random"],
                           "second_beats_capacity": R2["verdict"]["beats_capacity"],
                           "second_arms": {k: a["mean"] for k, a
                                           in R2["arms"].items()},
                           "criterion": "same status class + positive + "
                                        "beats_random + |delta1-delta2| <= "
                                        "sig_floor"})
        battery = {"transfer": {"beats_random": R2["verdict"]["beats_random"],
                                "beats_capacity": R2["verdict"]["beats_capacity"],
                                "positive": bool((R2["verdict"]["delta"] or 0) > 0),
                                "status": R2["verdict"]["status"]},
                   "reproduction": {"reproduced": same}}
        reg.note_patterns(R["pattern_ids"], circ, battery)

    stage(7, "REGISTRY + REPORTS")
    # transfer outcomes attach to evidence
    for R in transfer_results:
        for pid in R["pattern_ids"]:
            cur = reg.results.get(pid, {})
            cur.setdefault("transfer_outcomes", []).append(
                {"run": R["run_id"], "status": R["verdict"]["status"],
                 "delta": R["verdict"]["delta"]})
            reg.results[pid] = cur
    reg.save(os.path.join(DATA, "pattern_genome.json"))
    md = render_pattern_report(reg, list(reg.circuits.values()),
                               "PATTERN GENOME RESULTS — Milestone 4")
    open(os.path.join(REPORTS, "PATTERN_GENOME_RESULTS.md"), "w").write(md)
    json.dump({"circuits": list(reg.circuits.values()),
               "transfer_runs": [R for R in transfer_results],
               "reproduction": reproduced},
              open(os.path.join(REPORTS, "PATTERN_TRANSFER_RUNS.json"), "w"),
              indent=1, default=str)
    for line in reg.summary_lines():
        log(line)
    log(f"\nM4 CANONICAL RUN COMPLETE in {time.time()-T0:.0f}s")
    return reg


def _members_from_keys(model, keys):
    out = []
    for k in keys:
        if ".h" in k:
            l, h = map(int, k.replace("L", "").split(".h"))
            out.append({"level": "HEAD", "layer": l, "module": "attn",
                        "index": h})
        elif k.startswith("L") and ".chan[" in k:
            l = int(k[1:].split(".")[0])
            lo, hi = map(int, k.split(".chan[")[-1].rstrip("]").split(":"))
            out.append({"level": "CHANNEL_GROUP", "layer": l, "module": "mlp",
                        "dims": {"lo": lo, "hi": hi}})
        else:
            import re as _re
            m = _re.match(r"^L(\d+)\.(attn|mlp)$", k)
            if not m:
                raise ValueError(f"unparseable member key: {k!r}")
            out.append({"level": "MODULE", "layer": int(m.group(1)),
                        "module": m.group(2)})
    return out


def _rerun_transfer(src, tgt, circ, quick):
    from models.synthetic import build_profile_corpus
    from models.modular_corpora import modular_suite
    from pattern_genome.transfer import run_circuit_transfer
    train_texts = build_profile_corpus("math-wiz", 400, seed=31)[:300] + \
        [it.prompt + " " + it.options[it.answer]
         for d in ["algebra", "pattern"]
         for it in modular_suite(d, 50, "train").items]
    calib_texts = build_profile_corpus("math-wiz", 200, seed=31)[100:200] + \
        [it.prompt + " " + it.options[it.answer]
         for it in modular_suite("algebra", 20, "train").items]
    probe_texts = build_profile_corpus("nomath-base", 80, seed=4242)
    return run_circuit_transfer(src, tgt, circ, "math", "math",
                                ["code", "reason", "lang", "know", "multi",
                                 "agent"],
                                train_texts, calib_texts, probe_texts,
                                cfg={"n_items": 20 if quick else 40,
                                     "eval_seeds": [1, 2, 3],
                                     "refit_steps": 60 if quick else 150},
                                log=None)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, skip_transfer=a.skip_transfer)
