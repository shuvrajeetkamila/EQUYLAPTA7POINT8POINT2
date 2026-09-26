"""Causal batteries for functional structures (Milestone 4 spec §3, §13, §14).

Pipeline steps after detection:
  ABLATE (levels as applicable) -> CONTROL (randomize, in controls.py),
  RESTORATION (activation patching = mediation proof, graded per member),
  INTERVENTION (amplify the structure), MINIMAL UNIT SEARCH (§14).

Search NEVER touches held-out data: subset search runs on the calib split
(fixed seed 400 draws); held-out is used once for final verification.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmarking.harness import eval_suite
from benchmarking.suites import micro_suite

from .schema import MemberRef, ABLATION_LEVELS


# --------------------------------------------------------------------------
# mask plumbing (MemberRef -> forward masks, mirroring the M2/M3 ablator)
# --------------------------------------------------------------------------
def _apply_members(model, members: List[dict], on: bool,
                   partial: Optional[List[str]] = None):
    """Set (on=True) or clear (on=False) forward masks for members. If
    `partial` (list of member keys) is given, only those are applied."""
    if not on:
        model.skip_layers, model.skip_modules = [], {}
        model.head_mask = None
        model.mlp_channel_mask = None
        return
    skip_modules: Dict[int, List[str]] = {}
    head_mask = None
    chan_masks: Dict[int, np.ndarray] = {}
    for m in members:
        ref = MemberRef.from_dict(m)
        if partial is not None and ref.key() not in partial:
            continue
        if ref.level == "MODULE":
            skip_modules.setdefault(ref.layer, []).append(ref.module or "attn")
        elif ref.level == "HEAD":
            if head_mask is None:
                head_mask = np.ones((model.spec.layers, model.spec.heads), np.float32)
            head_mask[ref.layer, ref.index] = 0.0
        elif ref.level == "CHANNEL_GROUP":
            if ref.layer not in chan_masks:
                chan_masks[ref.layer] = np.ones(model.spec.mlp_hidden, np.float32)
            chan_masks[ref.layer][ref.dims["lo"]:ref.dims["hi"]] = 0.0
    model.skip_layers = []
    model.skip_modules = skip_modules
    model.head_mask = head_mask
    model.mlp_channel_mask = chan_masks or None


def eval_with_members(model, members: List[dict], domain: str, n: int, seed: int,
                      partial: Optional[List[str]] = None) -> float:
    """Accuracy (%) with the given members ablated (forward-time masks)."""
    _apply_members(model, members, True, partial)
    try:
        r = eval_suite(model, micro_suite(domain, n=n, seed=seed), max_items=n)
        return float(r["accuracy"] * 100)
    finally:
        _apply_members(model, members, False)


# --------------------------------------------------------------------------
# ablation battery (spec §13 levels mapped onto member circuits)
# --------------------------------------------------------------------------
def ablation_battery(model, members: List[dict], domain: str,
                     n: int = 32, seed: int = 400, floor: float = 5.0,
                     other_domains: Optional[List[str]] = None) -> Dict:
    """Level 1 (single node: each member alone), level 5 (subgraph: leave-one-
    out), level 7 (entire circuit). Records full effect, per-member effects,
    joint-vs-sum, and capability specificity (target drop - mean other drop)."""
    base = eval_with_members(model, [], domain, n, seed)
    full = eval_with_members(model, members, domain, n, seed)
    keys = [MemberRef.from_dict(m).key() for m in members]
    singles = {}
    for k in keys:
        a = eval_with_members(model, members, domain, n, seed, partial=[k])
        singles[k] = base - a
    full_effect = base - full
    loo = {}
    for k in keys:
        keep = [x for x in keys if x != k]
        loo[k] = base - eval_with_members(model, members, domain, n, seed,
                                          partial=keep)
    other = {}
    for od in (other_domains or []):
        ob = eval_with_members(model, [], od, max(16, n // 2), seed + 1)
        of = eval_with_members(model, members, od, max(16, n // 2), seed + 1)
        other[od] = round(ob - of, 2)
    spec = (full_effect - float(np.mean(list(other.values()))) if other else None)
    return {
        "level_names": {str(k): v for k, v in ABLATION_LEVELS.items()},
        "baseline": round(base, 2), "full_circuit": round(full, 2),
        "effect": round(full_effect, 2),
        "effect_direction": "improves" if full_effect < 0 else "degrades",
        "significant": bool(abs(full_effect) >= floor),
        "floor": floor,
        "per_member_effect": {k: round(v, 2) for k, v in singles.items()},
        "max_single": round(max(singles.values()) if singles else 0.0, 2),
        "leave_one_out_effect": {k: round(v, 2) for k, v in loo.items()},
        "joint_vs_sum": round(full_effect - sum(singles.values()), 2),
        "other_domain_effects": other,
        "specificity": round(float(spec), 2) if spec is not None else None,
        "effect_specific": bool(spec is not None and spec >= 5.0),
    }


# --------------------------------------------------------------------------
# restoration battery (spec §3 RESTORATION; §17 level 4)
# --------------------------------------------------------------------------
def _encode_pair(model, prompt: str, option: str, cap: int) -> np.ndarray:
    p = list(model.tokenizer.encode(prompt, max_len=cap))
    o = list(model.tokenizer.encode(" " + option, max_len=8))
    return np.array((p + o)[-cap:], dtype=np.int64)


def restoration_battery(model, members: List[dict], domain: str,
                        n: int = 24, seed: int = 400) -> Dict:
    """Mediation test at capability level. For each calib item:
      1. clean forward with collect -> cache member module outputs;
      2. ablated forward (masks on) -> accuracy baseline for recovery;
      3. patched forward (masks on, patch[(mod_out,l)] = clean act) -> does
         re-supplying the removed activations recover the capability?
    Also the GRADED version: restore ONE member while the rest stay ablated
    (per-member mediation under damage). Recovery % = (patched-ablated)/
    (clean-ablated)*100. Patching exactly what was removed is expected to
    recover ~100% (mediation proof); partial restorations are the informative
    numbers."""
    from benchmarking.suites import SuiteItem
    mods = sorted({(MemberRef.from_dict(m).layer, MemberRef.from_dict(m).module)
                   for m in members if MemberRef.from_dict(m).level == "MODULE"})
    if not mods:
        return {"recovered_pct": None,
                "reason": "no MODULE members to patch (heads/chan not collection points)"}
    suite = micro_suite(domain, n=n, seed=seed)
    items = suite.items[:n]

    def score(forward_fn):
        hit = 0
        for it in items:
            best, best_lp = None, -1e18
            for oi, opt in enumerate(it.options):
                ids = _encode_pair(model, it.prompt, opt, model.spec.ctx_len)[None, :]
                lp = forward_fn(ids, it, oi)
                if lp > best_lp:
                    best_lp, best = lp, oi
            hit += int(best == it.answer)
        return 100.0 * hit / max(1, len(items))

    def plain(ids, it, oi):
        logits, _ = model.forward(ids)
        return _opt_logprob(model, logits, ids, it, oi)

    def patched(ids, it, oi, restore_keys=None):
        _, cache = model.forward(ids, collect=True)
        patch_dict = {}
        for (l, mod) in mods:
            key = MemberRef("MODULE", layer=l, module=mod).key()
            if restore_keys is None or key in restore_keys:
                patch_dict[(f"{mod}_out", l)] = cache[f"{mod}_out"][l]
        # ablate all members; patch the selected slots with clean activations
        _apply_members(model, members, True)
        model.patch = {k: np.asarray(v, np.float32) for k, v in patch_dict.items()}
        try:
            logits, _ = model.forward(ids)
            return _opt_logprob(model, logits, ids, it, oi)
        finally:
            model.patch = {}
            _apply_members(model, members, False)

    clean_acc = score(plain)
    ablated_acc = score(_ablated_scorer(model, members))
    full_restore = score(lambda ids, it, oi: patched(ids, it, oi, None))
    graded = {}
    for k in [MemberRef.from_dict(m).key() for m in members
              if MemberRef.from_dict(m).level == "MODULE"]:
        # restore ONLY member k's slot; other module members stay ablated
        keep = [k]
        # temporarily restrict ablation+patching to module members
        mod_members = [m for m in members if MemberRef.from_dict(m).level == "MODULE"]
        acc = score(lambda ids, it, oi: _graded_patch(model, mod_members, keep,
                                                      ids, it, oi))
        graded[k] = round(acc, 2)
    denom = clean_acc - ablated_acc
    recovered = (100.0 * (full_restore - ablated_acc) / denom) if denom > 1e-6 else None
    return {"clean_acc": round(clean_acc, 2), "ablated_acc": round(ablated_acc, 2),
            "full_restore_acc": round(full_restore, 2),
            "recovered_pct": round(float(recovered), 1) if recovered is not None else None,
            "graded_restore_acc": graded,
            "note": "patched slot = clean activations of the removed module output; "
                    "mediation proof + graded per-member mediation"}


def _ablated_scorer(model, members: List[dict]):
    def fn(ids, it, oi):
        _apply_members(model, members, True)
        try:
            logits, _ = model.forward(ids)
            return _opt_logprob(model, logits, ids, it, oi)
        finally:
            _apply_members(model, members, False)
    return fn


def _graded_patch(model, mod_members, keep_keys, ids, it, oi):
    _, cache = model.forward(ids, collect=True)
    patch_dict = {}
    for m in mod_members:
        ref = MemberRef.from_dict(m)
        if ref.key() in keep_keys:
            patch_dict[(f"{ref.module}_out", ref.layer)] = cache[f"{ref.module}_out"][ref.layer]
    _apply_members(model, mod_members, True)
    model.patch = {k: np.asarray(v, np.float32) for k, v in patch_dict.items()}
    try:
        logits, _ = model.forward(ids)
        return _opt_logprob(model, logits, ids, it, oi)
    finally:
        model.patch = {}
        _apply_members(model, mod_members, False)


def _opt_logprob(model, logits: np.ndarray, ids: np.ndarray, it, oi: int) -> float:
    """Mean teacher-forced logprob of the option tokens (mirrors
    ModelHandle.logprob_option)."""
    prompt_len = len(model.tokenizer.encode(it.prompt, max_len=model.spec.ctx_len))
    n_opt = ids.shape[1] - prompt_len
    if n_opt <= 0:
        n_opt = 1
    lp = 0.0
    for i in range(n_opt):
        pos = ids.shape[1] - n_opt + i - 1
        row = logits[0, pos]
        row = row - row.max()
        logz = np.log(np.exp(row).sum())
        tok = int(ids[0, pos + 1])
        lp += float(row[tok] - logz)
    return lp / n_opt


# --------------------------------------------------------------------------
# positive intervention (§17 level 5): amplify the structure
# --------------------------------------------------------------------------
def positive_intervention_battery(model, members: List[dict], domain: str,
                                  n: int = 32, seed: int = 401,
                                  gammas: Tuple[float, ...] = (1.5, 2.0)) -> Dict:
    """Scale member OUTPUT tensors by gamma (amplify) and measure capability
    change. Positive evidence = significant change beyond the floor."""
    from component_transfer.apply import snapshot, restore_snapshot
    floor = 5.0
    base = eval_with_members(model, members, domain, n, seed)
    out = {"baseline": round(base, 2), "arms": {}, "significant": False,
           "max_abs_change": 0.0}
    snap = snapshot(model)
    try:
        locs = _member_out_tensors(model, members)
        for gamma in gammas:
            restore_snapshot(model, snap)
            T = model.state_dict()
            for tname, sl in locs:
                T[tname][sl] = (T[tname][sl].astype(np.float64) * gamma).astype(np.float32)
            model.load_state_dict(T)
            acc = eval_with_members(model, [], domain, n, seed)
            out["arms"][f"x{gamma}"] = round(acc, 2)
        restore_snapshot(model, snap)
        vals = [abs(out["arms"][f"x{g}"] - base) for g in gammas]
        out["max_abs_change"] = round(max(vals) if vals else 0.0, 2)
        out["significant"] = bool(vals and max(vals) >= floor)
    finally:
        restore_snapshot(model, snap)
    return out


def _member_out_tensors(model, members: List[dict]) -> List[Tuple[str, tuple]]:
    locs = []
    for m in members:
        ref = MemberRef.from_dict(m)
        if ref.level == "MODULE":
            if ref.module == "mlp":
                locs.append((f"L{ref.layer}.mlp.2.W", np.s_[:, :]))
            else:
                locs.append((f"L{ref.layer}.attn.o.W", np.s_[:, :]))
        elif ref.level == "HEAD":
            d = model.spec.hidden // model.spec.heads
            h0 = ref.index * d
            locs.append((f"L{ref.layer}.attn.o.W", np.s_[h0:h0 + d, :]))
        elif ref.level == "CHANNEL_GROUP":
            locs.append((f"L{ref.layer}.mlp.2.W",
                         np.s_[ref.dims["lo"]:ref.dims["hi"], :]))
    return locs


# --------------------------------------------------------------------------
# §14 MINIMAL FUNCTIONAL UNIT SEARCH (calib split only)
# --------------------------------------------------------------------------
def minimal_unit_search(model, members: List[dict], domain: str,
                        n: int = 32, seed: int = 400, rho: float = 0.8,
                        full_effect: Optional[float] = None,
                        log=None) -> Dict:
    """Backward elimination: smallest member subset S with
    effect(S) >= rho * effect(full). Search on calib seed only; verification
    re-measures the retained subset (held-out verification happens later,
    once, in the runner)."""
    keys = [MemberRef.from_dict(m).key() for m in members]
    base = eval_with_members(model, [], domain, n, seed)
    if full_effect is None:
        full_effect = base - eval_with_members(model, members, domain, n, seed)
    target = rho * full_effect
    current = list(keys)
    steps = [{"kept": list(current), "effect": round(full_effect, 2)}]
    while len(current) > 1:
        best_eff, best_k = None, None
        for k in current:
            trial = [x for x in current if x != k]
            eff = base - eval_with_members(model, members, domain, n, seed,
                                           partial=trial)
            if best_eff is None or eff > best_eff:
                best_eff, best_k = eff, k
        if best_eff >= target:
            current = [x for x in current if x != best_k]
            steps.append({"kept": list(current), "effect": round(best_eff, 2),
                          "removed": best_k})
            if log:
                log(f"    mu-search: dropped {best_k} -> kept {current} "
                    f"(effect {best_eff:.1f} >= target {target:.1f})")
        else:
            if log:
                log(f"    mu-search: cannot drop {best_k} (effect {best_eff:.1f} "
                    f"< target {target:.1f}) — stop")
            break
    final_effect = base - eval_with_members(model, members, domain, n, seed,
                                            partial=current)
    return {"minimal_unit": current, "rho": rho,
            "full_circuit_effect": round(full_effect, 2),
            "minimal_effect": round(final_effect, 2),
            "retention_pct": round(100.0 * final_effect / (full_effect + 1e-9), 1),
            "target_effect": round(target, 2),
            "n_members_full": len(keys), "n_members_minimal": len(current),
            "search_steps": steps,
            "verified": bool(final_effect >= target)}
