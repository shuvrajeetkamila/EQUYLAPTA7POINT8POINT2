"""Report renderers for the Pattern Genome (Milestone 4 spec: REPORT TEXT FILE)."""
from __future__ import annotations

from typing import Dict, List, Optional

from .schema import EVIDENCE_LEVELS
from .registry import PatternRegistry


def render_circuit_line(c: dict) -> str:
    return (f"{c.get('circuit_id','?')} [{c.get('kind','?')}] "
            f"model={c.get('model','?')} capability={c.get('capability','?')} "
            f"members={c.get('members', c.get('member_keys', []))} "
            f"status={c.get('status','?')} L{c.get('evidence_level',0)}")


def render_battery(battery: Dict) -> str:
    L = []
    det = battery.get("detection") or {}
    if det:
        L.append(f"  detection: stat={det.get('stat')} p={det.get('p')} "
                 f"null_median={det.get('null_median')} "
                 f"-> {'SIGNIFICANT' if det.get('significant') else 'not significant'}")
    cor = battery.get("task_correlation") or {}
    if cor:
        L.append(f"  task-correlation: selectivity={cor.get('selectivity')} "
                 f"-> {'SIGNIFICANT' if cor.get('significant') else 'not significant'}")
    abl = battery.get("ablation")
    if abl:
        L.append(f"  ablation: base {abl.get('baseline')} -> full-circuit "
                 f"{abl.get('full_circuit')} (effect {abl.get('effect')}, "
                 f"floor {abl.get('floor')}, "
                 f"{'SIGNIFICANT' if abl.get('significant') else 'n.s.'}; "
                 f"direction {abl.get('effect_direction')})")
        L.append(f"    per-member: {abl.get('per_member_effect')}")
        L.append(f"    leave-one-out: {abl.get('leave_one_out_effect')}")
        L.append(f"    joint_vs_sum={abl.get('joint_vs_sum')} "
                 f"other-domains={abl.get('other_domain_effects')} "
                 f"specificity={abl.get('specificity')}")
    res = battery.get("restoration")
    if res:
        L.append(f"  restoration: clean {res.get('clean_acc')} / ablated "
                 f"{res.get('ablated_acc')} / restored {res.get('full_restore_acc')} "
                 f"-> recovered {res.get('recovered_pct')}% "
                 f"graded={res.get('graded_restore_acc')}")
    pos = battery.get("positive_intervention")
    if pos:
        L.append(f"  positive-intervention (amplify): {pos.get('arms')} "
                 f"max|change|={pos.get('max_abs_change')} "
                 f"{'SIGNIFICANT' if pos.get('significant') else 'n.s.'}")
    tra = battery.get("transfer")
    if tra:
        v = tra.get("verdict") or tra
        L.append(f"  transfer: {v.get('status')} base {v.get('baseline')} -> "
                 f"{v.get('transferred')} (delta {v.get('delta')}, floor "
                 f"{v.get('sig_floor')}, random ctl {v.get('random_control')}, "
                 f"capacity ctl {v.get('capacity_control')})")
    rep = battery.get("reproduction")
    if rep:
        L.append(f"  reproduction: {rep}")
    return "\n".join(L)


def render_pattern_report(reg: PatternRegistry, circuits: List[dict],
                          title: str = "PATTERN GENOME REPORT") -> str:
    L = [title, "=" * 74, ""]
    from collections import Counter
    tested = [r for r in reg.records if r.pattern_id in reg.results]
    st = reg.status_counts()
    L.append(f"Patterns registered: {len(reg.records)} (hypotheses, never facts)")
    L.append(f"Patterns tested: {len(tested)}   statuses: {st}")
    lv_hist = Counter(reg.results[p].get("evidence_level", 0) for p in
                      [r.pattern_id for r in tested])
    L.append("Evidence ladder histogram (tested patterns): " +
             ", ".join(f"L{k} {EVIDENCE_LEVELS[k]}: {v}"
                       for k, v in sorted(lv_hist.items())))
    L.append("")
    L.append("-- Circuits --")
    for c in circuits:
        L.append(render_circuit_line(c))
        if c.get("minimal_unit"):
            mu = c["minimal_unit"]
            L.append(f"    MINIMAL UNIT: {mu.get('minimal_unit')} "
                     f"retains {mu.get('retention_pct')}% of full-circuit effect "
                     f"({mu.get('minimal_effect')} of {mu.get('full_circuit_effect')}) "
                     f"verified={mu.get('verified')}")
    L.append("")
    L.append("-- Pattern evidence (tested only) --")
    for r in reg.records:
        res = reg.results.get(r.pattern_id)
        if not res:
            continue
        L.append(f"{r.pattern_id} {r.name} [{r.phase}] "
                 f"-> L{res.get('evidence_level', 0)} "
                 f"{res.get('status', 'NOT_TESTED')}")
        if res.get("reason"):
            L.append(f"    reason: {res['reason']}")
        bs = res.get("battery_summary") or {}
        for k, v in bs.items():
            L.append(f"    {k}: {v}")
    return "\n".join(L)
