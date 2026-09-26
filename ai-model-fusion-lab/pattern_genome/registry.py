"""Pattern registry runtime state (Milestone 4 spec §2).

Definitions live in patterns/definitions.py (immutable hypotheses); this
module tracks EVIDENCE: results per pattern, circuits nominated, statuses,
with JSON persistence to the data directory.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from .schema import (PatternRecord, CircuitRecord, EVIDENCE_LEVELS,
                     evidence_level, status_from_evidence)
from .patterns.definitions import all_records


class PatternRegistry:
    def __init__(self, records: Optional[List[PatternRecord]] = None):
        self.records: List[PatternRecord] = records or all_records()
        self.by_id: Dict[str, PatternRecord] = {r.pattern_id: r for r in self.records}
        self.results: Dict[str, dict] = {}
        self.circuits: Dict[str, dict] = {}

    # ---------------- persistence ----------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n_patterns": len(self.records),
                   "results": self.results,
                   "circuits": self.circuits},
                  open(path, "w"), indent=1, default=str)

    @classmethod
    def load(cls, path: str) -> "PatternRegistry":
        reg = cls()
        if os.path.exists(path):
            d = json.load(open(path))
            reg.results = d.get("results", {})
            reg.circuits = d.get("circuits", {})
        return reg

    # ---------------- updates ----------------
    def note_patterns(self, pattern_ids: List[str], circuit: CircuitRecord,
                      battery: Dict):
        """Attach a causal battery to every pattern that nominated the circuit;
        evidence ladder + status computed from the battery (§17)."""
        lvl = evidence_level(battery)
        st = status_from_evidence(lvl, battery)
        for pid in pattern_ids:
            cur = self.results.get(pid, {})
            prev_lvl = cur.get("evidence_level", 0)
            if lvl >= prev_lvl:
                cur.update({"evidence_level": lvl,
                            "evidence_level_name": EVIDENCE_LEVELS[lvl],
                            "status": st,
                            "circuits": sorted(set(cur.get("circuits", []) +
                                                   [circuit.circuit_id])),
                            "battery_summary": _battery_summary(battery),
                            "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
            else:
                cur["circuits"] = sorted(set(cur.get("circuits", []) +
                                             [circuit.circuit_id]))
                cur.setdefault("notes", []).append(
                    f"{circuit.circuit_id}: lower evidence ({lvl}) than existing "
                    f"record ({prev_lvl}) — kept existing")
            self.results[pid] = cur

    def set_unsupported(self, pattern_id: str, reason: str, detail: Optional[dict] = None):
        cur = self.results.get(pattern_id, {})
        cur.update({"evidence_level": cur.get("evidence_level", 0),
                    "evidence_level_name": EVIDENCE_LEVELS.get(cur.get("evidence_level", 0)),
                    "status": "UNSUPPORTED", "reason": reason,
                    "detail": detail or {}, "notes": cur.get("notes", []),
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.results[pattern_id] = cur

    def set_not_tested(self, pattern_id: str, reason: str):
        self.results[pattern_id] = {"evidence_level": 0,
                                    "evidence_level_name": EVIDENCE_LEVELS[0],
                                    "status": "NOT_TESTED", "reason": reason,
                                    "updated": time.strftime("%Y-%m-%d %H:%M:%S")}

    def add_circuit(self, circuit: CircuitRecord):
        self.circuits[circuit.circuit_id] = circuit.to_dict()

    # ---------------- queries ----------------
    def by_phase(self, phase: str) -> List[PatternRecord]:
        return [r for r in self.records if r.phase == phase]

    def status_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(r.get("status", "NOT_TESTED")
                            for r in self.results.values()))

    def summary_lines(self) -> List[str]:
        lines = []
        from collections import Counter
        tested = [r for r in self.records if r.pattern_id in self.results]
        lv = Counter(self.results[r.pattern_id].get("evidence_level", 0)
                     for r in tested)
        lines.append(f"patterns tested: {len(tested)}/100; "
                     f"evidence ladder histogram: " +
                     ", ".join(f"L{k}({EVIDENCE_LEVELS[k]}):{v}"
                               for k, v in sorted(lv.items())))
        for r in self.records:
            res = self.results.get(r.pattern_id)
            if res:
                lines.append(f"  {r.pattern_id} {r.name[:38]:<38} "
                             f"[{r.phase}] L{res.get('evidence_level', 0)} "
                             f"{res.get('status', 'NOT_TESTED')}")
        return lines


def _battery_summary(battery: Dict) -> Dict:
    out = {}
    if battery.get("detection"):
        out["detection"] = {k: battery["detection"].get(k)
                            for k in ("significant", "p", "stat")}
    if battery.get("task_correlation"):
        out["task_correlation"] = {k: battery["task_correlation"].get(k)
                                   for k in ("significant", "selectivity")}
    if battery.get("ablation"):
        a = battery["ablation"]
        out["ablation"] = {k: a.get(k) for k in
                           ("effect", "significant", "effect_specific",
                            "specificity", "max_single")}
    if battery.get("restoration"):
        out["restoration"] = {k: battery["restoration"].get(k)
                              for k in ("recovered_pct",)}
    if battery.get("positive_intervention"):
        out["positive_intervention"] = {
            k: battery["positive_intervention"].get(k)
            for k in ("significant", "max_abs_change")}
    if battery.get("transfer"):
        out["transfer"] = {k: battery["transfer"].get(k)
                           for k in ("status", "delta")}
    return out
