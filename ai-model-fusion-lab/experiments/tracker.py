"""[WORKING] Experiment database (JSONL + dedupe by config hash).

Every experiment records: id, sources, config, results, timings, license,
failure notes. Identical configurations are never re-run (spec §14).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

from models.base import config_hash


class ExperimentTracker:
    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "experiments.jsonl")
        self._index_cache: Optional[Dict[str, dict]] = None

    def _load_index(self) -> Dict[str, dict]:
        if self._index_cache is None:
            idx = {}
            if os.path.exists(self.path):
                for line in open(self.path):
                    try:
                        rec = json.loads(line)
                        idx[rec["config_hash"]] = rec
                    except Exception:
                        continue
            self._index_cache = idx
        return self._index_cache

    def all(self):
        return list(self._load_index().values())

    def seen(self, config: dict) -> bool:
        return config_hash(config) in self._load_index()

    def find(self, config: dict) -> Optional[dict]:
        return self._load_index().get(config_hash(config))

    def record(self, exp_type: str, config: dict, sources, results: dict,
               target_suite: str = "", notes: str = "", license: str = "unknown",
               status: str = "COMPLETED") -> dict:
        idx = self._load_index()
        h = config_hash(config)
        if h in idx:
            return idx[h]
        n = len(idx) + 1
        rec = {
            "id": f"EXP-{n:05d}",
            "config_hash": h,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": exp_type,
            "sources": sources,
            "config": config,
            "results": results,
            "target_suite": target_suite,
            "license": license,
            "status": status,
            "notes": notes,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        idx[h] = rec
        return rec

    def leaderboard(self, metric: str = "mean") -> list:
        rows = []
        for rec in self._load_index().values():
            vals = [v for v in rec["results"].values() if isinstance(v, (int, float))]
            if not vals:
                continue
            rows.append({"id": rec["id"], "type": rec["type"],
                         "target": rec.get("target_suite", ""),
                         "score": round(sum(vals) / len(vals), 1),
                         "results": rec["results"]})
        rows.sort(key=lambda r: -r["score"])
        return rows
