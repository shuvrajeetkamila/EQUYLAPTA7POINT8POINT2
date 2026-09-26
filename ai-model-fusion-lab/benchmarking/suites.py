"""[WORKING] Suite definitions.

Two suite families:
  * micro suites  — generated from models.corpora with held-out values,
    used for synthetic models AND anything sharing the micro tokenizer.
  * text suites   — plain controlled MC tests for real HF models (small,
    honest, locally executed; NOT MMLU-scale).

Suites are NEVER trained on (spec §16). Training corpora use the same
TEMPLATES but different values/name-pools — the gap is genuine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List

from models.corpora import DOMAINS, gen_suite


@dataclass
class SuiteItem:
    domain: str
    prompt: str
    options: List[str]
    answer: int


@dataclass
class Suite:
    name: str
    domain: str
    items: List[SuiteItem] = field(default_factory=list)

    def to_dict(self):
        return {"name": self.name, "domain": self.domain,
                "items": [asdict(i) for i in self.items]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["domain"],
                   [SuiteItem(**i) for i in d["items"]])


def data_dir() -> str:
    d = os.environ.get(
        "FUSIONLAB_DATA",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "fusionlab_data"))
    os.makedirs(d, exist_ok=True)
    return d


def micro_suite(domain: str, n: int = 60, seed: int = 777) -> Suite:
    items = gen_suite(domain, n, seed=seed)
    return Suite(f"micro-{domain}", domain, [SuiteItem(**asdict(i)) for i in items])


def all_micro_suites(n: int = 60) -> List[Suite]:
    return [micro_suite(d, n) for d in DOMAINS]


def freeze_private(suites: List[Suite]) -> str:
    """Freeze a 'private' holdout copy the optimizer never sees (spec §16)."""
    path = os.path.join(data_dir(), "private_suites.json")
    json.dump([s.to_dict() for s in suites], open(path, "w"))
    return path


def load_suite(name: str) -> Suite:
    priv = os.path.join(data_dir(), "private_suites.json")
    if os.path.exists(priv):
        for d in json.load(open(priv)):
            if d["name"] == name:
                return Suite.from_dict(d)
    dom = name.replace("micro-", "")
    if dom in DOMAINS:
        return micro_suite(dom)
    raise KeyError(name)
