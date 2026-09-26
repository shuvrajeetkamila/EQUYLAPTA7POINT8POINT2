"""[WORKING] Modular capability suites (M3, spec §20).

Stronger than the micro suites but still CPU-tractable and held-out by
construction. Every suite has TRAIN / HELD-OUT splits via disjoint value
ranges (train ranges never appear in held-out items). The private frozen
micro suites remain the final held-out test; these modular suites provide
refitting/calibration training data (their TRAIN split) and intermediate
evaluation.

Suites: algebra, pattern, code-transform, word-problems, paraphrase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# value-range partitioning: TRAIN ranges vs HELD-OUT ranges are disjoint
RANGES = {
    "algebra":  {"a": ((1, 6), (7, 12)), "x": ((1, 8), (9, 16)), "b": ((1, 9), (10, 19))},
    "pattern":  {"start": ((0, 15), (16, 35)), "step": ((2, 5), (6, 9))},
    "wordprob": {"n": ((1, 8), (9, 16))},
    "cxform":   {"names": 0, },  # name-pair partition below
    "paraphrase": {"pairs": 0},
}

OPS = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------- generators
def gen_algebra_train(rng, n) -> List[str]:
    out = []
    for _ in range(n):
        a = _ri(rng, *RANGES["algebra"]["a"][0])
        x = _ri(rng, *RANGES["algebra"]["x"][0])
        b = _ri(rng, *RANGES["algebra"]["b"][0])
        c = a * x + b
        out.append(f"solve x : {a} x + {b} = {c} , x is {x}")
    return out


def gen_pattern_train(rng, n) -> List[str]:
    out = []
    for _ in range(n):
        start = _ri(rng, *RANGES["pattern"]["start"][0])
        step = _ri(rng, *RANGES["pattern"]["step"][0])
        seq = [start + step * i for i in range(4)]
        out.append(f"pattern {seq[0]} {seq[1]} {seq[2]} {seq[3]} next is {seq[3] + step}")
    return out


def gen_wordprob_train(rng, n) -> List[str]:
    out = []
    for _ in range(n):
        n1 = _ri(rng, *RANGES["wordprob"]["n"][0])
        n2 = _ri(rng, *RANGES["wordprob"]["n"][0])
        n3 = _ri(rng, *RANGES["wordprob"]["n"][0])
        total = n1 + n2 - n3
        if total < 0:
            n3, total = n1 + n2, 0
        out.append(f"Tom has {n1} apples . He buys {n2} more . He gives away {n3} . Total = {total}")
    return out


CX_NAMES = [("add", "plus"), ("sub", "minus"), ("mul", "times"), ("maxof", "bigger")]
CX_OPS = [("+", "-"), ("-", "+"), ("*", "//"), ("+", "*")]


def gen_cxform_train(rng, n) -> List[str]:
    out = []
    for _ in range(n):
        i = int(rng.integers(0, len(CX_NAMES)))
        src, dst = CX_NAMES[i]
        op = CX_OPS[i][0]
        out.append(f"rename {src} to {dst} : def {src} ( a , b ) : return a {op} b becomes def {dst} ( a , b ) : return a {op} b")
    return out


PARAPHRASE = [("small", "big"), ("hot", "cold"), ("fast", "slow"), ("up", "down"),
              ("open", "shut"), ("day", "night"), ("full", "empty"), ("wet", "dry")]
TRAIN_PAIRS = PARAPHRASE[:5]
HELD_PAIRS = PARAPHRASE[5:]


def gen_paraphrase_train(rng, n) -> List[str]:
    out = []
    for _ in range(n):
        a, b = TRAIN_PAIRS[int(rng.integers(0, len(TRAIN_PAIRS)))]
        out.append(f"the opposite of {a} is {b}")
    return out


TRAIN_GEN = {
    "algebra": gen_algebra_train, "pattern": gen_pattern_train,
    "wordprob": gen_wordprob_train, "cxform": gen_cxform_train,
    "paraphrase": gen_paraphrase_train,
}


# ---------------------------------------------------------------- MC suites
def modular_suite(domain: str, n: int = 40, split: str = "heldout", seed: int = 4321):
    """Returns Suite with items; split in {'train','heldout'} with disjoint values."""
    from benchmarking.suites import Suite, SuiteItem
    rng = np.random.default_rng(seed + (0 if split == "train" else 777))
    idx = 1 if split == "heldout" else 0
    items: List[SuiteItem] = []

    def add(prompt, correct, wrongs):
        opts = [str(correct)] + [str(w) for w in wrongs[:3]]
        order = rng.permutation(len(opts))
        opts = [opts[i] for i in order]
        items.append(SuiteItem(domain, prompt, opts, opts.index(str(correct))))

    if domain == "algebra":
        for _ in range(n):
            a = _ri(rng, *RANGES["algebra"]["a"][idx])
            x = _ri(rng, *RANGES["algebra"]["x"][idx])
            b = _ri(rng, *RANGES["algebra"]["b"][idx])
            c = a * x + b
            add(f"solve x : {a} x + {b} = {c} , x is", x, [x + 1, x - 1, c - b])
    elif domain == "pattern":
        for _ in range(n):
            start = _ri(rng, *RANGES["pattern"]["start"][idx])
            step = _ri(rng, *RANGES["pattern"]["step"][idx])
            seq = [start + step * i for i in range(4)]
            add(f"pattern {seq[0]} {seq[1]} {seq[2]} {seq[3]} next is", seq[3] + step,
                [seq[3] + step + 1, seq[3] + 1, seq[3] + step - 1])
    elif domain == "wordprob":
        for _ in range(n):
            n1 = _ri(rng, *RANGES["wordprob"]["n"][idx])
            n2 = _ri(rng, *RANGES["wordprob"]["n"][idx])
            n3 = _ri(rng, *RANGES["wordprob"]["n"][idx])
            total = max(0, n1 + n2 - n3)
            add(f"Tom has {n1} apples . He buys {n2} more . He gives away {n3} . Total =",
                total, [total + 1, max(0, n1 + n2), max(0, total - 1)])
    elif domain == "cxform":
        names = CX_NAMES if split == "train" else [("divi", "share"), ("modof", "remainder")]
        ops = {"+": ["-", "*"], "-": ["+", "*"], "//": ["+", "-"], "*": ["+", "-"]}
        for _ in range(n):
            src, dst = names[int(rng.integers(0, len(names)))]
            op = list(ops)[int(rng.integers(0, 2))] if split == "train" else "+"
            wrongs = [f"def {src} ( a , b ) : return a {op} b",
                      f"def {dst} ( a , b ) : return b {op} a"]
            add(f"rename {src} to {dst} : def {src} ( a , b ) : return a {op} b becomes",
                f"def {dst} ( a , b ) : return a {op} b", wrongs)
    elif domain == "paraphrase":
        pairs = TRAIN_PAIRS if split == "train" else HELD_PAIRS
        allw = [b for _, b in PARAPHRASE]
        for _ in range(n):
            a, b = pairs[int(rng.integers(0, len(pairs)))]
            wrongs = [w for w in allw if w != b][:3]
            add(f"the opposite of {a} is", b, wrongs)
    else:
        raise KeyError(domain)
    return Suite(f"modular-{domain}", domain, items)


MODULAR_DOMAINS = ["algebra", "pattern", "wordprob", "cxform", "paraphrase"]
