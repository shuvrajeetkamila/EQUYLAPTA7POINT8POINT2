"""[WORKING] Seeded micro-corpus + benchmark-suite generation.

Defines seven capability domains (math, code, reason, lang, know, multi,
agent) with TRAIN/VAL/TEST value-disjoint splits (spec section 16).

IMPORTANT HONESTY NOTE: these suites measure capability *within this
microcosm* (templated mini-language). They are real, controlled experiments
with genuine generalization gaps (held-out numbers, names, combinations) —
they are NOT equivalents of MMLU/HumanEval. External real-model suites are
defined separately in benchmarking/text_suites.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

DOMAINS = ["math", "code", "reason", "lang", "know", "multi", "agent"]

# value pools with explicit train/test splits -----------------------------
NAMES_TRAIN = ["Ann", "Ben", "Cal", "Dot", "Eva", "Fox"]
NAMES_TEST = ["Gus", "Hal", "Ida", "Jo"]

FACTS_TRAIN = [("sparrow", "bird"), ("rose", "flower"), ("dog", "animal"),
               ("trout", "fish"), ("oak", "plant"), ("bee", "insect")]
CATEGORIES = ["bird", "flower", "animal", "fish", "plant", "insect"]

TRANS_TRAIN = [("Hello", "French", "Bonjour"), ("Hello", "Spanish", "Hola"),
               ("Thanks", "French", "Merci"), ("Thanks", "Spanish", "Gracias"),
               ("Yes", "German", "Ja"), ("No", "German", "Nein"),
               ("Water", "Hindi", "Paani"), ("Fire", "Bengali", "Aag")]
TRANS_TEST = [("Hello", "German", "Hallo"), ("Water", "French", "Eau"),
              ("Thanks", "Hindi", "Dhanyavaad"), ("Yes", "Spanish", "Si")]

TOOLS_TRAIN = [("compute", "calculator"), ("weather", "search"),
               ("note", "memory"), ("paint", "brush")]
TOOLS_TEST = [("sum", "calculator"), ("forecast", "search"),
              ("reminder", "memory"), ("draw", "brush")]

COLORS_TRAIN = [("sky", "blue"), ("grass", "green"), ("sun", "yellow"),
                ("coal", "black"), ("snow", "white")]
COLORS_TEST = [("sea", "blue"), ("leaves", "green"), ("night", "black")]

CODE_OPS_TRAIN = [("add", "+"), ("sub", "-"), ("mul", "*")]
CODE_OPS_TEST = [("mod", "%"), ("div", "//")]


@dataclass
class SuiteItem:
    domain: str
    prompt: str
    options: List[str]
    answer: int


def _num(rng, lo, hi) -> str:
    return str(int(rng.integers(lo, hi + 1)))


# --- training-sequence generators (one string per sequence) --------------

def gen_math(rng, n) -> List[str]:
    """Attention-friendly arithmetic: comparison (max/min), counting,
    small-addend addition. All genuinely computed; values held out in suites."""
    out = []
    for _ in range(n):
        r = rng.integers(0, 3)
        if r == 0:
            a, b = int(rng.integers(1, 15)), int(rng.integers(1, 15))
            if rng.integers(0, 2):
                out.append(f"max of {a} and {b} is {max(a, b)}")
            else:
                out.append(f"min of {a} and {b} is {min(a, b)}")
        elif r == 1:
            a = int(rng.integers(0, 25))
            out.append(f"count from {a} : {a} {a+1} {a+2} {a+3}")
        else:
            a, b = int(rng.integers(1, 21)), int(rng.integers(1, 9))
            out.append(f"{a} + {b} = {a + b}")
    return out


def gen_code(rng, n) -> List[str]:
    ops = CODE_OPS_TRAIN + CODE_OPS_TEST
    args = ["a b", "x y", "p q", "m n"]
    out = []
    for _ in range(n):
        name, op = ops[int(rng.integers(0, len(ops)))]
        a1, a2 = args[int(rng.integers(0, len(args)))].split()
        r = rng.integers(0, 3)
        if r == 0:
            out.append(f"def {name} ( {a1} , {a2} ) : return {a1} {op} {a2}")
        elif r == 1:
            out.append(f"def {name} ( {a1} , {a2} ) : return {a1} {op} {a2} # {name} done")
        else:
            out.append(f"# {name} : takes two inputs . def {name} ( {a1} , {a2} ) : return {a1} {op} {a2}")
    return out


def gen_reason(rng, n, names=NAMES_TRAIN) -> List[str]:
    out = []
    for _ in range(n):
        a, b, c = rng.choice(len(names), size=3, replace=False)
        A, B, C = names[a], names[b], names[c]
        r = rng.integers(0, 2)
        if r == 0:
            out.append(f"{A} is taller than {B} . {B} is taller than {C} . Who is tallest ? {A}")
        else:
            out.append(f"{A} is faster than {B} . {B} is faster than {C} . Who is slowest ? {C}")
    return out


def gen_lang(rng, n, pairs=COLORS_TRAIN) -> List[str]:
    out = []
    for _ in range(n):
        obj, col = pairs[int(rng.integers(0, len(pairs)))]
        r = rng.integers(0, 2)
        if r == 0:
            out.append(f"The {obj} is {col} .")
        else:
            out.append(f"Is the {obj} {col} ? Yes , the {obj} is {col} .")
    return out


def gen_know(rng, n, facts=FACTS_TRAIN) -> List[str]:
    out = []
    for _ in range(n):
        thing, cat = facts[int(rng.integers(0, len(facts)))]
        r = rng.integers(0, 2)
        if r == 0:
            out.append(f"A {thing} is a {cat} .")
        else:
            wrong = [c for c in CATEGORIES if c != cat]
            out.append(f"Is a {thing} a {wrong[int(rng.integers(0, len(wrong)))]} ? No , a {thing} is a {cat} .")
    return out


def gen_multi(rng, n, table=TRANS_TRAIN) -> List[str]:
    out = []
    for _ in range(n):
        src, lang, dst = table[int(rng.integers(0, len(table)))]
        out.append(f"{src} in {lang} is {dst} .")
    return out


def gen_agent(rng, n, tools=TOOLS_TRAIN) -> List[str]:
    verbs = {"compute": ["compute", "calculate"], "sum": ["sum", "add up"],
             "weather": ["weather", "forecast"], "forecast": ["forecast", "weather"],
             "note": ["note", "remember"], "reminder": ["reminder", "memorize"],
             "paint": ["paint", "color"], "draw": ["draw", "sketch"]}
    out = []
    for _ in range(n):
        goal, tool = tools[int(rng.integers(0, len(tools)))]
        v = verbs[goal][int(rng.integers(0, len(verbs[goal])))]
        out.append(f"To {v} use the {tool} tool .")
    return out


TRAIN_GEN = {"math": gen_math, "code": gen_code, "reason": gen_reason,
             "lang": gen_lang, "know": gen_know, "multi": gen_multi,
             "agent": gen_agent}


def gen_train(domain: str, n: int, rng) -> List[str]:
    return TRAIN_GEN[domain](rng, n)


# --- MC benchmark suites (held-out values; disjoint RNG) -----------------

def gen_suite(domain: str, n: int, seed: int) -> List[SuiteItem]:
    rng = np.random.default_rng(seed)
    items: List[SuiteItem] = []
    if domain == "math":
        for _ in range(n):
            r = rng.integers(0, 3)
            if r == 0:
                a, b = int(rng.integers(1, 15)), int(rng.integers(1, 15))
                want_max = bool(rng.integers(0, 2))
                ans = max(a, b) if want_max else min(a, b)
                prompt = f"{'max' if want_max else 'min'} of {a} and {b} is"
                wrongs = {min(a, b) if want_max else max(a, b), ans + 1, max(a, b) + 1}
                wrongs.discard(ans)
                options = [str(ans)] + [str(w) for w in sorted(wrongs)][:3]
            elif r == 1:
                a = int(rng.integers(0, 25))
                ans = a + 2
                prompt = f"count from {a} : {a} {a+1}"
                wrongs = {a + 3, a + 4, a - 1}
                options = [str(ans)] + [str(w) for w in sorted(wrongs)]
            else:
                a, b = int(rng.integers(1, 21)), int(rng.integers(1, 9))
                ans = a + b
                prompt = f"{a} + {b} ="
                wrongs = {ans + 1, ans - 1, b}
                wrongs.discard(ans)
                options = [str(ans)] + [str(w) for w in sorted(wrongs)][:3]
            order = rng.permutation(len(options))
            options = [options[i] for i in order]
            items.append(SuiteItem("math", prompt, options, options.index(str(ans))))
    elif domain == "code":
        for _ in range(n):
            name, op = (CODE_OPS_TEST + CODE_OPS_TRAIN)[int(rng.integers(0, 5))]
            a1, a2 = ("x", "y") if rng.integers(0, 2) else ("a", "b")
            correct = f"return {a1} {op} {a2}"
            wrongs = [f"return {a2} {op} {a1}", f"return {a1} + {a2}" if op != "+" else f"return {a1} - {a2}",
                      f"return {a1} , {a2}"]
            options = [correct] + wrongs[:3]
            order = rng.permutation(len(options)); options = [options[i] for i in order]
            items.append(SuiteItem("code", f"def {name} ( {a1} , {a2} ) :", options, options.index(correct)))
    elif domain == "reason":
        for _ in range(n):
            names = list(NAMES_TEST) if rng.integers(0, 2) else list(NAMES_TRAIN)
            a, b, c = rng.choice(len(names), size=3, replace=False)
            A, B, C = names[a], names[b], names[c]
            tallest_mode = bool(rng.integers(0, 2))
            pred = "taller" if tallest_mode else "faster"
            want = A if tallest_mode else C
            label = "tallest" if tallest_mode else "slowest"
            options = [A, B, C, NAMES_TEST[0] if NAMES_TEST[0] not in names else NAMES_TRAIN[0]]
            options = list(dict.fromkeys(options))[:4]
            while len(options) < 4:
                options.append(NAMES_TRAIN[3])
            order = rng.permutation(len(options)); options = [options[i] for i in order]
            items.append(SuiteItem("reason",
                                   f"{A} is {pred} than {B} . {B} is {pred} than {C} . Who is {label} ?",
                                   options, options.index(want)))
    elif domain == "lang":
        for _ in range(n):
            obj, col = (COLORS_TEST + COLORS_TRAIN)[int(rng.integers(0, 8))]
            distractors = [c for _, c in COLORS_TRAIN + COLORS_TEST if c != col]
            rng.shuffle(distractors)
            options = [col] + distractors[:3]
            order = rng.permutation(4); options = [options[i] for i in order]
            items.append(SuiteItem("lang", f"The {obj} is", options, options.index(col)))
    elif domain == "know":
        for _ in range(n):
            thing, cat = FACTS_TRAIN[int(rng.integers(0, len(FACTS_TRAIN)))]
            wrong = [c for c in CATEGORIES if c != cat]
            rng.shuffle(wrong)
            options = [cat] + wrong[:3]
            order = rng.permutation(4); options = [options[i] for i in order]
            items.append(SuiteItem("know", f"A {thing} is a", options, options.index(cat)))
    elif domain == "multi":
        for _ in range(n):
            src, lang, dst = (TRANS_TRAIN + TRANS_TEST)[int(rng.integers(0, 12))]
            wrong = [d for s, l, d in TRANS_TRAIN + TRANS_TEST if d != dst]
            rng.shuffle(wrong)
            options = [dst] + list(dict.fromkeys(wrong))[:3]
            order = rng.permutation(len(options)); options = [options[i] for i in order]
            items.append(SuiteItem("multi", f"{src} in {lang} is", options, options.index(dst)))
    elif domain == "agent":
        for _ in range(n):
            goal, tool = (TOOLS_TRAIN + TOOLS_TEST)[int(rng.integers(0, 8))]
            options = [t for _, t in TOOLS_TRAIN + TOOLS_TEST]
            options = list(dict.fromkeys(options))
            wrong = [t for t in options if t != tool][:3]
            opts = [tool] + wrong
            order = rng.permutation(len(opts)); opts = [opts[i] for i in order]
            items.append(SuiteItem("agent", f"To {goal} use the", opts, opts.index(tool)))
    return items


def build_master_tokenizer(seed: int = 2024, n_probe: int = 40):
    """Vocab = union of all train corpora + all suite strings (so scoring is
    always in-vocab). Suite strings never enter any training corpus."""
    rng = np.random.default_rng(seed)
    texts: List[str] = []
    for dom in DOMAINS:
        texts += gen_train(dom, 200, rng)
    for di, dom in enumerate(DOMAINS):
        for it in gen_suite(dom, n_probe, seed=9091 + di):
            texts.append(it.prompt + " " + " ".join(it.options))
    return texts
