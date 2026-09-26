"""[WORKING] Plain-text controlled MC suites for real HF models.

Small, locally executed, honest. NOT MMLU/HumanEval equivalents — but real
measurements on real models, with held-out discipline.
"""
from __future__ import annotations

import random
from typing import List

from .suites import Suite, SuiteItem


def build_text_suites(n: int = 20, seed: int = 31337) -> List[Suite]:
    rng = random.Random(seed)
    suites: List[Suite] = []

    def mc(name, domain, items):
        out = []
        for prompt, correct, wrongs in items:
            opts = [correct] + wrongs[:3]
            rng.shuffle(opts)
            out.append(SuiteItem(domain, prompt, opts, opts.index(correct)))
        suites.append(Suite(name, domain, out))

    math_items = []
    for _ in range(n):
        a, b = rng.randint(2, 40), rng.randint(2, 40)
        want_max = rng.random() < 0.5
        ans = max(a, b) if want_max else min(a, b)
        w1 = "max" if want_max else "min"
        other = min(a, b) if want_max else max(a, b)
        math_items.append((f"Question: what is the {w1} of {a} and {b}? Answer:",
                           f" {ans}", [f" {other}", f" {a+b}", f" {abs(a-b)}"]))
    mc("text-math", "math", math_items)

    reason_items = []
    names = ["Anna", "Bill", "Cara", "Dan", "Elena", "Fred"]
    for _ in range(n):
        A, B, C = rng.sample(names, 3)
        prop = rng.choice(["older", "taller", "faster"])
        reason_items.append((
            f"{A} is {prop} than {B}. {B} is {prop} than {C}. Who is the {prop[0]}{'est' if prop!='older' else 'ldest'}? Answer:",
            f" {A}", [f" {B}", f" {C}", f" None"]))
    mc("text-reasoning", "reasoning", reason_items)

    code_items = [
        ("Q: What does this Python function return? def f(): return 2 + 3\nAnswer:", " 5", [" 23", " 6", " None"]),
        ("Q: Which Python keyword defines a function?\nAnswer:", " def", [" var", " loop", " class"]),
        ("Q: What is the value of len([1, 2, 3]) in Python?\nAnswer:", " 3", [" 2", " 4", " 6"]),
        ("Q: In Python, what does 'if __name__ == \"__main__\":' mean?\nAnswer:",
         " script run directly", [" import error", " loop forever", " nothing"]),
    ]
    while len(code_items) < n:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        code_items.append((f"Q: What does print({a} * {b}) output in Python?\nAnswer:", f" {a*b}",
                           [f" {a+b}", f" {a}-b", f" {a}{b}"]))
    mc("text-coding", "coding", code_items)

    lang_items = [
        ("Q: Choose the grammatically correct sentence:\nAnswer:", " The cat sat on the mat.",
         [" The cat sat the mat on.", " Sat the cat on the mat.", " The on mat cat sat."]),
        ("Q: Plural of 'mouse' is:\nAnswer:", " mice", [" mouses", " mouse", " mices"]),
    ] * 10
    mc("text-language", "language", lang_items[:n])

    know_items = [
        ("Q: What is the capital of France?\nAnswer:", " Paris", [" London", " Rome", " Berlin"]),
        ("Q: What is the capital of Japan?\nAnswer:", " Tokyo", [" Beijing", " Seoul", " Osaka"]),
        ("Q: Is a whale a fish or a mammal?\nAnswer:", " mammal", [" fish", " bird", " insect"]),
        ("Q: How many continents are there?\nAnswer:", " seven", [" five", " six", " eight"]),
        ("Q: What gas do plants absorb?\nAnswer:", " carbon dioxide", [" oxygen", " helium", " nitrogen"]),
    ] * 8
    mc("text-knowledge", "knowledge", know_items[:n])

    multi_items = [
        ("Translate to French: 'hello'\nAnswer:", " bonjour", [" hola", " danke", " ciao"]),
        ("Translate to Spanish: 'thank you'\nAnswer:", " gracias", [" merci", " bitte", " arigato"]),
        ("Translate to German: 'yes'\nAnswer:", " ja", [" oui", " si", " nein"]),
        ("Translate to Hindi (Latin script): 'water'\nAnswer:", " paani", [" aag", " hawa", " mitti"]),
        ("Translate to Bengali (Latin script): 'fire'\nAnswer:", " aag", [" jal", " hawa", " matrix"]),
    ] * 8
    mc("text-multilingual", "multilingual", multi_items[:n])

    agent_items = [
        ("Q: To compute 23 * 45 quickly, which tool do you use?\nAnswer:", " calculator", [" image viewer", " calendar", " paint"]),
        ("Q: To find today's weather, which tool do you use?\nAnswer:", " search", [" calculator", " file editor", " compiler"]),
        ("Q: To remember a note for later, which tool do you use?\nAnswer:", " memory", [" calculator", " search", " paint"]),
    ] * 7
    mc("text-agent", "agent", agent_items[:n])
    return suites


def load_text_suite(name: str, n: int = 20) -> Suite:
    for s in build_text_suites(n=n):
        if s.name == name:
            return s
    raise KeyError(name)
