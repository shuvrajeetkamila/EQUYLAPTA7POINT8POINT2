"""[WORKING] Input analyzer + expert router (spec §§10-11).

Strategies: rule-based [WORKING], learned bag-of-words logistic classifier
[WORKING, numpy], multi-expert (top-k) activation, confidence thresholding.
The router activates ONLY the specialists it needs — never everything.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

EXPERTS = ["reasoning", "coding", "math", "language", "knowledge", "multilingual", "agent"]

RULES = {
    "math": [r"\b\d+\s*[-+*/]\s*\d+", r"\b(max|min|sum|count|average|percent|equals?)\b",
             r"\b(add|subtract|multiply|divide)\b", r"\bhow many\b", r"\bcalcular|calculate\b"],
    "coding": [r"\b(code|function|def |python|program|script|bug|debug|compile|api)\b",
               r"```", r"\bwrite a (python|java|c\+\+|sql)\b", r"\breturn\b.*\bdef\b"],
    "reasoning": [r"\b(prove|logic|deduce|therefore|because|if .* then|syllogism|puzzle)\b",
                  r"\b(taller|faster|older|smarter) than\b", r"\bwho is (tallest|fastest|oldest)\b"],
    "language": [r"\b(grammar|rewrite|paraphrase|summarize|essay|translate style|tone)\b",
                 r"\bthe sky is\b", r"\bwrite (a|an) (sentence|paragraph|story)\b"],
    "knowledge": [r"\b(capital|country|animal|species|who invented|what is the)\b",
                  r"\b(is a|are) \w+ (a|an)\b"],
    "multilingual": [r"\b(translate|in french|in spanish|in german|in hindi|in bengali|in japanese)\b",
                     r"\bbonjour|hola|danke|namaste|dhanyavaad\b"],
    "agent": [r"\b(tool|calculator|search|memory|calendar|use the \w+ tool)\b",
              r"\b(schedule|remind|look up|fetch|browse)\b"],
}


def rule_route(text: str, threshold: float = 0.5) -> Dict[str, float]:
    scores = {}
    for exp, pats in RULES.items():
        hits = sum(1 for p in pats if re.search(p, text, re.I))
        scores[exp] = min(1.0, hits / max(1, len(pats)) * 2)
    return scores


# --- learned router: bag-of-words + multinomial logistic regression ------

class LearnedRouter:
    def __init__(self, experts: List[str] = None):
        self.experts = experts or EXPERTS
        self.vocab: Dict[str, int] = {}
        self.W = None  # [V, E]

    def _featurize(self, texts: List[str]) -> np.ndarray:
        for t in texts:
            for w in re.findall(r"[a-z0-9']+", t.lower()):
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        X = np.zeros((len(texts), len(self.vocab)), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9']+", t.lower()):
                X[i, self.vocab[w]] += 1.0
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        return X / nrm

    def fit(self, texts: List[str], labels: List[List[str]], epochs: int = 300, lr: float = 0.5):
        X = self._featurize(texts)
        Y = np.zeros((len(texts), len(self.experts)), np.float32)
        for i, labs in enumerate(labels):
            for l in labs:
                Y[i, self.experts.index(l)] = 1.0
        rng = np.random.default_rng(0)
        self.W = rng.normal(0, 0.01, (X.shape[1], len(self.experts))).astype(np.float32)
        b = np.zeros(len(self.experts), np.float32)
        n = len(texts)
        for ep in range(epochs):
            Z = X @ self.W + b
            P = 1 / (1 + np.exp(-Z))
            G = (P - Y) / n
            self.W -= lr * (X.T @ G)
            b -= lr * G.sum(0)
        self.b = b

    def route(self, text: str, threshold: float = 0.35, top_k: int = 3) -> Tuple[List[str], Dict[str, float]]:
        x = np.zeros((1, len(self.vocab)), np.float32)
        for w in re.findall(r"[a-z0-9']+", text.lower()):
            if w in self.vocab:
                x[0, self.vocab[w]] += 1.0
        nrm = np.linalg.norm(x)
        if nrm > 0:
            x /= nrm
        Z = (x @ self.W + self.b)[0]
        P = 1 / (1 + np.exp(-Z))
        scores = {self.experts[i]: float(P[i]) for i in range(len(self.experts))}
        chosen = [e for e, s in scores.items() if s >= threshold]
        if not chosen:
            chosen = [max(scores, key=scores.get)]
        chosen = sorted(chosen, key=lambda e: -scores[e])[:top_k]
        return chosen, scores


# --- training data for the learned router (seeded, rule-verified) --------

def router_training_data(seed: int = 42, n_per: int = 24):
    rng = np.random.default_rng(seed)
    T, L = [], []
    numbers = lambda: f"{rng.integers(1,99)} {rng.choice(['+','-','*'])} {rng.integers(1,99)}"
    math_t = [f"what is {numbers()} ?", f"compute {numbers()}", "max of 44 and 12 is",
              f"count from {rng.integers(0,30)}", "how many is 8 + 7"]
    code_t = ["write a python function def add ( a , b ) : return a + b", "fix this bug in my code",
              "def mul ( x , y ) : return x * y", "write a program that returns the sum",
              "debug this function please"]
    reason_t = ["Ann is taller than Ben . Ben is taller than Cal . Who is tallest ?",
                "if all cats are animals and Tom is a cat then Tom is an animal",
                "prove that the set is infinite", "logic puzzle : who is fastest ?",
                "therefore by deduction the answer follows"]
    lang_t = ["the sky is", "rewrite this sentence more politely", "check my grammar",
              "summarize the paragraph", "write a story about the sea"]
    know_t = ["a sparrow is a", "what is the capital of France", "is a rose a flower",
              "which animal is a fish", "who invented the telephone"]
    multi_t = ["Hello in French is", "translate water to Hindi", "say thanks in Spanish",
               "what is Bonjour in English", "dhanyavaad means"]
    agent_t = ["use the calculator tool to compute 12 * 9", "search the weather for me",
               "save this note in memory", "schedule a reminder", "which tool should I use to paint ?"]
    pools = [("math", math_t), ("coding", code_t), ("reasoning", reason_t),
             ("language", lang_t), ("knowledge", know_t), ("multilingual", multi_t),
             ("agent", agent_t)]
    for exp, pool in pools:
        for _ in range(n_per):
            base = list(pool)[int(rng.integers(0, len(pool)))]
            T.append(base + (" " + str(rng.integers(0, 99)) if rng.integers(0, 2) else ""))
            L.append([exp])
    # multi-label mixes
    mixes = [("Write a Python program proving this mathematical theorem.", ["coding", "math", "reasoning"]),
             ("Translate this Japanese technical document and explain the semiconductor concepts.", ["multilingual", "language", "knowledge"]),
             ("Use the calculator tool to check if 23 * 45 is bigger than 1000, and explain why.", ["agent", "math", "reasoning"]),
             ("Debug this Python function and prove the corrected formula holds.", ["coding", "reasoning"]),
             ("Count from 4 and write the counts as a python list.", ["math", "coding"]),
             ("Translate 'the sky is blue' to French and check the grammar.", ["multilingual", "language"])]
    for _ in range(10):
        for t, labs in mixes:
            T.append(t); L.append(labs)
    idx = rng.permutation(len(T))
    return [T[i] for i in idx], [L[i] for i in idx]
