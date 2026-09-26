"""[WORKING] Synthetic capability-model factory.

Builds and trains tiny seeded transformers whose TRAINING DATA mix gives
them different genuine capability profiles (math-heavy, code-heavy, ...).
All models share one tokenizer + architecture -> honestly merge-compatible.
Scores on held-out suites are real measurements, not injected.

Weights are stored OUTSIDE the source package (fusionlab_data/), per spec §19.
Backend auto-selection: torch fast path when available, numpy reference
otherwise (identical interface and interchangeable weights).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .backend_numpy import MicroTransformer
from .base import Tokenizer
from .corpora import DOMAINS, gen_suite, gen_train

# capability mix per profile (sums to 1.0)
PROFILES: Dict[str, Dict[str, float]] = {
    "math-wiz":   {"math": .70, "code": .05, "reason": .05, "lang": .05, "know": .05, "multi": .05, "agent": .05},
    "code-smith": {"code": .70, "math": .05, "reason": .05, "lang": .05, "know": .05, "multi": .05, "agent": .05},
    "logic-owl":  {"reason": .70, "math": .05, "code": .05, "lang": .05, "know": .05, "multi": .05, "agent": .05},
    "polyglot":   {"multi": .55, "lang": .20, "know": .10, "math": .05, "code": .03, "reason": .05, "agent": .02},
    "generalist": {"math": .17, "code": .17, "reason": .15, "lang": .15, "know": .13, "multi": .12, "agent": .11},
    "weak-base":  {"math": .07, "code": .07, "reason": .07, "lang": .07, "know": .07, "multi": .07, "agent": .07,
                   "_scale": 0.25},   # trained on much less data overall
    # M3: math-free target (same everything, zero math training data)
    "nomath-base": {"math": 0.0, "code": .20, "reason": .18, "lang": .18,
                    "know": .16, "multi": .16, "agent": .12},
}


def data_dir() -> str:
    d = os.environ.get("FUSIONLAB_DATA",
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    os.pardir, "fusionlab_data"))
    d = os.path.abspath(d)
    os.makedirs(os.path.join(d, "models"), exist_ok=True)
    return d


def master_tokenizer() -> Tokenizer:
    dd = data_dir()
    cache = os.path.join(dd, "tokenizer.json")
    if os.path.exists(cache):
        return Tokenizer.from_dict(json.load(open(cache)))
    from .corpora import build_master_tokenizer
    texts = build_master_tokenizer()
    tok = Tokenizer.learn(texts, extra=["used", "the", "tool", "in", "is", "a", "to"])
    json.dump(tok.to_dict(), open(cache, "w"))
    return tok


def build_profile_corpus(profile: str, n_seqs: int, seed: int = 1234) -> List[str]:
    rng = np.random.default_rng(seed)
    mix = dict(PROFILES[profile])
    scale = float(mix.pop("_scale", 1.0))
    texts: List[str] = []
    for dom in DOMAINS:
        k = int(round(mix.get(dom, 0.0) * n_seqs * scale))
        if k > 0:
            texts += gen_train(dom, k, rng)
    rng.shuffle(texts)
    return texts


def make_micro(name: str, d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
               ctx_len: int = 32, vocab_size: int = 512, tokenizer: Optional[Tokenizer] = None,
               seed: int = 0):
    """Auto-select the fastest available backend (torch > numpy)."""
    try:
        from .backend_torch import TORCH_OK, TorchMicroTransformer
        if TORCH_OK:
            return TorchMicroTransformer(name, d_model=d_model, n_layers=n_layers,
                                         n_heads=n_heads, ctx_len=ctx_len,
                                         vocab_size=vocab_size, tokenizer=tokenizer, seed=seed)
    except Exception:
        pass
    return MicroTransformer(name, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                            ctx_len=ctx_len, vocab_size=vocab_size, tokenizer=tokenizer, seed=seed)


def train_micro(name: str, profile: str, n_seqs: int = 2600, epochs: int = 8,
                d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                ctx_len: int = 32, lr: float = 4e-3, batch: int = 64,
                seed: int = 7, log=None, tok: Optional[Tokenizer] = None,
                texts: Optional[List[str]] = None):
    """Train one micro model on its profile corpus. Deterministic given seeds."""
    tok = tok or master_tokenizer()
    texts = texts if texts is not None else build_profile_corpus(profile, n_seqs, seed=seed)
    enc = []
    for t in texts:
        ids = tok.encode(t, max_len=ctx_len)
        if len(ids) >= 4:
            enc.append(ids)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(enc))
    n_val = max(8, len(enc) // 12)
    val = [enc[i] for i in order[:n_val]]
    trn = [enc[i] for i in order[n_val:]]

    m = make_micro(name, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                   ctx_len=ctx_len, vocab_size=tok.vocab_size, tokenizer=tok, seed=seed + 2)
    hist = []
    t0 = time.time()
    opt_state: dict = {}
    is_torch = m.backend == "torch-synthetic"
    for ep in range(epochs):
        ep_rng = np.random.default_rng(seed + 100 + ep)
        ep_order = ep_rng.permutation(len(trn))
        trn_sorted = sorted(ep_order, key=lambda i: len(trn[i]))  # length-bucketed
        losses = []
        ep_lr = lr * (1.0 - 0.6 * ep / max(1, epochs - 1))
        for bs in range(0, len(trn_sorted), batch):
            idx = trn_sorted[bs:bs + batch]
            seqs = [trn[i] for i in idx]
            L = max(len(s) for s in seqs)
            arr = np.zeros((len(seqs), L), dtype=np.int64)  # 0 = <pad> (masked)
            for r, s in enumerate(seqs):
                arr[r, :len(s)] = s
            if is_torch:
                loss = m.train_step(arr, None, ep_lr, opt_state)
            else:
                loss, G = m.loss_and_grads(arr)
                m.adam_step(G, lr=ep_lr)
            losses.append(loss)
        vl = quick_val_loss(m, val)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": vl})
        if log and (ep % 5 == 0 or ep == epochs - 1):
            log(f"  [{name}] epoch {ep+1}/{epochs} train={np.mean(losses):.3f} val={vl:.3f} ({time.time()-t0:.0f}s)")
    m.train_history = hist
    m.profile = profile
    return m


def finetune_from(base, texts: List[str], epochs: int = 35, lr: float = 5e-3,
                  batch: int = 128, seed: int = 3, log=None):
    """Same-init fine-tune: train `base` (weights already loaded) on `texts`,
    keeping determinism. Returns the same model object (M3 transfer lineage)."""
    import numpy as _np
    enc = []
    for t in texts:
        ids = base.tokenizer.encode(t, max_len=base.spec.ctx_len)
        if len(ids) >= 4:
            enc.append(ids)
    rng = _np.random.default_rng(seed)
    opt: dict = {}
    is_torch = base.backend == "torch-synthetic"
    for ep in range(epochs):
        order = rng.permutation(len(enc))
        losses = []
        lr_ep = lr * (1.0 - 0.5 * ep / max(1, epochs - 1))
        for bs in range(0, len(order), batch):
            seqs = [enc[i] for i in order[bs:bs + batch]]
            L = max(len(s) for s in seqs)
            arr = _np.zeros((len(seqs), L), _np.int64)
            for r, s in enumerate(seqs):
                arr[r, :len(s)] = s
            if is_torch:
                losses.append(base.train_step(arr, None, lr_ep, opt))
            else:
                loss, G = base.loss_and_grads(arr)
                base.adam_step(G, lr=lr_ep)
                losses.append(loss)
        if log and (ep % 10 == 0 or ep == epochs - 1):
            log(f"    ft epoch {ep+1}/{epochs} loss={_np.mean(losses):.3f}")
    return base


def quick_val_loss(m, val_seqs: List[np.ndarray]) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(val_seqs), 32):
        seqs = val_seqs[i:i + 32]
        L = max(len(s) for s in seqs)
        arr = np.zeros((len(seqs), L), dtype=np.int64)
        for r, s in enumerate(seqs):
            arr[r, :len(s)] = s
        logits, _ = m.forward(arr)
        lg = logits[:, :-1, :]
        sh = arr[:, 1:]
        z = lg.reshape(-1, lg.shape[-1])
        z = z - z.max(-1, keepdims=True)
        P = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        keep = sh.reshape(-1) != 0
        tgt = np.zeros_like(P); tgt[np.arange(len(P)), sh.reshape(-1)] = 1.0
        tot += -float((tgt[keep] * np.log(P[keep] + 1e-9)).sum(-1).sum())
        n += int(keep.sum())
    return tot / max(1, n)


def save_model(m):
    dd = data_dir()
    np.savez_compressed(os.path.join(dd, "models", m.name + ".npz"), **m.state_dict())
    meta = {"name": m.name, "profile": getattr(m, "profile", "unknown"),
            "spec": m.spec.to_dict(), "tokenizer": m.tokenizer.to_dict(),
            "backend": m.backend,
            "train_history": getattr(m, "train_history", [])}
    json.dump(meta, open(os.path.join(dd, "models", m.name + ".json"), "w"))


def load_model(name: str):
    dd = data_dir()
    meta = json.load(open(os.path.join(dd, "models", name + ".json")))
    tok = Tokenizer.from_dict(meta["tokenizer"])
    s = meta["spec"]
    m = make_micro(name, d_model=s["hidden"], n_layers=s["layers"], n_heads=s["heads"],
                   ctx_len=s["ctx_len"], vocab_size=s["vocab"], tokenizer=tok)
    m.load_state_dict(dict(np.load(os.path.join(dd, "models", name + ".npz"))))
    m.train_history = meta.get("train_history", [])
    m.profile = meta.get("profile", "unknown")
    return m


def saved_names() -> List[str]:
    dd = data_dir()
    return sorted(f[:-4] for f in os.listdir(os.path.join(dd, "models")) if f.endswith(".npz"))
