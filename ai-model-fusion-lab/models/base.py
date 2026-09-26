"""Model handle interface shared by all backends.

A ModelHandle is the ONLY way the rest of the lab touches a model.
Two backends implement it:

  * models.backend_numpy.MicroTransformer  — seeded synthetic tiny LM [WORKING]
  * models.backend_hf.HFHandle             — real HuggingFace causal LMs [REQUIRES EXTERNAL MODEL]

Everything downstream (autopsy, benchmarks, ablation, merging) works through
this interface so large external models can replace micro models without
changing the core application (spec milestone 3).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

TensorDict = Dict[str, np.ndarray]


@dataclass
class ArchSpec:
    """Architectural facts used by the compatibility analyzer."""
    arch: str                    # e.g. "micro-gpt", "gpt2", "llama"
    layers: int
    hidden: int
    heads: int
    mlp_hidden: int
    vocab: int
    ctx_len: int
    positional: str              # "learned", "rope", ...
    norm: str                    # "layernorm", "rmsnorm", ...
    attention: str               # "mha", "gqa", "mha+rope", ...
    mlp_act: str                 # "gelu", "swiglu", ...
    tie_embeddings: bool
    modality: str = "text"
    moe: bool = False
    quant: Optional[str] = None  # None | "int8" | "4bit" ...
    license: str = "unknown"
    source: str = "synthetic"    # "synthetic" | "hf" | "local-dir"
    hf_id: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Tokenizer:
    """Greedy longest-match word tokenizer; digits always split per-digit
    so arithmetic is learnable algorithmically rather than by memorization."""

    def __init__(self, tokens: Sequence[str]):
        self.itos: List[str] = ["<pad>", "<unk>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> List[str]:
        out = []
        for chunk in text.split():
            i = 0
            while i < len(chunk):
                if chunk[i].isdigit():
                    out.append(chunk[i]); i += 1
                    continue
                for j in range(len(chunk), i, -1):
                    if chunk[i:j] in self.stoi:
                        out.append(chunk[i:j]); i = j
                        break
                else:
                    out.append(chunk[i]); i += 1
        return out

    @classmethod
    def learn(cls, texts: Sequence[str], extra: Sequence[str] = ()) -> "Tokenizer":
        vocab = set(extra)
        for t in texts:
            for chunk in t.split():
                if chunk.isdigit():
                    vocab.update(chunk)
                else:
                    vocab.add(chunk)
        vocab.update(list("0123456789"))
        return cls(sorted(vocab))

    def encode(self, text: str, max_len: Optional[int] = None) -> np.ndarray:
        ids = [self.stoi.get(t, 1) for t in self.tokenize(text)]
        if max_len is not None:
            ids = ids[:max_len]
        return np.array(ids, dtype=np.int64)

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos[int(i)] for i in ids if int(i) > 0)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        tok = cls.__new__(cls)
        tok.itos = list(d["itos"])
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        return tok


class ModelHandle:
    """Common interface. All tensor math uses numpy float32 state dicts."""

    backend = "abstract"

    def __init__(self, name: str, spec: ArchSpec, tokenizer: Tokenizer):
        self.name = name
        self.spec = spec
        self.tokenizer = tokenizer
        # ablation controls (persistent, set by the ablation engine)
        self.skip_layers: List[int] = []
        self.skip_modules: Dict[int, List[str]] = {}
        self.head_mask: Optional[np.ndarray] = None  # [L, H] with 0/1
        self.mlp_channel_mask: Optional[Dict[int, np.ndarray]] = None  # L -> [F] 0/1
        self.patch: Dict = {}                        # activation patching hooks

    # --- to implement -------------------------------------------------
    def forward(self, ids: np.ndarray, collect: bool = False):
        raise NotImplementedError

    def state_dict(self) -> TensorDict:
        raise NotImplementedError

    def load_state_dict(self, sd: TensorDict):
        raise NotImplementedError

    def trainable(self) -> bool:
        return False

    # --- shared utilities ---------------------------------------------
    def param_count(self) -> int:
        return int(sum(v.size for v in self.state_dict().values()))

    def logprob_option(self, prompt_ids: np.ndarray, option_ids: np.ndarray) -> float:
        """Mean teacher-forced logprob of option_ids continuing prompt_ids."""
        if len(option_ids) == 0:
            return 0.0
        ids = np.concatenate([prompt_ids, option_ids])[None, :]
        if ids.shape[1] > self.spec.ctx_len:
            keep = self.spec.ctx_len
            ids = ids[:, -keep:]
        logits, _ = self.forward(ids)
        lp = 0.0
        n = len(option_ids)
        for i in range(n):
            pos = ids.shape[1] - n + i - 1          # predicts option token i
            row = logits[0, pos]
            row = row - row.max()
            logz = np.log(np.exp(row).sum())
            lp += float(row[option_ids[i]] - logz)
        return lp / n

    def generate(self, ids: np.ndarray, max_new_tokens: int = 8,
                 temperature: float = 0.0) -> np.ndarray:
        cur = list(ids)
        for _ in range(max_new_tokens):
            window = np.array(cur[-self.spec.ctx_len:], dtype=np.int64)[None, :]
            logits, _ = self.forward(window)
            nxt = int(logits[0, -1].argmax()) if temperature <= 0 else int(
                np.random.default_rng(0).choice(len(logits[0, -1]),
                                                p=_softmax(logits[0, -1] / temperature)))
            cur.append(nxt)
        return np.array(cur, dtype=np.int64)

    def fingerprint(self) -> str:
        sd = self.state_dict()          # materialize ONCE (M2: gpt2-sized models OOM otherwise)
        h = hashlib.sha256()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(np.ascontiguousarray(sd[k]).tobytes())
        return h.hexdigest()[:16]

    def arch_dict(self) -> dict:
        return self.spec.to_dict()


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def state_dict_hash(sd: TensorDict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k]).tobytes())
    return h.hexdigest()[:16]


def config_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
