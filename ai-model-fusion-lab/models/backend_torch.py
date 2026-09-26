"""[WORKING] Torch fast-path backend for the micro transformer.

Same interface, same state-dict key names and weight layout as the numpy
reference backend (backend_numpy.MicroTransformer), so:
  * weights are interchangeable (merge algorithms work across backends)
  * cross-backend outputs are validated to match (tests/test_backends.py)

Used automatically when torch is available; numpy backend remains the
zero-dependency fallback and the numerical reference implementation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import ArchSpec, ModelHandle, TensorDict, Tokenizer

try:
    import torch
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:  # pragma: no cover
    TORCH_OK = False


class TorchMicroTransformer(ModelHandle):
    backend = "torch-synthetic"

    def __init__(self, name: str, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, ctx_len: int = 32, vocab_size: int = 512,
                 tokenizer: Optional[Tokenizer] = None, seed: int = 0):
        if not TORCH_OK:
            raise RuntimeError("torch not available; use backend_numpy.MicroTransformer")
        tok = tokenizer or Tokenizer(["<extra>"])
        if tok.vocab_size < vocab_size:
            extra = [f"<r{i}>" for i in range(vocab_size - tok.vocab_size)]
            tok = Tokenizer(tok.itos[2:] + extra)
        spec = ArchSpec(arch="micro-gpt", layers=n_layers, hidden=d_model, heads=n_heads,
                        mlp_hidden=4 * d_model, vocab=tok.vocab_size, ctx_len=ctx_len,
                        positional="learned", norm="layernorm", attention="mha",
                        mlp_act="gelu", tie_embeddings=True, license="Apache-2.0",
                        source="synthetic")
        super().__init__(name, spec, tok)
        g = torch.Generator().manual_seed(seed)
        self.torch_seed = seed
        d, V, T, F = d_model, tok.vocab_size, ctx_len, 4 * d_model
        self.t: Dict[str, "torch.Tensor"] = {}
        init = lambda *s: torch.randn(*s, generator=g) * 0.02  # noqa: E731
        self.t["We"] = init(V, d)
        self.t["Wp"] = init(T, d)
        for l in range(n_layers):
            pre = f"L{l}."
            self.t[pre + "ln1.g"] = torch.ones(d); self.t[pre + "ln1.b"] = torch.zeros(d)
            self.t[pre + "attn.qkv.W"] = init(d, 3 * d); self.t[pre + "attn.qkv.b"] = torch.zeros(3 * d)
            self.t[pre + "attn.o.W"] = init(d, d); self.t[pre + "attn.o.b"] = torch.zeros(d)
            self.t[pre + "ln2.g"] = torch.ones(d); self.t[pre + "ln2.b"] = torch.zeros(d)
            self.t[pre + "mlp.1.W"] = init(d, F); self.t[pre + "mlp.1.b"] = torch.zeros(F)
            self.t[pre + "mlp.2.W"] = init(F, d); self.t[pre + "mlp.2.b"] = torch.zeros(d)
        self.t["lnf.g"] = torch.ones(d); self.t["lnf.b"] = torch.zeros(d)
        for k in self.t:
            self.t[k].requires_grad_(True)
        self._ln = torch.nn.LayerNorm(d_model, elementwise_affine=False)

    # ------------------------------------------------------------------
    def forward(self, ids: np.ndarray, collect: bool = False):
        x = torch.tensor(ids)
        if x.dim() == 1:
            x = x[None, :]
        p, S = self.t, self.spec
        B, T = x.shape
        d, H = S.hidden, S.heads
        e = d // H
        h = p["We"][x] + p["Wp"][:T][None]
        cache: Dict = {"resid_in": {}, "attn_out": {}, "mlp_out": {}, "attn_probs": {}, "hiddens": {}}
        if collect:
            cache["embed"] = h.detach().numpy().copy()
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        hm = self.head_mask
        for l in range(S.layers):
            if collect:
                cache["resid_in"][l] = h.detach().numpy().copy()
            if ("resid", l) in self.patch:
                pt = torch.tensor(np.broadcast_to(self.patch[("resid", l)], (B, T, d)), dtype=torch.float32)
                h = pt
                if collect:
                    cache["resid_in"][l] = h.detach().numpy().copy()
            pre = f"L{l}."
            skip_attn = l in self.skip_layers or "attn" in self.skip_modules.get(l, [])
            skip_mlp = l in self.skip_layers or "mlp" in self.skip_modules.get(l, [])
            h1 = self._ln(h) * p[pre + "ln1.g"] + p[pre + "ln1.b"]
            qkv = h1 @ p[pre + "attn.qkv.W"] + p[pre + "attn.qkv.b"]
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, H, e).transpose(1, 2); k = k.view(B, T, H, e).transpose(1, 2)
            v = v.view(B, T, H, e).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(e)
            att = att.masked_fill(mask, -1e9)
            probs = att.softmax(-1)
            if hm is not None:
                probs = probs * torch.tensor(hm[l], dtype=torch.float32)[None, :, None, None]
            if ("attn_probs", l) in self.patch:
                probs = torch.tensor(np.broadcast_to(self.patch[("attn_probs", l)],
                                                     (B, H, T, T)), dtype=torch.float32)
            ctx = (probs @ v).transpose(1, 2).reshape(B, T, d)
            ao = ctx @ p[pre + "attn.o.W"] + p[pre + "attn.o.b"]
            if skip_attn:
                ao = torch.zeros_like(ao)
            if ("attn_out", l) in self.patch:
                ao = torch.tensor(np.broadcast_to(self.patch[("attn_out", l)], (B, T, d)), dtype=torch.float32)
            if collect:
                cache["attn_out"][l] = ao.detach().numpy().copy()
                cache["attn_probs"][l] = probs.detach().numpy()
            h = h + ao
            h2 = self._ln(h) * p[pre + "ln2.g"] + p[pre + "ln2.b"]
            z = h2 @ p[pre + "mlp.1.W"] + p[pre + "mlp.1.b"]
            act = F.gelu(z, approximate="tanh")
            cm = (self.mlp_channel_mask or {}).get(l)
            if cm is not None:
                act = act * torch.tensor(cm, dtype=torch.float32)[None, None, :]
            m = act @ p[pre + "mlp.2.W"] + p[pre + "mlp.2.b"]
            if skip_mlp:
                m = torch.zeros_like(m)
            if ("mlp_out", l) in self.patch:
                m = torch.tensor(np.broadcast_to(self.patch[("mlp_out", l)], (B, T, d)), dtype=torch.float32)
            if collect:
                cache["mlp_out"][l] = m.detach().numpy().copy()
            h = h + m
            if collect:
                cache["hiddens"][l] = h.detach().numpy().copy()
        hf = self._ln(h) * p["lnf.g"] + p["lnf.b"]
        logits = hf @ p["We"].T
        if collect:
            cache["final_hidden"] = hf.detach().numpy()
            return logits.detach().numpy(), cache
        return logits.detach().numpy(), None

    def forward_torch(self, ids: "torch.Tensor", collect: bool = False):
        """Differentiable twin of forward() returning torch logits."""
        p, S = self.t, self.spec
        B, T = ids.shape
        d, H = S.hidden, S.heads
        e = d // H
        h = p["We"][ids] + p["Wp"][:T][None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        hm = self.head_mask
        for l in range(S.layers):
            if ("resid", l) in self.patch:
                h = torch.tensor(np.broadcast_to(self.patch[("resid", l)], (B, T, d)), dtype=torch.float32)
            pre = f"L{l}."
            skip_attn = l in self.skip_layers or "attn" in self.skip_modules.get(l, [])
            skip_mlp = l in self.skip_layers or "mlp" in self.skip_modules.get(l, [])
            h1 = self._ln(h) * p[pre + "ln1.g"] + p[pre + "ln1.b"]
            qkv = h1 @ p[pre + "attn.qkv.W"] + p[pre + "attn.qkv.b"]
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, H, e).transpose(1, 2); k = k.view(B, T, H, e).transpose(1, 2)
            v = v.view(B, T, H, e).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(e)
            att = att.masked_fill(mask, -1e9)
            probs = att.softmax(-1)
            if hm is not None:
                probs = probs * torch.tensor(hm[l], dtype=torch.float32)[None, :, None, None]
            ctx = (probs @ v).transpose(1, 2).reshape(B, T, d)
            ao = ctx @ p[pre + "attn.o.W"] + p[pre + "attn.o.b"]
            if skip_attn:
                ao = torch.zeros_like(ao)
            h = h + ao
            h2 = self._ln(h) * p[pre + "ln2.g"] + p[pre + "ln2.b"]
            z = h2 @ p[pre + "mlp.1.W"] + p[pre + "mlp.1.b"]
            act = F.gelu(z, approximate="tanh")
            cm = (self.mlp_channel_mask or {}).get(l)
            if cm is not None:
                act = act * torch.tensor(cm, dtype=torch.float32)[None, None, :]
            m = act @ p[pre + "mlp.2.W"] + p[pre + "mlp.2.b"]
            if skip_mlp:
                m = torch.zeros_like(m)
            h = h + m
        hf = self._ln(h) * p["lnf.g"] + p["lnf.b"]
        return hf @ p["We"].T

    # ------------------------------------------------------------------
    def state_dict(self) -> TensorDict:
        return {k: v.detach().numpy().astype(np.float32) for k, v in self.t.items()}

    def load_state_dict(self, sd: TensorDict):
        for k in self.t:
            self.t[k] = torch.tensor(np.asarray(sd[k], dtype=np.float32))
            self.t[k].requires_grad_(True)
        self.opt = None

    def trainable(self) -> bool:
        return True

    def logprob_option(self, prompt_ids: np.ndarray, option_ids: np.ndarray) -> float:
        if len(option_ids) == 0:
            return 0.0
        ids = np.concatenate([prompt_ids, option_ids])[None, :]
        if ids.shape[1] > self.spec.ctx_len:
            ids = ids[:, -self.spec.ctx_len:]
        with torch.no_grad():
            logits = self.forward_torch(torch.tensor(ids))
        lp = 0.0
        n = len(option_ids)
        logp = torch.log_softmax(logits[0], dim=-1)
        for i in range(n):
            pos = ids.shape[1] - n + i - 1
            lp += float(logp[pos, option_ids[i]])
        return lp / n

    # --- training -------------------------------------------------------
    def loss_and_grads(self, ids: np.ndarray, target_probs: Optional[np.ndarray] = None):
        x = torch.tensor(ids)
        logits = self.forward_torch(x)
        T = ids.shape[1]
        lg = logits[:, :-1, :]
        sh = x[:, 1:]
        logP = torch.log_softmax(lg.reshape(-1, self.spec.vocab), -1)
        if target_probs is None:
            keep = (sh.reshape(-1) != 0)
            tgt_idx = sh.reshape(-1)
            loss_vec = -logP[torch.arange(len(tgt_idx)), tgt_idx.clamp(min=0)]
            loss = loss_vec[keep].mean() if keep.any() else torch.tensor(0.0)
        else:
            tgt = torch.tensor(target_probs, dtype=torch.float32).reshape(logP.shape)
            keep = (sh.reshape(-1) != 0)
            loss = -(tgt.exp() * logP).sum(-1)[keep].mean() if keep.any() else torch.tensor(0.0)
        return float(loss.detach()), None  # grads handled by torch autograd via train_step

    def train_step(self, ids: np.ndarray, target_probs: Optional[np.ndarray],
                   lr: float, opt_state: dict, clip: float = 1.0) -> float:
        if "opt" not in opt_state:
            opt_state["opt"] = torch.optim.Adam(list(self.t.values()), lr=lr)
            opt_state["lr"] = lr
        opt_state["opt"].param_groups[0]["lr"] = lr
        x = torch.tensor(ids)
        logits = self.forward_torch(x)
        lg = logits[:, :-1, :]
        sh = x[:, 1:]
        logP = torch.log_softmax(lg.reshape(-1, self.spec.vocab), -1)
        if target_probs is None:
            keep = (sh.reshape(-1) != 0)
            tgt_idx = sh.reshape(-1)
            loss_vec = -logP[torch.arange(len(tgt_idx)), tgt_idx.clamp(min=0)]
            loss = loss_vec[keep].mean()
        else:
            # target_probs: probabilities aligned to sh (shape [B, T-1, V] or [B*(T-1), V])
            tgt = torch.tensor(target_probs, dtype=torch.float32).reshape(logP.shape)
            keep = (sh.reshape(-1) != 0)
            loss = -(tgt * logP).sum(-1)[keep].mean()
        opt_state["opt"].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.t.values(), clip)
        opt_state["opt"].step()
        return float(loss.detach())

    def param_groups(self) -> Dict:
        out = {"embedding": {"We": self.state_dict()["We"], "Wp": self.state_dict()["Wp"]},
               "final_norm": {k: self.state_dict()[k] for k in ("lnf.g", "lnf.b")},
               "layers": {}}
        sd = self.state_dict()
        for l in range(self.spec.layers):
            pre = f"L{l}."
            out["layers"][l] = {
                "attn": {k[len(pre + "attn."):]: v for k, v in sd.items() if k.startswith(pre + "attn")},
                "mlp": {k[len(pre + "mlp."):]: v for k, v in sd.items() if k.startswith(pre + "mlp")},
                "norm1": {k[len(pre + "ln1."):]: v for k, v in sd.items() if k.startswith(pre + "ln1")},
                "norm2": {k[len(pre + "ln2."):]: v for k, v in sd.items() if k.startswith(pre + "ln2")},
            }
        return out
