"""[WORKING] Pure-numpy micro GPT-style transformer.

Why numpy: the whole lab must run on a CPU-only sandbox with zero mandatory
heavy deps, fully seeded and reproducible. The same interface is implemented
for real HuggingFace models in backend_hf.py, so micro models can be replaced
by large external models without touching the pipeline.

Implements: forward with ablation hooks (layer skip, module skip, head mask,
activation patching), explicit backward for training, Adam optimizer.
Backward correctness is verified by gradient checks in tests/test_gradcheck.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import ArchSpec, ModelHandle, TensorDict, Tokenizer

_GELU_C = float(np.sqrt(2.0 / np.pi))


def gelu(z: np.ndarray) -> np.ndarray:
    u = _GELU_C * (z + 0.044715 * z ** 3)
    return 0.5 * z * (1.0 + np.tanh(u))


def gelu_grad(z: np.ndarray) -> np.ndarray:
    u = _GELU_C * (z + 0.044715 * z ** 3)
    t = np.tanh(u)
    du = _GELU_C * (1.0 + 3 * 0.044715 * z ** 2)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t ** 2) * du


def layernorm_fwd(x: np.ndarray, g: np.ndarray, b: np.ndarray):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    std = np.sqrt(var + 1e-5)
    xhat = (x - mu) / std
    return g * xhat + b, (xhat, std)


def layernorm_bwd(dh: np.ndarray, xhat: np.ndarray, std: np.ndarray,
                  g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dg = (dh * xhat).sum(axis=(0, 1))
    db = dh.sum(axis=(0, 1))
    dxhat = dh * g
    m1 = dxhat.mean(-1, keepdims=True)
    m2 = (dxhat * xhat).mean(-1, keepdims=True)
    dx = (dxhat - m1 - xhat * m2) / std
    return dx, dg, db


class MicroTransformer(ModelHandle):
    """Tiny decoder-only transformer, word-level, learned positions, tied output."""

    backend = "numpy-synthetic"

    def __init__(self, name: str, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, ctx_len: int = 32, vocab_size: int = 512,
                 tokenizer: Optional[Tokenizer] = None, seed: int = 0):
        tok = tokenizer or Tokenizer(["<extra>"])
        # ensure tokenizer vocab matches requested size (pad with unused tokens)
        if tok.vocab_size < vocab_size:
            extra = [f"<r{i}>" for i in range(vocab_size - tok.vocab_size)]
            tok = Tokenizer(tok.itos[2:] + extra)
        spec = ArchSpec(
            arch="micro-gpt", layers=n_layers, hidden=d_model, heads=n_heads,
            mlp_hidden=4 * d_model, vocab=tok.vocab_size, ctx_len=ctx_len,
            positional="learned", norm="layernorm", attention="mha",
            mlp_act="gelu", tie_embeddings=True, license="Apache-2.0",
            source="synthetic")
        super().__init__(name, spec, tok)
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()
        self.grads: Dict[str, np.ndarray] = {}
        self.opt_state: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.t = 0

    # ------------------------------------------------------------------
    def _init_params(self) -> TensorDict:
        d, V, L, H = self.spec.hidden, self.spec.vocab, self.spec.layers, self.spec.heads
        T, F = self.spec.ctx_len, self.spec.mlp_hidden
        p: TensorDict = {}
        std = 0.02
        p["We"] = (self.rng.normal(0, std, (V, d))).astype(np.float32)
        p["Wp"] = (self.rng.normal(0, std, (T, d))).astype(np.float32)
        for l in range(L):
            pre = f"L{l}."
            p[pre + "ln1.g"] = np.ones(d, np.float32)
            p[pre + "ln1.b"] = np.zeros(d, np.float32)
            p[pre + "attn.qkv.W"] = (self.rng.normal(0, std, (d, 3 * d))).astype(np.float32)
            p[pre + "attn.qkv.b"] = np.zeros(3 * d, np.float32)
            p[pre + "attn.o.W"] = (self.rng.normal(0, std, (d, d))).astype(np.float32)
            p[pre + "attn.o.b"] = np.zeros(d, np.float32)
            p[pre + "ln2.g"] = np.ones(d, np.float32)
            p[pre + "ln2.b"] = np.zeros(d, np.float32)
            p[pre + "mlp.1.W"] = (self.rng.normal(0, std, (d, F))).astype(np.float32)
            p[pre + "mlp.1.b"] = np.zeros(F, np.float32)
            p[pre + "mlp.2.W"] = (self.rng.normal(0, std, (F, d))).astype(np.float32)
            p[pre + "mlp.2.b"] = np.zeros(d, np.float32)
        p["lnf.g"] = np.ones(d, np.float32)
        p["lnf.b"] = np.zeros(d, np.float32)
        return p

    # ------------------------------------------------------------------
    def forward(self, ids: np.ndarray, collect: bool = False):
        p, S = self.params, self.spec
        B, T = ids.shape
        d, H = S.hidden, S.heads
        e = d // H
        x = p["We"][ids] + p["Wp"][:T][None, :, :]                     # [B,T,d]
        cache: Dict = {"resid_in": {}, "attn_out": {}, "mlp_out": {},
                       "attn_probs": {}, "hiddens": {}}
        if collect:
            cache["embed"] = x.copy()

        hm = self.head_mask
        for l in range(S.layers):
            if collect:
                cache["resid_in"][l] = x.copy()
            if ("resid", l) in self.patch:
                x = np.broadcast_to(self.patch[("resid", l)], x.shape).astype(np.float32).copy()
                if collect:
                    cache["resid_in"][l] = x.copy()
            pre = f"L{l}."
            ablate_block = l in self.skip_layers
            skip_attn = ablate_block or "attn" in self.skip_modules.get(l, [])
            skip_mlp = ablate_block or "mlp" in self.skip_modules.get(l, [])

            h, _ = layernorm_fwd(x, p[pre + "ln1.g"], p[pre + "ln1.b"])
            qkv = h @ p[pre + "attn.qkv.W"] + p[pre + "attn.qkv.b"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, H, e).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, e).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, e).transpose(0, 2, 1, 3)
            att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(e)             # [B,H,T,T]
            mask = np.triu(np.ones((T, T), bool), 1)
            att = np.where(mask[None, None], -1e9, att)
            probs = self._softmax(att)
            if hm is not None:
                probs = probs * hm[l][None, :, None, None]
            if ("attn_probs", l) in self.patch:
                probs = np.broadcast_to(self.patch[("attn_probs", l)], probs.shape).astype(np.float32)
            ctx = probs @ v                                            # [B,H,T,e]
            ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, d)
            ao = ctx @ p[pre + "attn.o.W"] + p[pre + "attn.o.b"]
            if skip_attn:
                ao = np.zeros_like(ao)
            if ("attn_out", l) in self.patch:
                ao = np.broadcast_to(self.patch[("attn_out", l)], ao.shape).astype(np.float32).copy()
            if collect:
                cache["attn_out"][l] = ao.copy()
                cache["attn_probs"][l] = probs
            x = x + ao

            h2, _ = layernorm_fwd(x, p[pre + "ln2.g"], p[pre + "ln2.b"])
            z = h2 @ p[pre + "mlp.1.W"] + p[pre + "mlp.1.b"]
            act = gelu(z)
            cm = (self.mlp_channel_mask or {}).get(l)
            if cm is not None:
                act = act * cm[None, None, :]
            m = act @ p[pre + "mlp.2.W"] + p[pre + "mlp.2.b"]
            if skip_mlp:
                m = np.zeros_like(m)
            if ("mlp_out", l) in self.patch:
                m = np.broadcast_to(self.patch[("mlp_out", l)], m.shape).astype(np.float32).copy()
            if collect:
                cache["mlp_out"][l] = m.copy()
            x = x + m
            if collect:
                cache["hiddens"][l] = x.copy()

        hf, _ = layernorm_fwd(x, p["lnf.g"], p["lnf.b"])
        logits = hf @ p["We"].T                                        # tied output
        if collect:
            cache["final_hidden"] = hf
            return logits, cache
        return logits, None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)

    # ------------------------------------------------------------------
    def backward(self, ids: np.ndarray, dlogits: np.ndarray) -> TensorDict:
        """dlogits: [B,T,V] gradient of loss wrt logits. Returns grads dict."""
        p, S = self.params, self.spec
        B, T = ids.shape
        d, H = S.hidden, S.heads
        e = d // H
        G: TensorDict = {k: np.zeros_like(v) for k, v in p.items()}

        hf = None
        # recompute forward intermediates (kept simple; tiny models)
        x = p["We"][ids] + p["Wp"][:T][None, :, :]
        resid_pres, resid_mids = [], []
        h1s, h2s, zs, probs_s, ctxs, qs, ks, vs = [], [], [], [], [], [], [], []
        for l in range(S.layers):
            resid_pres.append(x)
            pre = f"L{l}."
            h1, _ = layernorm_fwd(x, p[pre + "ln1.g"], p[pre + "ln1.b"])
            qkv = h1 @ p[pre + "attn.qkv.W"] + p[pre + "attn.qkv.b"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, H, e).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, e).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, e).transpose(0, 2, 1, 3)
            qs.append(q); ks.append(k); vs.append(v)
            att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(e)
            mask = np.triu(np.ones((T, T), bool), 1)
            att = np.where(mask[None, None], -1e9, att)
            probs = self._softmax(att)
            hm = self.head_mask
            if hm is not None:
                probs = probs * hm[l][None, :, None, None]
            ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            ao = ctx @ p[pre + "attn.o.W"] + p[pre + "attn.o.b"]
            if l in self.skip_layers or "attn" in self.skip_modules.get(l, []):
                ao = np.zeros_like(ao)
            x = x + ao
            resid_mids.append(x)
            h2, _ = layernorm_fwd(x, p[pre + "ln2.g"], p[pre + "ln2.b"])
            z = h2 @ p[pre + "mlp.1.W"] + p[pre + "mlp.1.b"]
            act = gelu(z)
            cm = (self.mlp_channel_mask or {}).get(l)
            if cm is not None:
                act = act * cm[None, None, :]
            m = act @ p[pre + "mlp.2.W"] + p[pre + "mlp.2.b"]
            if l in self.skip_layers or "mlp" in self.skip_modules.get(l, []):
                m = np.zeros_like(m)
            x = x + m
            h1s.append(h1); h2s.append(h2); zs.append(z); probs_s.append(probs); ctxs.append(ctx)
        x_final = x
        hf, _ = layernorm_fwd(x, p["lnf.g"], p["lnf.b"])

        # tied output head
        dhf = dlogits.reshape(-1, S.vocab) @ p["We"]                     # [B*T,d]
        G["We"] += dlogits.reshape(-1, S.vocab).T @ hf.reshape(-1, d)
        dhf = dhf.reshape(B, T, d)
        dx, dg, db = layernorm_bwd(dhf, *layernorm_fwd(x_final, p["lnf.g"], p["lnf.b"])[1], p["lnf.g"])
        G["lnf.g"] += dg; G["lnf.b"] += db

        for l in reversed(range(S.layers)):
            pre = f"L{l}."
            resid_pre = resid_pres[l]        # residual entering layer l
            resid_mid = resid_mids[l]        # residual entering layer-l MLP norm
            skip_attn = l in self.skip_layers or "attn" in self.skip_modules.get(l, [])
            skip_mlp = l in self.skip_layers or "mlp" in self.skip_modules.get(l, [])

            # --- MLP branch (input: resid_mid) ---
            dx_mid = dx
            if not skip_mlp:
                G[pre + "mlp.2.b"] += dx.sum((0, 1))
                G[pre + "mlp.2.W"] += gelu(zs[l]).reshape(-1, S.mlp_hidden).T @ dx.reshape(-1, d)
                da = dx @ p[pre + "mlp.2.W"].T
                dz = da * gelu_grad(zs[l])
                G[pre + "mlp.1.b"] += dz.sum((0, 1))
                G[pre + "mlp.1.W"] += h2s[l].reshape(-1, d).T @ dz.reshape(-1, S.mlp_hidden)
                dh2 = dz @ p[pre + "mlp.1.W"].T
                dx2, dg2, db2 = layernorm_bwd(dh2, *layernorm_fwd(resid_mid, p[pre + "ln2.g"], p[pre + "ln2.b"])[1], p[pre + "ln2.g"])
                G[pre + "ln2.g"] += dg2; G[pre + "ln2.b"] += db2
                dx_mid = dx + dx2

            # --- attention branch (input: resid_pre) ---
            dx_pre = dx_mid
            if not skip_attn:
                dao = dx_mid
                G[pre + "attn.o.b"] += dao.sum((0, 1))
                G[pre + "attn.o.W"] += ctxs[l].reshape(-1, d).T @ dao.reshape(-1, d)
                dctx = (dao @ p[pre + "attn.o.W"].T).reshape(B, T, H, e).transpose(0, 2, 1, 3)
                probs = probs_s[l]
                dp = dctx @ vs[l].transpose(0, 1, 3, 2)                  # [B,H,T,T]
                hm = self.head_mask
                if hm is not None:
                    dp = dp * hm[l][None, :, None, None]
                ds = probs * (dp - (dp * probs).sum(-1, keepdims=True))
                ds = ds / np.sqrt(e)
                dq = ds @ ks[l]                                          # [B,H,T,e]
                dk = ds.transpose(0, 1, 3, 2) @ qs[l]
                dv = probs.transpose(0, 1, 3, 2) @ dctx
                dq = dq.transpose(0, 2, 1, 3).reshape(B, T, d)
                dk = dk.transpose(0, 2, 1, 3).reshape(B, T, d)
                dv = dv.transpose(0, 2, 1, 3).reshape(B, T, d)
                dqkv = np.concatenate([dq, dk, dv], axis=-1)
                G[pre + "attn.qkv.b"] += dqkv.sum((0, 1))
                G[pre + "attn.qkv.W"] += h1s[l].reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
                dh1 = dqkv @ p[pre + "attn.qkv.W"].T
                dx1, dg1, db1 = layernorm_bwd(dh1, *layernorm_fwd(resid_pre, p[pre + "ln1.g"], p[pre + "ln1.b"])[1], p[pre + "ln1.g"])
                G[pre + "ln1.g"] += dg1; G[pre + "ln1.b"] += db1
                dx_pre = dx_mid + dx1
            dx = dx_pre

        # embeddings
        np.add.at(G["We"], ids, dx)
        G["Wp"][:T] += dx.sum(0)
        return {k: v for k, v in G.items()}

    # ------------------------------------------------------------------
    def state_dict(self) -> TensorDict:
        return {k: v for k, v in self.params.items()}

    def load_state_dict(self, sd: TensorDict):
        for k in self.params:
            self.params[k] = np.asarray(sd[k], dtype=np.float32).reshape(self.params[k].shape)
        self.opt_state.clear()
        self.t = 0

    def trainable(self) -> bool:
        return True

    # --- training ------------------------------------------------------
    def loss_and_grads(self, ids: np.ndarray, target_probs: Optional[np.ndarray] = None):
        """Next-token CE (target_probs=None) or soft-target KL for distillation.
        Positions whose target is <pad>(0) are masked out of loss/grads."""
        logits, _ = self.forward(ids)
        T = ids.shape[1]
        logits = logits[:, :-1, :]
        shifted = ids[:, 1:]
        P = self._softmax(logits.reshape(-1, self.spec.vocab))
        if target_probs is None:
            tgt = np.zeros_like(P)
            tgt[np.arange(len(P)), shifted.reshape(-1)] = 1.0
        else:
            tgt = target_probs.reshape(P.shape)
        mask = (shifted.reshape(-1) != 0)
        n_masked = int(mask.sum())
        loss = -float((tgt[mask] * np.log(P[mask] + 1e-9)).sum(-1).mean()) if n_masked else 0.0
        dlogits = np.zeros_like(P)
        if n_masked:
            dlogits[mask] = ((P - tgt)[mask] / n_masked).astype(np.float32)
        # full gradient wrt original logits (last position unused)
        full = np.zeros((ids.shape[0], T, self.spec.vocab), np.float32)
        full[:, :-1, :] = dlogits.reshape(ids.shape[0], T - 1, -1)
        G = self.backward(ids, full)
        return loss, G

    def adam_step(self, grads: TensorDict, lr: float = 3e-3, b1: float = 0.9,
                  b2: float = 0.999, eps: float = 1e-8, clip: float = 1.0):
        total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = clip / total if total > clip else 1.0
        self.t += 1
        for k, g in grads.items():
            g = g * scale
            if k not in self.opt_state:
                self.opt_state[k] = (np.zeros_like(g), np.zeros_like(g))
            m, v = self.opt_state[k]
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            mhat = m / (1 - b1 ** self.t)
            vhat = v / (1 - b2 ** self.t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    def param_groups(self) -> Dict:
        out = {"embedding": {"We": self.params["We"], "Wp": self.params["Wp"]},
               "final_norm": {k: self.params[k] for k in ("lnf.g", "lnf.b")},
               "layers": {}}
        for l in range(self.spec.layers):
            pre = f"L{l}."
            out["layers"][l] = {
                "attn": {k.split(".", 2)[-1]: v for k, v in self.params.items() if k.startswith(pre + "attn")},
                "mlp": {k.split(".", 2)[-1]: v for k, v in self.params.items() if k.startswith(pre + "mlp")},
                "norm1": {k.split(".", 2)[-1]: v for k, v in self.params.items() if k.startswith(pre + "ln1")},
                "norm2": {k.split(".", 2)[-1]: v for k, v in self.params.items() if k.startswith(pre + "ln2")},
            }
        return out


def ctxs_perm(*args):  # pragma: no cover
    raise NotImplementedError
