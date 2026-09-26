"""[REQUIRES EXTERNAL MODEL] HuggingFace backend implementing ModelHandle.

Loads real open-weight causal LMs (e.g. gpt2, pythia-70m) via transformers.
Supports ablation hooks (layer skip, module skip, head mask) so the whole
autopsy/ablation/localization pipeline runs identically on real models.
Falls back with a clear error if transformers/torch/network unavailable —
the lab then uses the synthetic backend (spec §18 MODE B).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from .base import ArchSpec, ModelHandle, TensorDict, Tokenizer

try:
    import torch
    import transformers
    TORCH_OK = True
except Exception:
    TORCH_OK = False


class HFTokenizerAdapter:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str, max_len: Optional[int] = None) -> np.ndarray:
        ids = self.tok.encode(text)
        if max_len is not None:
            ids = ids[:max_len]
        return np.array(ids, dtype=np.int64)

    def decode(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids])

    @property
    def vocab_size(self):
        return len(self.tok)


def _cfg_get(cfg, *names, default=None):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    return default


class HFHandle(ModelHandle):
    backend = "hf"

    def __init__(self, name: str, revision: str = "main", dtype="float32"):
        if not TORCH_OK:
            raise RuntimeError("transformers/torch not installed — use synthetic backend")
        tok = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=getattr(torch, dtype))
        model.eval()
        self.model = model
        cfg = model.config
        n_layer = int(_cfg_get(cfg, "n_layer", "num_hidden_layers", default=0))
        n_embd = int(_cfg_get(cfg, "n_embd", "hidden_size", "d_model", default=0))
        n_head = int(_cfg_get(cfg, "n_head", "num_attention_heads", default=0))
        n_pos = int(_cfg_get(cfg, "n_positions", "max_position_embeddings", "n_ctx", default=0))
        n_inner = int(_cfg_get(cfg, "n_inner", "intermediate_size", default=0) or 4 * n_embd)
        rope = "rope" in json.dumps(cfg.to_dict()).lower()
        rms = _cfg_get(cfg, "rms_norm_eps", default=None) is not None
        spec = ArchSpec(
            arch=name, layers=n_layer, hidden=n_embd, heads=n_head,
            mlp_hidden=n_inner, vocab=int(cfg.vocab_size),
            ctx_len=n_pos, positional="rope" if rope else "learned",
            norm="rmsnorm" if rms else "layernorm",
            attention="mha", mlp_act="gelu",
            tie_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
            license="see model card", source="hf", hf_id=name)
        super().__init__(name, spec, HFTokenizerAdapter(tok))
        self._hooks = []
        self._collect = {}

    # ------------------------------------------------------------------
    def _blocks(self):
        """Locate decoder blocks across GPT2/GPTNeoX/Llama-style models."""
        m = self.model
        for attr in ("transformer.h", "gpt_neox.layers", "model.layers", "model.decoder.layers"):
            obj = m
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, (list, torch.nn.ModuleList)):
                    return list(obj)
            except AttributeError:
                continue
        raise RuntimeError("could not locate decoder blocks for this architecture")

    def _submodules(self, block):
        attn = getattr(block, "attn", None) or getattr(block, "attention", None) \
            or getattr(block, "self_attention", None) \
            or getattr(block, "self_attn", None)
        mlp = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
        return attn, mlp

    # ------------------------------------------------------------------
    def _attention_modules(self):
        attns = []
        for block in self._blocks():
            attn, _mlp = self._submodules(block)
            if attn is None:
                raise RuntimeError("no attention module found in a block")
            attns.append(attn)
        return attns

    def _head_hook(self, layer: int):
        """Zero the channel slice of masked heads in attention output [B,T,D].
        Assumes contiguous per-head blocks (asserted via hidden/heads)."""
        e = self.spec.hidden // self.spec.heads
        if e * self.spec.heads != self.spec.hidden:
            raise RuntimeError("hidden not divisible by heads; head ablation unsafe")
        mask = self.head_mask

        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() != 3 or t.shape[-1] != self.spec.hidden:
                return out  # unexpected layout: do not corrupt silently
            zeroed = torch.zeros_like(t[..., :e])
            parts = []
            for h in range(self.spec.heads):
                sl = t[..., h * e:(h + 1) * e]
                parts.append(zeroed if mask[layer, h] == 0 else sl)
            new_t = torch.cat(parts, dim=-1)
            if isinstance(out, tuple):
                return (new_t,) + out[1:]
            return new_t
        return hook

    @staticmethod
    def _zero_hook(module, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)

    def _install_hooks(self, collect: bool = False):
        self._remove_hooks()
        blocks = self._blocks()
        for l, block in enumerate(blocks):
            if collect:
                pre = self._collect.setdefault(l, {})
                attn, mlp = self._submodules(block)
                if attn is not None:
                    self._hooks.append(attn.register_forward_hook(
                        lambda mod, inp, out, l=l: self._capture(l, "attn", out)))
                if mlp is not None:
                    self._hooks.append(mlp.register_forward_hook(
                        lambda mod, inp, out, l=l: self._capture(l, "mlp", out)))
            if l in self.skip_layers:
                self._hooks.append(block.register_forward_hook(self._zero_hook))
            else:
                attn, mlp = self._submodules(block)
                mods = self.skip_modules.get(l, [])
                if "attn" in mods and attn is not None:
                    self._hooks.append(attn.register_forward_hook(self._zero_hook))
                if "mlp" in mods and mlp is not None:
                    self._hooks.append(mlp.register_forward_hook(self._zero_hook))
        # head ablation via output-slice hooks (portable; the head_mask kwarg
        # proved to be a silent no-op on some archs — see MILESTONE_2_AUDIT.md)
        if self.head_mask is not None and not collect:
            for l, attn in enumerate(self._attention_modules()):
                if (self.head_mask[l] == 0).any():
                    self._hooks.append(attn.register_forward_hook(self._head_hook(l)))

    def _capture(self, layer: int, name: str, out):
        t = out[0] if isinstance(out, tuple) else out
        if isinstance(t, torch.Tensor):
            self._collect[layer][name] = t.detach().numpy().astype(np.float32)
        return out

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._collect = {}

    def forward(self, ids: np.ndarray, collect: bool = False):
        if ids.ndim == 1:
            ids = ids[None, :]
        t = torch.tensor(ids)
        self._install_hooks(collect=collect)
        try:
            with torch.no_grad():
                out = self.model(t, output_hidden_states=collect)
        finally:
            captures = self._collect
            hs = out.hidden_states if collect else None
            self._remove_hooks()
        if not collect:
            return out.logits.detach().numpy(), None
        n_layers = self.spec.layers
        cache = {"resid_in": {}, "attn_out": {}, "mlp_out": {},
                 "attn_probs": {}, "hiddens": {}}
        for l in range(n_layers):
            # hidden_states[l] = residual ENTERING layer l; [-2] enters final norm
            if hs is not None and l < len(hs):
                cache["resid_in"][l] = hs[l].detach().numpy().astype(np.float32)
            if l in captures:
                cache["attn_out"][l] = captures[l].get("attn")
                cache["mlp_out"][l] = captures[l].get("mlp")
            if hs is not None and l + 1 < len(hs):
                cache["hiddens"][l] = hs[l + 1].detach().numpy().astype(np.float32)
        cache["final_hidden"] = (hs[-1].detach().numpy().astype(np.float32)
                                 if hs is not None else None)
        return out.logits.detach().numpy(), cache

    def state_dict(self) -> TensorDict:
        return {k: v.detach().numpy().astype(np.float32)
                for k, v in self.model.state_dict().items()}

    def load_state_dict(self, sd: TensorDict):
        self.model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})

    def logprob_option(self, prompt_ids: np.ndarray, option_ids: np.ndarray) -> float:
        if len(option_ids) == 0:
            return 0.0
        ids = np.concatenate([prompt_ids, option_ids])[None, :]
        if ids.shape[1] > self.spec.ctx_len:
            ids = ids[:, -self.spec.ctx_len:]
        logits, _ = self.forward(ids)
        logp = logits[0] - _logsumexp(logits[0], axis=-1, keepdims=True)
        n = len(option_ids)
        lp = 0.0
        for i in range(n):
            pos = ids.shape[1] - n + i - 1
            lp += float(logp[pos, option_ids[i]])
        return lp / n

    def param_groups(self) -> Dict:
        import re
        sd = self.state_dict()
        g = {"embedding": {}, "final_norm": {}, "layers": {}}
        layer_pat = re.compile(r"^(.+)\.(\d+)\.(.+)$")
        for k, v in sd.items():
            m = layer_pat.match(k)
            if not m:
                if "wte" in k or "embed" in k or "word_embeddings" in k:
                    g["embedding"]["wte"] = v
                elif "wpe" in k or "position" in k:
                    g["embedding"]["wpe"] = v
                elif "ln_f" in k or "final" in k:
                    g["final_norm"][k] = v
                continue
            l = int(m.group(2))
            if l >= self.spec.layers:
                g["final_norm"][k] = v
                continue
            lname = m.group(3).lower()
            grp = ("attn" if ("attn" in lname or "query" in lname or "key" in lname
                              or "value" in lname or "dense" in lname or "attention" in lname)
                   else "mlp" if ("mlp" in lname or "linear" in lname or "fc" in lname
                                  or "gate" in lname or "up_" in lname or "down" in lname)
                   else "norm1" if ("ln_1" in lname or "input_layernorm" in lname
                                    or "ln_1." in lname)
                   else "norm2" if ("ln_2" in lname or "post_attention" in lname)
                   else "other")
            g["layers"].setdefault(l, {}).setdefault(grp, {})[m.group(3)] = v
        return g


def _logsumexp(a, axis=-1, keepdims=False):
    m = a.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def hf_available(name: str = "gpt2") -> bool:
    if not TORCH_OK:
        return False
    try:
        transformers.AutoTokenizer.from_pretrained(name)
        return True
    except Exception:
        return False
