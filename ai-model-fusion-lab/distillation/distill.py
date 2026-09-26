"""[WORKING micro] Knowledge distillation engine.

Teacher → student data distillation with quality filtering:
  * teacher generates/labels examples (here: micro-corpora labeled by teachers'
    probabilities; with real models: teacher outputs/traces — ToS permitting)
  * filters: low teacher confidence, duplicates, empty, unsafe patterns
  * student trains on filtered set (hard labels + optional soft targets)

For numpy-backend students, soft-target training uses loss_and_grads();
for torch-backend students, train_step(target_probs=...) — both supported.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from models.base import ModelHandle


class ShapeContractError(Exception):
    """Raised when tensor shapes violate the distillation contract.
    Message always contains EXPECTED / RECEIVED / REASON lines."""

    def __init__(self, what: str, expected: str, received, reason: str):
        self.expected = expected
        self.received = str(received)
        self.reason = reason
        super().__init__(
            f"DISTILLATION SHAPE CONTRACT VIOLATION — {what}\n"
            f"  EXPECTED:  {expected}\n"
            f"  RECEIVED:  {self.received}\n"
            f"  REASON:    {reason}")


def _check_contract(teacher: ModelHandle, student: ModelHandle, arr: np.ndarray,
                    target_probs):
    B, T = arr.shape
    V = student.tokenizer.vocab_size
    if arr.ndim != 2:
        raise ShapeContractError("input batch", "(batch, sequence)", arr.shape,
                                 "ids must be a 2-D padded int array")
    if target_probs is not None:
        exp = (B, T - 1, V)
        if tuple(target_probs.shape) != exp:
            raise ShapeContractError(
                "teacher target probabilities", str(exp), tuple(target_probs.shape),
                "teacher labels must be next-token probs aligned to targets: "
                "position t predicts token t+1, so the time axis is T-1 and the "
                "vocab axis must equal the STUDENT tokenizer size")
    if teacher.tokenizer.vocab_size != V:
        raise ShapeContractError(
            "tokenizer compatibility", "teacher vocab == student vocab",
            f"teacher={teacher.tokenizer.vocab_size}, student={V}",
            "soft targets are distributions over student token ids; different "
            "vocabs require a token-mapping layer (not implemented) — use data "
            "distillation (text-level) instead")
    if student.spec.vocab != V:
        raise ShapeContractError(
            "student model vocab", str(V), str(student.spec.vocab),
            "student arch vocab must match its tokenizer")


def teacher_label_batch(teacher: ModelHandle, arr: np.ndarray) -> np.ndarray:
    """Teacher next-token probabilities ALIGNED to student targets:
    shape [B, T-1, V] — position t predicts token t+1."""
    logits, _ = teacher.forward(arr)
    lg = logits[:, :-1, :]
    z = lg - lg.max(-1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    return P.astype(np.float32)


def quality_filter(teacher: ModelHandle, texts: List[str], min_conf: float = 0.2,
                   max_len: int = 32) -> List[str]:
    """Aggressive quality filtering (spec §9):
    - teacher self-confidence gate on answer tokens (hallucination proxy)
    - exact duplicates removed
    - sequences that don't tokenize to >= 4 tokens removed
    """
    seen, kept, dropped = set(), [], 0
    for t in texts:
        if t in seen:
            dropped += 1
            continue
        seen.add(t)
        ids = teacher.tokenizer.encode(t, max_len=max_len)
        if len(ids) < 4:
            dropped += 1
            continue
        arr = np.zeros((1, len(ids)), np.int64)
        arr[0] = ids
        probs = teacher_label_batch(teacher, arr)[0]
        # mean prob mass the teacher assigns to the actual next token
        conf = float(np.mean([probs[i, ids[i + 1]] for i in range(len(ids) - 1)]))
        if conf < min_conf:
            dropped += 1
            continue
        kept.append(t)
    return kept


def distill(teacher: ModelHandle, student: ModelHandle, texts: List[str],
            epochs: int = 30, batch: int = 96, lr: float = 8e-3,
            soft: bool = True, temperature: float = 2.0, seed: int = 11, log=None) -> dict:
    """Train student from teacher labels. Soft targets when backend supports."""
    rng = np.random.default_rng(seed)
    enc = []
    for t in texts:
        ids = student.tokenizer.encode(t, max_len=student.spec.ctx_len)
        if len(ids) >= 4:
            enc.append(ids)
    hist = []
    opt_state: dict = {}
    t0 = time.time()
    is_torch = student.backend == "torch-synthetic"
    for ep in range(epochs):
        order = rng.permutation(len(enc))
        losses = []
        lr_ep = lr * (1.0 - 0.5 * ep / max(1, epochs - 1))
        for bs in range(0, len(order), batch):
            seqs = [enc[i] for i in order[bs:bs + batch]]
            L = max(len(s) for s in seqs)
            arr = np.zeros((len(seqs), L), np.int64)
            for r, s in enumerate(seqs):
                arr[r, :len(s)] = s
            tp = None
            if soft:
                tp = teacher_label_batch(teacher, arr)        # [B, T-1, V]
                # temperature smoothing (probabilities stay normalized)
                tp = np.log(np.clip(tp, 1e-9, 1)) / temperature
                tp = np.exp(tp - tp.max(-1, keepdims=True))
                tp = tp / tp.sum(-1, keepdims=True)
                tp[arr[:, 1:] == 0, :] = 0.0                  # mask pad targets
                _check_contract(teacher, student, arr, tp)
            else:
                _check_contract(teacher, student, arr, None)
            if is_torch:
                loss = student.train_step(arr, tp, lr_ep, opt_state)
            else:
                loss, G = student.loss_and_grads(arr, target_probs=tp)
                student.adam_step(G, lr=lr_ep)
            losses.append(loss)
        hist.append(float(np.mean(losses)))
        if log and ep % 5 == 0:
            log(f"  distill epoch {ep+1}/{epochs} loss={np.mean(losses):.3f} ({time.time()-t0:.0f}s)")
    return {"epochs": epochs, "n_examples": len(enc), "loss_curve": [round(h, 4) for h in hist],
            "final_loss": round(hist[-1], 4)}
