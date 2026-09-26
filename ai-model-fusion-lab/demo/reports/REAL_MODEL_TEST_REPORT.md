# REAL MODEL TEST REPORT — Milestone 3 (§17, §19)

Experiment: activation-space alignment between two REAL pretrained models —
**gpt2** (d_model 768, GPT-2 arch, byte-BPE tokenizer) →
**EleutherAI/pythia-70m** (d_model 512, GPTNeoX arch, different tokenizer).
Weights: real, loaded from HuggingFace hub caches (never bundled into the
release; ~550 MB + ~160 MB, Apache-2.0 / open licenses).

## Compatibility engine findings (§18)

| check | result |
|---|---|
| tokenizer match | **NO** — vocabularies and segmentation differ; detected automatically |
| exact compatibility (same arch+fingerprint) | **NO** |
| structural compatibility (same shapes) | **NO** (768≠512, 12L vs 6L) |
| activation-space alignable | YES in principle (fit attempted; see below) |
| weight-level transfer possible | **NO** — different architectures AND tokenizers; any tensor mapping would be a silent reshaping, which the engine refuses (§7) |
| remaining honest routes | DISTILL (logit/hidden-state distillation) or learned ROUTING; both out of scope for the §1–40 component-transfer question |

## Alignment fits (paired sentence-mean activations, 43 text prompts from the
frozen text suites; train 27 / val 6 / held-out 10)

| method | train loss | val loss | held-out loss | held-out R² | flags |
|---|---|---|---|---|---|
| linear | 6.5e-06 | 0.208 | 0.551 | **−0.013** | SATURATED: n_train=27 < d=768 (underdetermined) |
| low_rank (r=128) | 6.5e-06 | 0.208 | 0.551 | −0.013 | SATURATED (same) |
| learned (MLP) | 2.27e8 | 2.27e8 | 2.27e8 | −4.2e8 | SATURATED; optimization diverged |

Reading: the near-zero train loss with held-out R² ≈ 0 means the linear map
interpolated its 27 training sentences and generalizes at the level of a
constant predictor (an artifact of sentence-mean pooling — the reported
"aligned cos 0.9996" is degenerate for near-constant predictions; R² is the
honest number). The engine's saturation flags fired on all three fits, so **no
alignment-quality claim is made** — exactly the §7 rule against claiming
"perfect alignment" from saturated fits.

## Verdict
- REAL-model cross-architecture component transplant: **NOT POSSIBLE** with
  this engine's no-silent-reshape guarantees (tokenizer + shape mismatches).
- REAL-model activation alignment at this data scale: **UNDERDETERMINED**
  (needs token-aligned pairing corpora; config exposed in §17 YAML for scaled
  runs — `experiments/real_transfer.yaml`-style config keys accepted by the
  runner).
- No REAL-model number is mixed into SYNTHETIC transfer results anywhere in
  this milestone (§36).

## What was REAL in this milestone
- gpt2 and pythia-70m weights, tokenizers, activations: REAL (HF).
- The alignment engine, saturation detection, mismatch detection: REAL code
  paths, executed (alignments/align-bc92145c.npz etc. persisted).
- Everything in TRANSFER_MATRIX.md / COMPONENT_TRANSFER_RESULTS.md: SYNTHETIC
  micro-models, labeled as such throughout.
