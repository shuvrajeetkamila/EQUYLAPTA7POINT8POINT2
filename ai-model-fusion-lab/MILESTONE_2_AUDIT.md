# MILESTONE 2 — CODEBASE AUDIT (PHASE 0)

Date: 2026-09-17 · Auditor: autonomous agent · Method: every claim below was
verified by *executing* the relevant subsystem in this environment (Python
3.13, CPU-only, numpy 2.3.5, torch 2.x CPU, transformers 4.x), not by reading
alone. Evidence: `/home/user/fusionlab_data/m2_audit_quickrun.log`,
probe transcripts reproduced in this file.

## Environment facts discovered

* Installed packages do NOT persist between sessions — torch/transformers were
  absent at Milestone-2 start and had to be reinstalled. The numpy synthetic
  path is the only zero-install path; this is by design (spec §18 Mode B).
* All Milestone-1 artifacts (6 trained micro models, 40-experiment DB,
  tokenizer, frozen private suites) survived intact and reload correctly.

## Verdict table (subsystem → status → evidence)

| Subsystem | Status | Evidence |
|---|---|---|
| models/base.py (ModelHandle, tokenizer, logprob_option) | WORKING | used by every probe; truncation keeps the continuation tokens |
| models/backend_numpy.py | WORKING | gradient check 7.4e-4; reference backend |
| models/backend_torch.py | WORKING | bit-equivalent to numpy (1.5e-7); trains 111k-param models in ~85s/150ep |
| models/synthetic.py (factory, save/load) | WORKING | 6 profiles reloaded and re-benchmarked |
| models/corpora.py + benchmarking/suites.py | WORKING | frozen private holdout; train/test value-disjoint by construction |
| benchmarking/harness.py | PARTIAL | correct results, but scores every item TWICE (2× cost; painful on real models) |
| benchmarking/text_suites.py | PARTIAL | works; known suite artifacts (superlative wording, single-word agent format) documented in RESULTS.md |
| models/autopsy.py (structure + weight stats) | WORKING | gpt2 + pythia autopsy via param_groups; tree_map renders |
| models/autopsy.py activation_autopsy | **BROKEN for HF** | `KeyError: 'resid_in'` — HF forward(collect=True) returns empty cache. WORKING for numpy/torch backends |
| models/registry.py + compatibility | WORKING | shape-gate blocks incompatible merges (tested) |
| models/hf_probe.py | PARTIAL | verified live for several ids in M1; depends on HF config field naming; no revision pinning of results |
| models/backend_hf.py — load/forward/layer/module ablation | WORKING | gpt2: layer-ablation logit delta 97.7, module-ablation delta 7.8 |
| models/backend_hf.py — **head_mask ablation** | **BROKEN (silent no-op)** | zeroing head (0,0) on gpt2 produced logit delta **0.0** — the `head_mask` forward kwarg does nothing in this transformers version/arch combination. Head-level claims in M1 were therefore micro-only; this is a Milestone-2 P1 fix |
| models/backend_hf.py — arch coverage | PARTIAL | verified: GPT-2, GPT-NeoX/Pythia. Unverified: Llama/Qwen/Mistral/Gemma style (no checkpoints loaded in M1). Plan: arch-gauntlet verification with tiny random checkpoints (weights random, architecture real; capability scores meaningless and never reported) |
| ablation/ablator.py (layer/module scan, contribution report) | WORKING | canonical run deltas recorded; evidence-phrased output |
| ablation/ablator.py head_ablation_scan | PARTIAL | works on micro backends via head_mask; **broken on HF** (inherits the no-op bug above) |
| merging/strategies.py (avg, slerp, ties, dare, task-arithmetic, layer-selective) | WORKING | shape-gate test rejects mismatched dicts; all strategies finite |
| merging/engine.py (merge experiments, transplant, specialist builder) | WORKING | canonical run: 4 methods × coefs recorded; independent-model merges REJECTED by design; same-base merges accepted |
| distillation/distill.py | PARTIAL→FIXED-IN-M1 | M1 crashed on shape mismatch; fixed (teacher labels now [B,T-1,V], probabilities). Audit re-run: completes end-to-end (quick run EXP-00040). Still needs: explicit shape-contract assertions + clear errors (Phase 1A) |
| adapters/projection.py | PARTIAL | ridge alignment works but R²=1.0 saturated at n≈d; needs low-rank option + before/after protocol + honest n<d caveat |
| experts/specialist.py — Specialist, routing MC fusion | WORKING | routed super-model matched best-per-suite in canonical run |
| experts/specialist.py — **SuperModel.stats()** | **BROKEN** | `AttributeError: 'list' object has no attribute 'items'` — never exercised in M1 demo |
| router/router.py | WORKING | mixed queries route correctly (canonical run transcript) |
| experiments/tracker.py | WORKING | 40 experiments, config-hash dedupe verified by test |
| licensing/registry.py | WORKING | closed source → NOT DISTRIBUTABLE (tested) |
| app/server.py + dashboard | PARTIAL | serves, API live; dead code in /api/state; missing Milestone-2 Genome section |
| cli.py | WORKING | all spec-§21 commands ran in M1 |
| demo/run_demo.py | WORKING (after M1 fixes) | quick run now completes end-to-end (170s) with zero crashes |
| demo/resume_stages.py | **BROKEN (dead code executes)** | lines 41–53 train 4 models whose results are never used (~4 min wasted per run); unused imports |
| tests/test_core.py | WORKING | 10/10 pass (1 skip without torch) |
| make_release.sh | WORKING | 83 KB source zip |

## Undocumented assumptions found (to fix or document)

1. **Head layout**: head ablation (once fixed) assumes contiguous per-head
   channel blocks in attention output `[B, T, H*e]` — true for GPT-2, GPT-NeoX,
   Llama, Qwen2, Mistral, Gemma attention modules; must be asserted per arch
   (head-count × head-dim == hidden).
2. **Tokenizer**: micro suites assume the shared 141-token vocab; HF models use
   their own tokenizer with text suites — cross-family scoring is meaningless
   and never done.
3. **Context length**: `logprob_option` left-truncates to keep the option
   tokens; suites must respect `ctx_len` (micro ctx=32 — enforced in corpora).
4. **`head_mask` attribute semantics differ per backend** (micro: probability
   mask inside softmax; HF: was kwarg — now hook-based slice masking).
5. **eval determinism**: suites are seeded; bootstrap CI seeded (rng(0)) — OK.
6. **tie_embeddings** assumed in micro backends only; HF makes no such
   assumption.

## Capability claims audit (Phase 24 honesty check)

* "Capability differences" — REAL (trained-data-induced, held-out measured).
* "Head-level ablation on real models" — was claimed by interface existence
  only; **audit verdict: NOT WORKING on HF in M1** → fixed in Phase 1B.
* "Distillation" — works post-fix; regression test being added.
* "Vision" — absent; no claims made. Still [SIMULATED/REQUIRES EXTERNAL MODEL].
* "Super-model" — routed ensemble of real models/specialists; no claim of a
  single fused frontier model. Unchanged in M2.

## Phase-1 fix list derived from this audit

1. Distillation shape-contract assertions with EXPECTED/RECEIVED/REASON errors.
2. HF head ablation via attention-output-slice hooks (+ per-arch attention
   module discovery + shape assertions); regression tests on gpt2 + pythia.
3. activation_autopsy for HF via hooks (residual/attn/mlp per layer).
4. SuperModel.stats() fix + route-once fix.
5. eval_suite single-pass scoring.
6. resume_stages.py dead-code removal.
7. Arch gauntlet: tiny-random Llama/Qwen2/Mistral/Gemma + real gpt2/pythia must
   pass LOAD→AUTOPSY→FORWARD→BENCHMARK→ABLATION before "supported".
