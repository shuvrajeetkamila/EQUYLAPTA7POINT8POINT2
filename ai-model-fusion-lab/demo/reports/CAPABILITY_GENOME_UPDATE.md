# CAPABILITY GENOME UPDATE — Milestone 3

Status tag: [WORKING] — all numbers below are measured on the local synthetic
models (SYNTHETIC label applies to every model mentioned; no REAL model is
part of the genome program beyond the separate alignment experiment).

## New genomes (canonical-run lineage, 2026-09-17)

| model | lineage | params | records | baselines (held-out micro, 3-seed) |
|---|---|---|---|---|
| nomath-base | lineage-A root (never sees math data) | ~111k | full scan | math 16.7 · code 85–100 · reason 15–21.7 |
| A-math | nomath-base + micro-math train-draws ft (lr 3e-3 × 45ep + 500 retention) | ~111k | 17 candidates ≥3Δ | **math 49.2** (+32.5 vs base) · code 30–35 · reason 13.3 |
| A-code | nomath-base + micro-code train-draws ft (lr 1e-3 × 30ep) | ~111k | 6 candidates | code ≈ base (no gain; ceiling) |
| A-reason | nomath-base + micro-reason train-draws ft (lr 1e-3 × 30ep) | ~111k | 14 candidates | reason 38.3 (+23.3 vs base) |

Pre-existing genomes (M1/M2): math-wiz, code-smith, logic-owl, generalist,
generalist-lg, weak-base — unchanged, still loadable, still used as
independent-lineage controls.

## Component identification results now in the genomes

- A-math: 17 genome-ablation candidates; 24 nominations after merging the
  training-delta channel; screen winner **L0.attn.head3**
  (target_drop +25.0 / max_other_drop +2.5 / specificity **+22.5**).
- A-code: best screen candidate coding_module_attn (+30.0 specificity) — but the
  target is already at ceiling (100), so it transfers nothing (NO_TRANSFER).
- A-reason: reasoning_head (+7.5 specificity) — transfer delta −1.25 (NO_TRANSFER).

## TRANSFER_STATUS updates (§23 — never PROVEN)

| component | after canonical run | after reproduction pass |
|---|---|---|
| math_head_205d6f | PROMISING (cell 4 evidence) | **FAILED** (did not reproduce: +4.16 < floor 12.48) |
| math_head_287f94 / headline comp | FAILED (INSUFFICIENT) | FAILED (stable: +4.17 < floor 14.14) |
| math_head_beb231 (math-wiz) | FAILED (INSUFFICIENT) | FAILED (failed controls + interference) |
| coding_module_attn_373b65 | FAILED (no headroom) | — |
| reasoning_head_515c4f | FAILED (delta −1.25) | — |

Library totals: 47 metadata records, 81 external tensor stores
(fusionlab_data/components/*.npz), indexed under
fusionlab_data/library/{math,coding,reasoning}/.

## Honesty notes
- Genome baselines are 3-seed means; single-seed numbers are never used for
  candidate claims.
- Training-delta candidates carry delta=0.0 by definition — rank evidence only.
- The A-* genomes were trained on micro-format items from TRAIN seed draws
  (200–209); eval seeds (1,2,3) and screen seeds (1234–1236) draws are disjoint.
  This is an in-distribution held-out protocol and is labeled as such.
- Catastrophic forgetting in A-math (code 85→~32) is REPORTED, not hidden: it
  is itself evidence that capabilities compete for the same tiny weight budget,
  and the specificity screen exists to prevent such components from being
  mistaken for clean "math components".
