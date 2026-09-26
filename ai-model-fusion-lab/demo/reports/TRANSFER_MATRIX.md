# TRANSFER MATRIX — Milestone 3 (§11)

Canonical run: 2026-09-17, full mode (n=40 items/seed, held-out micro suites,
eval seeds [1,2] for matrix cells / [1,2,3] for the headline; refit 150 steps).
All components selected by the specificity screen (genome-ablation candidates +
same-lineage training-delta nominees; cross-suite ablation in the SOURCE).

## Cells (baseline → calibrated, held-out, mean over seeds)

| # | source → target | capability | component (specificity) | baseline | transferred | Δ | random ctl | capacity ctl | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A-math → nomath-base | math | math_head_205d6f (+22.5) | 25.0 | 32.5 | +7.5 | 31.25 | 31.25 | INSUFFICIENT_EVIDENCE |
| 2 | A-code → nomath-base | coding | coding_module_attn_373b65 (+30.0) | 100.0 | 100.0 | 0.0 | 10.0 | 10.0 | NO_TRANSFER |
| 3 | A-reason → nomath-base | reasoning | reasoning_head_515c4f (+7.5) | 21.25 | 20.0 | −1.25 | 15.0 | 15.0 | NO_TRANSFER |
| 4 | A-math → A-code | math | math_head_205d6f (+22.5) | 23.75 | 30.0 | +6.25 | 20.0 | 28.75 | SPECIALIZED_TRANSFER_EVIDENCE |
| 5 | math-wiz → nomath-base | math | math_head_9b2a1a (+17.5) | 25.0 | 35.0 | +10.0 | 31.25 | 31.25 | INSUFFICIENT_EVIDENCE+INTERFERENCE |
| 6 | A-math → weak-base | math | math_head_205d6f (+22.5) | 37.5 | 37.5 | 0.0 | 35.0 | 33.75 | NO_TRANSFER+INTERFERENCE |

Row highlights:
- **Cell 2 is the honest "no headroom" case**: the target already scores 100 on
  held-out coding; there is nothing to transfer. Recorded, not hidden.
- **Cell 4 initially claimed SPECIALIZED_TRANSFER_EVIDENCE** (passed random AND
  parameter-matched capacity controls at seeds [1,2]). The §23 reproduction
  pass re-ran it with seeds [1,2,3]: delta fell to **+4.16 (std 6.24,
  significance floor 12.48) → NO_EVIDENCE_OF_SPECIALIZED_TRANSFER**.
  The claim was a two-seed fluctuation and was RETRACTED. No cell achieved
  REPRODUCED status.
- **Cells 1 and 5** are positive-but-below-floor / control-failing — exactly the
  INSUFFICIENT_EVIDENCE tier (§21), reported with n, seeds, std, floor.
- **Independent-lineage rows (5, 6)** underperform same-lineage rows, consistent
  with M2's alignment ceiling (cos 0.90, R² 0.815 at d64→d96).

## Verdict vocabulary (§16, never merged)
SPECIALIZED_TRANSFER_EVIDENCE · PARTIAL_TRANSFER · INSUFFICIENT_EVIDENCE ·
NO_EVIDENCE_OF_SPECIALIZED_TRANSFER · NO_TRANSFER (+INTERFERENCE suffix when a
non-target capability regresses >10 points). TRANSFER_STATUS uses
UNKNOWN/UNTESTED/FAILED/PARTIAL/PROMISING/REPRODUCED — "PROVEN" is deliberately
absent from the enum (§36).

## Statistics (§21)
Each cell: n=40 items/seed, seeds as noted, mean ± std across seed draws,
significance floor = max(5.0, 2·std). Per-seed scores for every cell are in the
experiment database (type=component-transfer, EXP-00877…EXP-00888) and in
REPRODUCTION.json. No cell report relies on a single seed draw.
