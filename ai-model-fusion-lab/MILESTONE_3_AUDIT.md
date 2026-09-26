# MILESTONE 3 AUDIT — EQUYLAPTA [AI MODEL]

Method: codebase inspected in place at M3 start; every WORKING claim below was
verified by execution in M2 (see MILESTONE_2_AUDIT.md + 23 passing tests) and
re-verified at M3 start (tests green, all M2 artifacts reload).

## What exists and its M3-readiness

| Subsystem | M2 status | M3 gap (what must be added) |
|---|---|---|
| capability_genome/ | WORKING — hierarchical discovery L1→L4, restoration checks, confidence, chance baselines, 6 genomes incl. real gpt2 | contribution **vectors** per component; **transfer evidence** fields; **TRANSFER_STATUS** (§23); capability **signature** per model (§5) |
| component_extraction/transplant.py | WORKING — LAYER/MODULE/HEAD/CHANNEL_GROUP cut&copy with exact-slice tests, A/B/C protocol | formal **ComponentRecord** with full metadata (§3); finer types: Q/K/V/O/MLP-up/MLP-down projections (§3); tensor payloads stored OUTSIDE project; component library (§24) |
| component_synthesis/engine.py | WORKING — fallback hierarchy, regression guard, construction reports | raw vs aligned vs refitted vs calibrated **ladder comparison** (§9); **random controls** (§15); transfer labels (§16) |
| adapters/projection.py + lowrank.py | WORKING — ridge + low-rank, before/after | train/**heldout** split for alignment fit (§7); saturation flagging (n≈d); projection checkpoints; normalization/statistics matching (§6); explicit DimensionMismatchError (never silent reshape) |
| ablation/ablator.py | WORKING — layer/module/head/channel ablation + restoration | capability-vector evaluation helper for Test F non-target control (§6) |
| distillation/distill.py | WORKING — shape contracts, quality filter | label separation DISTILLED_TRANSFER vs component transfer (§16) — already distinct in DB, keep separate |
| models/backend_{numpy,torch}.py | WORKING — grad-checked, bit-equivalent, ablation hooks incl. channel masks | per-position activation collection exists (collect=True); needs **matched-prompt pair collector** for cross-tokenizer alignment (§7, §19) |
| models/backend_hf.py | WORKING — 7-arch gauntlet, hook-based head/module/layer ablation, activation autopsy | same pair-collector need for real models |
| benchmarking/suites.py + corpora | WORKING — micro suites (frozen private), text suites | **modular stronger suites** (§20): algebra, pattern, code-transform, word problems, paraphrase — with train/val/held-out splits |
| experiments/tracker.py | WORKING — dedupe, leaderboard | transfer-record convenience fields (method/label/control) |
| router/, licensing/, registry/, hf_probe, selector, arch_gauntlet | WORKING | no changes required for M3 core |
| experts/specialist.py | WORKING (stats/route fixed in M2) | untouched (super-model explicitly NOT the focus, §39) |
| GUI | PARTIAL | **Component Transfer Lab** panel (§29) + data-connected visual map (§30) |
| tools/check_release_size.py | NOT YET IMPLEMENTED | §34 |
| component_transfer/ | **NOT YET IMPLEMENTED** | the core of M3: alignment / refitting / calibration / transfer engine |
| transfer matrix, interference matrix | NOT YET IMPLEMENTED | §11, §13 |

## Bugs blocking M3: none known
M2 closed the blocking bugs (HF head ablation no-op, stats()/activation_autopsy
crashes, distillation contracts, OOM in stats/fingerprint). All regression
tests pass at M3 start.

## Evidence standard carried into M3 (§36-37)
Claims will be labeled by level: L1 correlation … L7 combination. M2 reached
L1–L3 (with restoration). M3 target: L4 (transfer improves another model) and
L5 (aligned/refitted transfer), with random + parameter-matched controls and
held-out evaluation deciding the verdict. `PROVEN` remains forbidden.
