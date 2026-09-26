# COMPONENT TRANSFER RESULTS — Milestone 3

## The experiment (§9 five-arm ladder + adapter arms + controls)

Central question: *can an experimentally identified component from Model A be
transformed (aligned → refitted → calibrated) and inserted into Model B so that
B's HELD-OUT capability measurably improves beyond random-harvest and
parameter-matched capacity controls?*

**Headline pair (same-init lineage):** A-math (source, math +32.5 over base) →
nomath-base (target, math-deficient by construction, zero math training data).
Held-out micro-math, n=40/seed × 3 seeds (1,2,3).

### Ladder — A-math → nomath-base, component math_head_205d6f (HEAD L0.3, specificity +22.5)

| system | held-out mean ± std |
|---|---|
| TARGET_BASELINE | 20.8 ± 7.2 |
| DIRECT_TRANSFER (raw insert) | 21.7 ± 7.2 |
| ALIGNED_INSERT (activation-scale) | 21.7 ± 7.2 |
| REFITTED_TRANSFER (+150-step masked-gradient slice refit, 4,144 params) | 21.7 ± 7.2 |
| CALIBRATED (+residual gate g=2.0, calib split) **← final** | 25.0 ± 7.1 |
| ADAPTER_TRANSFER (task arithmetic, guarded gate; gate→1.0) | 21.7 ± 7.2 |
| ADAPTER_REFITTED (adapter + refit) | 18.3 ± 5.1 |
| RANDOM_HARVEST_CONTROL (same-size component, random init) | 33.3 ± 8.2 |
| CAPACITY_CONTROL (param-matched adapter, trained identically) | 30.8 ± 5.1 |

Verdict: **NO_EVIDENCE_OF_SPECIALIZED_TRANSFER+INTERFERENCE** —
delta +4.17 is below the significance floor (14.14 = 2·std), does not beat the
random control, does not beat the capacity control, is not consistent across
seeds, and regresses non-target capabilities (code −29.2, lang −18.0,
know −34.7).

### Component identification (two independent channels)
1. **Genome ablation** (M2 engine): 692 ablation records total; for A-math,
   17 candidates at |Δ|≥3 on micro-math.
2. **Training-delta analysis** (new, same-lineage only): rank heads/channel
   groups by L2 displacement vs the same-init base — skill location by
   construction. Rank evidence only (never read as effect size).

Both feed the **specificity screen**: forward-time ablation of each candidate
in the SOURCE across all 7 micro suites; specificity = target_drop −
max_other_drop. Generic fragility (any-ablation-kills-everything) is filtered
out — the M2 lesson, mechanized. Selected component: L0.attn.head3
(target_drop +25.0, max_other_drop +2.5, specificity +22.5).

### Component-scaling study (§12: "more components" must be measured)
Top-K training-delta channel groups (K=1,2,4,8), one shared guarded gate,
all gates chosen on calib/probe splits:

| K | gate | held-out math (base 20.8) | code | reason |
|---|---|---|---|---|
| 1 | 0.2 | 20.8 | 100.0 | 21.7 |
| 2 | 0.2 | 20.0 | 100.0 | 21.7 |
| 4 | 0.2 | 20.0 | 100.0 | 21.7 |
| 8 | 0.4 | 20.0 | 100.0 | 23.3 |

Flat. More components ≠ more transferred skill. Interference matrix
(singles/pairs/triple, gates per component): no pair/triple beat its best
single; the guarded gate correctly drives useless components to gate 0.

### Full-model interpolation bound (diagnostic, NOT a component transfer)
B0→A_math weight-space interpolation, held-out math (n=40×3 seeds):
g=0.2→21.7 · g=0.4→26.7 · g=0.6→26.7 · g=0.8→49.2 · g=1.0→**51.7**
(guard CE damage ≤ 0). Interpretation: the +32.5 math skill **is** present and
weight-accessible, but it is DISTRIBUTED — no single component or small set of
components carries a measurable share of it under strict held-out criteria.

### Multi-component + interference (§12–13)
Singles/pairs/triple with per-component guarded gates and joint refit:
best single = math head alone; every pair/triple ≤ its best single component;
no synergy detected; interference only where replacement damaged the host
module. All runs preserved in the experiment database.

## Component library (§24)
47 metadata records + 81 external tensor stores (npz), keyed by capability:
math / coding / reasoning. Every record carries full provenance (source model,
architecture, layer, shapes, baseline/ablation/restoration scores, alignment,
refit, calibration methods, TRANSFER_STATUS, notes). TRANSFER_STATUS values
actually present after the canonical run + reproduction: FAILED (majority,
incl. the retracted cell), PROMISING→FAILED on retest, UNTESTED (never
transferred). "PROVEN" absent by design.

## Label vocabulary used (§16 — never merged)
DIRECT_TRANSFER, ALIGNED_TRANSFER(=ALIGNED_INSERT), REFITTED_TRANSFER,
CALIBRATED (ladder state), ADAPTER_TRANSFER, ADAPTER_REFITTED — each reported
separately in every run; the final-system selection between CALIBRATED /
ADAPTER_TRANSFER / ADAPTER_REFITTED is recorded in `final_arm`, never blended.
