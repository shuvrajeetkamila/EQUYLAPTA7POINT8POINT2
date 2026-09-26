# Milestone-1 Canonical Run — Measured Results

Environment: CPU-only sandbox, 6 micro models (111k params each, torch backend,
seeded), gpt2 as an external real model. Everything below is **measured output**
of `python cli.py demo` (+ resume of stages 7b/9–11), logged in
`fusionlab_data/demo_output.log`, every experiment in
`fusionlab_data/experiments.jsonl`.

## Capability matrix (held-out MC suites, n=40/suite, accuracy %)

| MODEL       | agent | code | know | lang | math | multi | reason |
|-------------|------:|-----:|-----:|-----:|-----:|------:|-------:|
| math-wiz    | 62.5  | 22.5 | 100  | 67.5 | **75.0** | 67.5 | 20.0 |
| code-smith  | 75.0  | **100** | 100 | 65.0 | 30.0 | 70.0 | 10.0 |
| logic-owl   | 77.5  | 92.5 | 100  | 75.0 | 32.5 | 70.0 | **40.0** |
| polyglot    | 35.0  | 37.5 | 100  | **82.5** | 37.5 | 62.5 | 25.0 |
| generalist  | 72.5  | 100  | 100  | 82.5 | 30.0 | 70.0 | 17.5 |
| weak-base   | 85.0  | 85.0 | 100  | 60.0 | 27.5 | 70.0 | 52.5 |

Real differentials where profiles differ (math 75 vs 27–37; code 100 vs 22.5).
Honest noise: weak-base is oddly strong on agent/reason — small-data control
learns the general template style; reported as measured, not smoothed over.

## Real model (external, gpt2 163M params, plain-text suites n=15)

math 40% · knowledge 60% · multilingual 40% · reasoning 13% · coding 13% · agent 0%.
Caveats: tiny n; the reasoning superlative wording ("fasterest") and the
single-word agent continuation format are awkward — suite artifacts noted,
numbers reported anyway. `EleutherAI/pythia-70m` also loads (70.4M params) and
ablation on it was verified separately (text-math 0.30 → 0.20 with layer 0 removed).

## Ablation / localization (generalist, evidence-phrased)

```
baseline: micro-math 30 | micro-code 100 | micro-reason 18
removing layer 0        -> math −12.5 pts  [high contribution]
removing layer 1        -> math −2.5 pts
removing all-attn       -> code −87.5 pts | reason −10.0 pts
removing all-mlp        -> math +5.0 pts (within noise)
```

FINDINGS ARE EVIDENCE, not "the math layer": with a 2-layer residual model the
embedding path carries much of the signal; deltas are suite- and seed-dependent.

## Merging experiments (the scientifically interesting part)

1. **Independent models (different data mixes, different inits): ALL merges
   degraded the target capability → REJECTED by the specialist builder.**
   math: best merge 38% vs best parent 75%; code: 15% vs 100%; reason: 25% vs 40%.
   weighted-avg/slerp/ties/dare all tried; recorded as EXP-00011…34.
   *Interpretation:* weight merging preserves capability for same-base
   fine-tunes, not independently-trained models — the lab *discovered* this.

2. **Same-base fine-tune pair (shared base → math-tune + code-tune): merging works.**

   | model | micro-math | micro-code |
   |---|---:|---:|
   | shared base        | 20 | 75 |
   | ft-math            | 52 | 100 |
   | ft-code            | 32 | 100 |
   | **merged avg**     | **45** | **100** |
   | merged slerp       | 42 | 100 |
   | merged ties        | 18 | 100 |
   | merged dare        | 18 | 82 |

   Naive average/SLERP keep both capabilities; TIES/DARE trimming hurts at this
   tiny scale (little redundancy to exploit). Best same-base merge (EXP-00038,
   mean 72.5) tops the leaderboard.

## Segment transplant [RESEARCH EXPERIMENT]

`math-wiz[L0:2] → weak-base`: math 22% (donor 75%, recipient-before 27.5%) —
transplant did NOT transfer capability. Recorded, not hidden.

## Distillation (teacher math-wiz → weak-base-init student, filtered data)

Quality filter kept 507/1200 examples. Student: loss 2.55→1.43;
held-out math 42% vs control 28% vs teacher 75% — genuine partial transfer.

## Representation alignment (adapters)

Ridge projection math-wiz→generalist hidden states: cosine 1.0, R² 1.0 —
**saturated estimate** (n≈80 samples ≈ hidden dim 64); caveat recorded.

## Router + super-model

```
'What is 23 + 45 ?'                              -> math, reasoning
'Write a python function def add ( a , b ) :'    -> coding
'Ann is taller than Ben ... Who is tallest ?'    -> reasoning
'Write a Python program proving this theorem.'   -> coding + math + reasoning
'Hello in French is'                             -> multilingual
'To compute 12 * 9 use the'                      -> agent
```

Routed super-model on held-out suites: math 75 · code 100 · reason 40 · lang 75 —
matches the best specialist per suite with zero scores invented.

## Leaderboard (experiment DB, top entries)

```
EXP-00038 [merge-same-base] micro-math  mean=72.5
EXP-00035 [merge-same-base] micro-math  mean=71.2
EXP-00036 [merge-same-base] micro-math  mean=58.8
EXP-00037 [merge-same-base] micro-math  mean=50.0
EXP-00039 [distillation]    -           mean=50.0
```

39 experiments recorded, deduped by config hash. License engine: micro sources
Apache-2.0 + gpt2 MIT → COMPATIBLE; any `proprietary-closed` source →
NOT DISTRIBUTABLE.
