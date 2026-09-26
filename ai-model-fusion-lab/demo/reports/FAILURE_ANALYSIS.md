# FAILURE ANALYSIS — Milestone 3 (structured failures, §31, §36)

Every failed or retracted result below is preserved with its mechanism. None
were silently dropped; each shaped the final design.

## F1. Hot fine-tune destroys the specialist's own capability
- **What happened:** first same-init fine-tune recipe (profile corpus,
  lr 5e-3 × 45 ep) gave A-math math 65 but collapsed A-code's code score
  90 → 40 (worse than the base it started from).
- **Diagnosis:** profile corpora pull weights toward a different distribution
  than the held-out suites; a converged 111k-param model has no slack.
- **Fix:** measured recipe sweep (7 configs) → per-capability recipes on
  micro-format train-draw items; A-math 3e-3×45 (math +32.5), A-code 1e-3×30
  (no gain — reported as "no headroom"), A-reason 1e-3×30 (+23.3).

## F2. Candidate selection by |ablation delta| nominates generic fragility
- **What happened:** every L0 head of the first A-math showed identical +45
  "math evidence"; ablating anything destroyed everything.
- **Diagnosis:** in a 2-layer model, single-head ablation measures generic
  brittleness, not capability localization.
- **Fix:** the specificity screen (cross-suite ablation profile; M2 lesson
  mechanized). It now gates every transfer.

## F3. Whole-module replacement destroys the host's function
- **What happened:** inserting the screen-winning MODULE_MLP (specificity +10)
  lifted nothing and regressed target code by 36–40 points in early runs.
- **Diagnosis:** replacement removes the target's own module function; the
  source module's "math skill" does not compensate for the lost general
  function at this scale.
- **Fix:** added the task-arithmetic ADAPTER mechanism ((1−g)·W_t + g·W_s,
  guarded gate) — and even at the optimal gate, held-out math did not improve.

## F4. Unguarded calibration picks destructive settings
- **What happened:** residual-gate calibration improved calib loss while
  held-out math fell (g=1.5–2.0 amplifying a harmful branch).
- **Fix:** two-sided calibration reporting (§14) + guarded gate search that
  penalizes general/non-target CE damage. Guard damage on the probe split is
  ~0 for adapters, revealing that capability damage (code MC) is invisible to
  raw-LM-CE guards — recorded as a finding, and guarded further by the
  non-target regression veto in the verdict.

## F5. Single positive matrix cell did not reproduce
- **What happened:** A-math → A-code [math] showed +6.25 beating BOTH controls
  (random 20.0, capacity 28.75) at seeds [1,2] → SPECIALIZED_TRANSFER_EVIDENCE.
- **Reproduction (§23):** seeds [1,2,3] → delta +4.16, std 6.24, floor 12.48 →
  NO_EVIDENCE_OF_SPECIALIZED_TRANSFER. Claim retracted; status FAILED.
- **Lesson:** two-seed positives at this noise level (±6–7) are fluctuations.

## F6. Component-scaling is flat
- K=1..8 top training-delta channel groups with a shared guarded gate:
  held-out math never left the baseline band (20.0–20.8 vs base 20.8).
- Conclusion: the fine-tune delta is distributed; sparse component sketches
  carry no measurable slice of the skill.

## F7. Real-model alignment fits are underdetermined
- gpt2 (d768) → pythia-70m (d512), 43 sentence pairs: linear/low_rank fits
  SATURATED (27 train < 768 dims), held-out R² ≈ −0.013 (degenerate
  mean-predictor under sentence-mean pooling); the learned fit diverges
  (loss 2.3e8). All flagged SATURATED by the alignment engine — no "perfect
  alignment" claims from saturated fits (§7). Weight-level transfer is
  impossible anyway (different tokenizers + architectures); the honest
  ceiling for real-model cross-model transfer here is DISTILL/ROUTING.

## F8. Known limitations of the harness (recorded, not hidden)
- micro-suite MC accuracy at n=40 has ±6–8 noise; the significance floor
  (max(5, 2σ)) and 3-seed consistency requirement exist because of this.
- Sentence-mean pooling for cross-tokenizer activation pairing is coarse;
  a token-mapped CCA/procrustes study is future work (§17 YAML config exists
  for heavier runs).
- The capacity control (param-matched adapter trained on the same data)
  frequently outperforms component insertion — which is precisely the §15
  logic for the NO EVIDENCE verdict, and it is applied verbatim.
