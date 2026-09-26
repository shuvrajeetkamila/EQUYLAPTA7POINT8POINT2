# MILESTONE 3 REPORT — Real Component Transfer Engine

EQUYLAPTA [AI MODEL] · AI Model Fusion Lab · 2026-09-17
Status: COMPLETE (canonical run + reproduction pass executed; all numbers below
are from the executed full run unless tagged otherwise).

---

## 1. Executive summary

The M3 central question — *can an experimentally identified component from
Model A be aligned/refitted/calibrated and transferred into Model B to produce
measurable HELD-OUT capability improvement that beats random-harvest AND
parameter-matched capacity controls, reproducibly, without destroying other
capabilities?* — was answered under the strictest reading of the spec:

> **Final answer: NO.** (No reproducible evidence of specialized single-
> component transfer was found; the full evidence chain, all mechanisms, all
> controls, and the one retracted positive are documented.)

The engine itself works: identification (2 channels), specificity screening,
dimension-safe extraction/insertion, 7 alignment methods with saturation
flags, masked-gradient refitting, guarded calibration, the full ladder with
random + capacity controls, bidirectional/matrix, multi-component scaling,
interference checks, and the real-model compatibility/alignment experiment all
executed and are unit-tested (40/40). What failed is the *science*, honestly:
at micro-model scale, capability skill is distributed across the whole
fine-tune delta; no component or small component set carries a measurable,
control-beating, reproducible share of it.

## 2. What was built (spec § → artifact)

| § | requirement | state |
|---|---|---|
| 2 | ≤120 MB + download guide | PASS — release content 1.5 MB; `MODEL_DOWNLOAD_GUIDE.md` (M2) unchanged; real weights never bundled |
| 3 | formal ComponentRecord | DONE — 18-type taxonomy, full provenance, external tensor store (47 metas / 81 npz) |
| 4 | Tests A–F | DONE — baseline/ablation/restoration evidence in every record; random + capacity + non-target vector in every run |
| 5 | signatures + contribution vectors | DONE — 3-seed capability vectors; genome-ablation + training-delta nominators |
| 6–7 | ≥7 alignment methods, held-out mandatory, saturation flags, no silent reshape | DONE — identity/linear/low_rank/learned/activation_stats/CCA-style/paired-projection; DimensionMismatchError; SATURATED flags fired on the real-model fits |
| 8 | refitting methods A–E, param counts | DONE — masked-gradient slice-only (exact trainable count 4,144 / 33,088), low-rank factorize, adapter+refit |
| 9 | five-arm ladder | DONE — raw→aligned→refitted→calibrated + ADAPTER_TRANSFER/ADAPTER_REFITTED, separate labels |
| 10 | strict success criteria | APPLIED VERBATIM — the one cell that beat both controls was still required to reproduce; it did not |
| 11 | bidirectional + matrix | DONE — 6 cells incl. sibling, third-target, independent-lineage |
| 12–13 | multi-component + interference | DONE — singles/pairs/triple with per-component guarded gates + K-scaling study |
| 14 | calibration reported separately | DONE — gate grids + guard damage reported per arm |
| 15 | capacity control → NO EVIDENCE clause | EXERCISED — verdict NO_EVIDENCE_OF_SPECIALIZED_TRANSFER where capacity won |
| 16 | distinct labels, never merged | DONE — final_arm recorded; labels never pooled into one success number |
| 17 | ≥1 real-model experiment | DONE — gpt2→pythia-70m (REAL), honestly underdetermined |
| 18 | compatibility engine (6 classes) | DONE — EXACT/STRUCTURAL/ALIGNABLE/REFITTABLE/DISTILL/ROUTING verdicts |
| 19 | tokenizer mismatch detection | DONE — detected and blocked gpt2/pythia transplant |
| 20 | no-leak splits | DONE — train draws 200–209, eval 1–3, screen 1234–1236, modular train/heldout disjoint (unit-tested) |
| 21 | statistical reporting | DONE — n, seeds, std, per-seed, sig floor max(5, 2σ); INSUFFICIENT_EVIDENCE tier used |
| 22 | transfer CLI + GUI | DONE — `cli.py transfer …` + Transfer Lab in the dashboard (live component map, run controls, ladder table) |
| 23 | TRANSFER_STATUS never PROVEN | DONE — enum lacks PROVEN by design; REPRODUCED reserved; nothing earned it |
| 24 | component library (metadata only in release) | DONE — fusionlab_data/library/* + components/*.npz (external to release zip) |
| 25–26 | automated selection / evolution after validation | CORRECTLY DORMANT — no validated transfer to automate (the spec forbids building these on unvalidated claims) |
| 27 | one TRANSFERRED_CAPABILITY_COMPONENT artifact | DONE — library record math_head_205d6f: full evidence chain, final status FAILED after reproduction (the honest artifact) |
| 28/39 | no super-model work; priority order | RESPECTED |
| 29–30 | GUI Transfer Lab + data-connected map | DONE — /api/components (library+lineage+matrix+headline), live run panel |
| 31 | structured failures | DONE — FAILURE_ANALYSIS.md F1–F8 + per-arm failure records in every run dict |
| 32 | provenance fields | DONE — source model/revision, arch, shapes, seeds, suite, n, timestamps |
| 33 | test suite | DONE — 40/40 (15 M2-era core + 25 transfer/adapter/suites) |
| 34 | release size check | DONE — tools/check_release_size.py prints PASS/FAIL, exit codes |
| 35 | report sections | THIS DOCUMENT |
| 36 | honesty rules | APPLIED — SYNTHETIC/REAL labeled; NOT EXECUTED tags where applicable; failed runs preserved |
| 37 | claim levels | L1–L3 confirmed (as M2); **L4 claimed, tested, and retracted**; L5–L7 not attempted |
| 38 | checklist | BELOW |
| 40 | eight artifacts | ALL PRESENT (list below) |

## 3. Method (what actually ran)

1. **Lineage construction** (deficit+skill by construction): nomath-base
   (150 ep, zero math) → A-math/A-code/A-reason by same-init fine-tunes on
   micro-format train-draw items (seeds 200–209; eval seeds 1–3 and screen
   seeds 1234–1236 untouched). Specialist gaps: math +32.5, reason +23.3,
   code +0.0 (ceiling — reported).
2. **Identification**: genome ablation scan per source + training-delta
   ranking vs the same-init base; merge; **specificity screen** (cross-suite
   forward ablation, specificity = target_drop − max_other_drop).
3. **Transfer ladder**: DIRECT → ALIGNED → REFITTED (masked-gradient, exact
   param counts) → CALIBRATED (residual gate) → ADAPTER (task arithmetic with
   guarded gate) → ADAPTER_REFITTED; controls: RANDOM_HARVEST (same-size
   random-init component) and CAPACITY (param-matched trained adapter).
4. **Held-out evaluation**: micro suites, n=40, 3 seeds, per-seed records,
   significance floor max(5, 2σ); non-target vector for interference (Test F).
5. **Matrix + reproduction**: 6 cells (incl. bidirectional, sibling target,
   third target, independent lineage); every positive/borderline cell re-run
   with the full seed set.
6. **Real models**: gpt2 → pythia-70m compatibility + alignment (REAL).

## 4. Results

- Headline (A-math→nomath-base): best system CALIBRATED 25.0 vs baseline
  20.8 (+4.17 ± 7.1) — below floor (14.14), below random (33.3) and capacity
  (30.8) controls, inconsistent across seeds, non-target regressions
  (code −29.2, lang −18.0, know −34.7) → **NO_EVIDENCE_OF_SPECIALIZED_
  TRANSFER+INTERFERENCE**.
- Matrix: 6 cells → 2 INSUFFICIENT_EVIDENCE, 3 NO_TRANSFER(±INTERFERENCE),
  1 SPECIALIZED_TRANSFER_EVIDENCE that **failed reproduction** (+6.25@2seeds →
  +4.16@3seeds, floor 12.48) → retracted. **Zero reproducible positives.**
- Scaling study K=1..8: flat (no component-count effect).
- Interference: guarded gates drive useless components to gate 0; no synergy.
- Interpolation bound (diagnostic, not a transfer): full-model task arithmetic
  reaches 51.7 held-out math (base 20.8) at g=1.0 with negative guard damage —
  the skill is weight-accessible but distributed; component-sparse sketches
  (≤8 groups) carry none of it measurably.
- Real models: transplant impossible (tokenizer+shape mismatches, engine
  refuses silent reshape); activation alignment underdetermined at feasible
  data scales (all fits SATURATED); honest routes = distill/routing.

## 5. What WORKS / PARTIAL / FAILED

**WORKS** (executed, tested): component records + library; extraction and
dimension-safe insertion (incl. cross-dim projected, biases as-is);
alignment suite with saturation flags; masked-gradient refitting; guarded
calibration; adapter task-arithmetic; specificity screen; training-delta
nominator; ladder + controls; verdict logic incl. INSUFFICIENT tier and
interference suffix; transfer CLI; GUI Transfer Lab; reproduction harness;
no-leak split discipline; size guard.

**PARTIAL**: real-model experiment — pipeline executes end-to-end and detects
every mismatch, but the alignment fit is data-limited (underdetermined), so no
alignment-quality claim is possible at this scale.

**FAILED** (preserved with mechanisms): single/few-component specialized
transfer under strict criteria (the central question); replacement-based
transfer (destroys host function); unguarded calibration; 2-seed evidence
(did not reproduce); hot fine-tune recipes (catastrophic forgetting).

**TESTED**: 40/40 unit tests; 7 transfer runs in the canonical pass + 3
reproduction runs; 7 fine-tune recipes; 3 real-model alignment fits.

**SYNTHETIC**: every model and number in the transfer program.
**REAL**: gpt2/pythia-70m weights, tokenizers, activations (report only; never
bundled).

## 6. Required checklist (§38)

- [x] Held-out data never used by any optimizer (train/calib/probe splits
      disjoint from eval seeds; modular train/heldout value ranges disjoint).
- [x] Every component has measured baseline/ablation/restoration evidence.
- [x] Random-harvest and parameter-matched capacity controls in every run.
- [x] Every transfer label kept distinct; no merged success number.
- [x] TRANSFER_STATUS vocabulary without PROVEN; nothing marked REPRODUCED.
- [x] Failed experiments preserved (FAILURE_ANALYSIS.md + experiment DB).
- [x] SYNTHETIC/REAL labels everywhere; no unlabeled mixing.
- [x] No silent reshape/truncate/pad (DimensionMismatchError; recorded
      transforms only).
- [x] Saturated fits never claimed as perfect alignment (flags fired).
- [x] Release ≤120 MB; real weights excluded; size check prints PASS.
- [x] Research question answered explicitly: **NO** (see §1).

## 7. Artifacts (§40)

1. `MILESTONE_3_REPORT.md` (this file)
2. `demo/reports/TRANSFER_MATRIX.md` (+ .json)
3. `demo/reports/COMPONENT_TRANSFER_RESULTS.md` (+ TRANSFER_HEADLINE.txt,
   interference_matrix.json, interpolation_bound.json, REPRODUCTION.json)
4. `demo/reports/CAPABILITY_GENOME_UPDATE.md`
5. `demo/reports/FAILURE_ANALYSIS.md`
6. `demo/reports/REAL_MODEL_TEST_REPORT.md` (+ real_model_alignment.json)
7. `RELEASE_SIZE_REPORT.txt`
8. `MILESTONE_3_AUDIT.md` (pre-run audit) + `fusionlab_data/experiments.jsonl`
   (component-transfer EXP-00737…EXP-00891: 73 runs) + component library
   (47 metadata / 81 tensor stores)

## 8. Claim level (§37)

- L1–L3 (architecture/gap/identification): **achieved and tested**.
- L4 (validated specialized component transfer): **attempted under the
  strictest criteria; the single qualifying observation did not reproduce and
  was retracted → not claimed**.
- L5–L7 (bidirectional/multi/evolutionary): not attempted — the spec forbids
  building them on unvalidated transfer.
- Real-model cross-architecture transplant: NOT POSSIBLE under no-silent-
  reshape guarantees; alignment underdetermined at this scale.

## 9. Honest conclusion

The engine can find, align, refit, calibrate, insert, control, and audit — and
when all of that is done correctly at this scale, the data say a micro-model's
capability is not localized in transferable components: the ONE honest way to
move the skill in weights is whole-model interpolation, which is merging, not
component transfer. That negative result is delivered with the full evidence
chain the spec demands, and the platform is ready to re-run the same gauntlet
on larger models where component locality may actually exist (config exposed,
`--quick` off, §17 YAML).
