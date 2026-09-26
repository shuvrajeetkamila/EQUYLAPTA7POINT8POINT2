# Complete Review — `shuvrajeetkamila/EQUYLAPTA7POINT8POINT2` by arena.ai agent

**Reviewed**: 2026-09-26 · Cloned at commit `c22330b` ("Uploading all files from old laptop", 1 commit, 542 tracked files, 9.2 MB)
**Method**: full tree walk, code reading of the core pipeline, execution of every test suite and reproduction harness, independent cross-checks of reports vs machine-readable results, secret scan.

---

## 1. What this repository is

An AI-research workspace dump documenting a program called **EQUYLAPTA** — a series of experiments (Milestones 2 → E4 → E5 → E6 → 6.5 → E7.0–E7.6 → 7.7 → 7.8 → 7.8.1 → **7.8.2**, dated 2026-09-19 … 2026-09-26) investigating one question:

> *Can a useful computation ("functional circuit") discovered inside Model A be surgically extracted, translated across architectures, and transplanted into Model B so that Model B gains that capability — beyond what matched random parameters achieve?*

Everything is built on **tiny synthetic transformers** (2–8 layers, d=48–96, ~100k–480k params) trained in-house on templated arithmetic/code/reasoning corpora (`ai-model-fusion-lab`), evaluated on **20-item multiple-choice micro-suites** (1 item = 5 pp of accuracy). Experiments progress donor d=64 ("Family A") → recipient d=96 ("Family B") cross-architecture slot translation with frozen donor weights and trainable `W_in`/`W_out` adapters, causal ablations, path/activation patching, shuffled/random/capacity-matched controls, 5-seed replication, depth and circuit-size sweeps.

**Top-level contents**:
- `ai-model-fusion-lab/` — the actual Python library (numpy micro-GPT, benchmarking, transfer/merge/distill tools, pattern-genome, CLI, web dashboard, Apache-2.0 + licensing engine). ~2,300 LOC core + demos.
- `EQUYLAPTA7POINT7/`, `EQUYLAPTA7POINT8/`, `EQUYLAPTA7POINT8_1/`, `EQUYLAPTA7POINT8_2/` — self-contained milestone packages (`run_*.py`, `src/`, `tests/`, `reproduce.sh`, reports, JSON registries).
- Root: ~150 result/registry JSONs, 20+ reports (.md and .txt duplicates), audit docs (`audit_7_8_1.md`, `audit_7_8_2.md`, `CORRECTION_LOG.md`, `E7_BASELINE_AUDIT.json`, …), leftover validation scripts and duplicate copies of everything.
- Total: 223 Python files / ~45,800 LOC.

---

## 2. Headline verdicts (mine, from actually running things)

| Area | Verdict |
|---|---|
| Code quality of `ai-model-fusion-lab` | **Good** — clean, dependency-light (numpy only), seeded/deterministic, explicit backprop with gradcheck tests, no `eval`/`exec`/shell-injection, no secrets found |
| Security / secrets | **Clean** — no credentials, no dangerous patterns; dashboard is stdlib-only local server |
| Licensing | **OK** — Apache-2.0 for lab source, thoughtful note that produced models inherit source-model licenses |
| Reproducibility (fresh machine) | **Broken** — `reproduce.sh` fails at the first step; required model artifacts are gitignored **and the code that builds half of them does not exist in the repo** |
| Test suites (as shipped) | **Mostly pass when data present; 6/8 and 8/85 fail here** purely due to missing `fusionlab_data/` (see below) |
| Internal consistency of *committed* 7.8.2 results vs its tables | **Good** — `validate_report.py` passes; my own spot-checks of tables vs `results.json` match |
| Internal consistency of 7.8.2 *narrative* (Q&A, claim_provenance) | **Poor** — hardcoded stale numbers from 7.8.1, false adjectives, "verified:true" claims that contradict `results.json` (details in §5) |
| Scientific honesty | **Unusually high for the era, but incomplete** — the program openly reports its own FAIL verdict; however the same narrative-vs-data disease its audits keep diagnosing is still present in the newest report |
| Bottom-line science | A **honest negative result**: causal activity confirmed (Level C), capability transfer NOT demonstrated (25% donor agreement vs 90% threshold; random control beats transplant on held-out accuracy). Consistent with the repo's own GitHub description: *v6.5 PASS (Level 3, transfer not demonstrated), v7.8.2 FAIL*. |

---

## 3. What I executed

| Command | Result |
|---|---|
| `python3 run_7_8_2.py` (master harness, 7.8.2) | **FAILS** at Phase 1: `FileNotFoundError: fusionlab_data/models/e4-math-4L.json` |
| `pytest EQUYLAPTA7POINT8_2/tests` | **2 passed, 6 failed** — all 6 failures = missing `fusionlab_data` models (`test_data_split`, `test_report_consistency` pass) |
| `pytest EQUYLAPTA7POINT8_1/tests` (smoke) | **2 passed, 3 failed** — same missing-data cause |
| `validate_report.py` (7.8.2 report vs results.json) | **PASS** — "zero discrepancies" (but see §5 for what it misses) |
| `pytest ai-model-fusion-lab/tests` | **69 passed, 8 skipped, 8 failed** — 1 failure missing `math-wiz` model; **7 failures are version drift**: root `results.json` was overwritten by the 7.8.2 run, so older-milestone tests (`test_e7_audit`, `test_e6_5_audit`) assert `milestone == "EQUYLAPTA7"` against an `"EQUYLAPTA_7.8.2"` file |
| Training benchmark (numpy, 4L/64d, 2600 seqs) | ~11.7 s/epoch → canonical regeneration of all needed models ≈ **1–2 hours** of compute (and still impossible for the e6 family, see §4) |
| Secret/PII scan across all text files | **Clean** |
| Byte-identical duplicate check | Root reports/`results.json` are exact copies of milestone-folder files (`results.json` == `EQUYLAPTA7POINT8_2/results.json`), plus `correction_log (2).md`, `artifact_size_report (2).json` upload litter |

---

## 4. Critical finding — the repo cannot reproduce its own experiments

`reproduce.sh` (both root and per-milestone) promises:

> `EQUYLAPTA 7.8.2 REPRODUCTION COMPLETED SUCCESSFULLY WITH 100% EVIDENCE INTEGRITY`

On a fresh clone it dies in 1 second. Two separate causes:

1. **Artifacts excluded**: `.gitignore` excludes `fusionlab_data/` ("Large Data & Weights") — the trained micro-model checkpoints (`e4-math-4L.npz/json`, `e6-base-4L`, …) live there. Commit message says "Uploading all files from old laptop" but the data folder never made it. The committed `test_results.json` ("8/8 PASS", pytest-9.0.3, Python 3.13.14 — identical to this sandbox, amusingly) can therefore not be reproduced.

2. **Missing builder code (worse)**: the donor family `e4-base-{2,4,6,8}L` / `e4-math-{2,4,6,8}L` *can* be regenerated via `ai-model-fusion-lab/demo/run_equylapta4.py::train_sweep_pair()`. But I searched the entire tree: **nothing ever builds the recipient family `e6-base-4L`, `e6-base-c-4L`, `e6-base-{2,6,8}L`** (d=96/6-head "Family B"). `run_equylapta6.py` imports `train_micro`/`save_model` but never calls them for these names — they are only ever `load_model()`-ed. The recipe for the central "cross-architecture" models of the whole program is **not in the repository**. Even with unlimited compute, 7.8.x cannot be rerun.

Additional reproducibility nits:
- `validate_report.py` and `report_generation.py` hardcode `/home/user/EQUYLAPTA7POINT8_2/...` author-machine paths (override by args where provided).
- `measure_workspace_size("/home/user")` walks the entire home directory as a "budget" metric (the 41.89 MB/50 MB line in the GitHub description).

---

## 5. Narrative-vs-data defects in the newest milestone (7.8.2)

7.8.2's charter is *"Elimination of Narrative vs. Empirical Discrepancies — every prose claim derived mechanically from machine-readable results."* The tables/sections 9–19 indeed interpolate from `results.json` and check out. But the generator (`EQUYLAPTA7POINT8_2/src/report_generation.py`) still contains **hardcoded literals from the 7.8.1 run**, and the validator is pattern-based and misses them. Concretely (all verified against `EQUYLAPTA7POINT8_2/results.json`):

| # | Where | Claim in report | Actual `results.json` |
|---|---|---|---|
| 1 | §1.3 & Q10 & §17 | Condition E held-out accuracy "30.0% vs baseline 35.0%" | **40.0%** vs 35.0% (`heldout_evaluations.Condition_E…`) |
| 2 | Q10 & `claim_provenance.json` CLAIM_5 | Random control "exhibits a negative causal drop (**-8.0 pp**)" | Condition F held-out causal drop = **+20.0 pp** (not −8.0; that figure matches *nothing* in 7.8.1 or 7.8.2 results) |
| 3 | Q8 & `claim_provenance.json` CLAIM_3 ("verified": true) | 8-node circuit "validation causal drop **+25.0 pp** vs **+5.0 pp** for 2 nodes" | size_sweep: 8-node **+10.0**, 2-node **+20.0** (the +25/+5 pair is 7.8.1 data) |
| 4 | Q8 | 8-node held-out "active accuracy (30.0%) and causal drop (+10.0 pp)" | **50.0%** and **+5.0 pp** |
| 5 | §16 | "…**increased** the causal drop from +20.0 pp to +10.0 pp" | it *decreased*; template hardcodes the verb "increased" |
| 6 | §16 | "…+5.0 pp (**identical** to the 2-node nucleus at +15.0 pp)" | 5.0 ≠ 15.0; the word "identical" is hardcoded in the template |
| 7 | §13 | "Condition F exhibited a **negative** causal drop (**+20.0 pp**)" | a +20 pp drop is positive; adjective contradicts the interpolated value |
| 8 | §17 & Q2 | "…a **statistically significant** +13.0 pp drop on held-out data" | the report's own 95% CI is **[−4.88, 30.88]** — contains 0; and +13.0 is the 5-seed mean, not the held-out split (+25.0) |
| 9 | §12 | "Primary Recipient Mediator: `Recipient_L1_mlp` (**+0.0 pp** causal drop). This confirms…" | `discover_recipient_pathway` picks `max()` over drops `{-5, 0, 0, 0}` → a **zero-effect** module is crowned "primary mediator"; the honest reading is "no recipient-side mediator detected" |

So the exact failure mode that `audit_7_8_2.md` diagnosed in 7.8.1 (stale template claims, selective reporting) is **still present in 7.8.2**, just in different places. `validate_report.py` ("zero discrepancies") only forbids a handful of specific legacy phrases.

Two more logical soft spots (arguable rather than flat errors):
- `criterion_5_replication` is marked **YES** while its own evidence string says `MIXED_SIGN_VARIABILITY` with a CI crossing zero.
- "Restoration error 0.0 pp" is trivially guaranteed: `restore_circuit()` just re-applies the same weights that produced the "active" state (it is not an independent re-derivation).

**Credit where due**: the machine tables, the FAIL verdict, the negative held-out activation-patching result (−10 pp), and the random-control superiority are all reported openly — including in the executive summary. For an LLM-generated research workspace, this candor is real.

---

## 6. Engineering quality notes (smaller items)

1. **Shared mutable root artifacts**: root `results.json`, `gradient_check.json`, etc. are write-targets of *each* milestone run (report generator mirrors to `/home/user/` root). Running a newer milestone silently breaks older milestones' tests (`test_e7_audit` now asserts `milestone == "EQUYLAPTA7"` on a 7.8.2 file → 7 lab-test failures).
2. **`load_model` API misuse in several tests / `gradient_validation.py`**: they pass `os.path.join(WORKSPACE, "fusionlab_data", "models", "e4-math-4L")` where the function expects a bare name. It only works because `os.path.join` discards the prefix for absolute paths — a landmine.
3. **Upload litter / duplication**: every report exists as both `.md` and `.txt`; milestone-folder copies are mirrored at root; `correction_log (2).md`, `artifact_size_report (2).json`; 20+ near-duplicate `functional_signature*.json`. A curated `history/` layout would halve the repo.
4. **Statistical power**: all headline numbers are multiples of 5 pp (n=20). Single-item flips move "causal drop" by ±5 pp; the negative seed and several sign flips are consistent with pure item noise. The report acknowledges "small validation sample sizes" (Q11) but still uses "statistically significant" language.
5. **`train_adapters` freeze-tags**: conditions whose name contains `Baseline/Donor_FDC/Random_Matched/Shuffled/Location_Matched` skip training ("FROZEN_TRANSFER_EVALUATION") — but `Condition_G3_Parameter_**Shuffle**` does *not* contain "Shuffled", so it trains (loss 11.17 in the table). Probably unintended asymmetry: G3 is the only G-control that gets adapter optimization.
6. **`run_7_8_2.py` depth sweep** requires `e4-math-{2,6,8}L` + `e6-base-{2,6,8}L` — six more missing checkpoints on top of the two mains.

---

## 7. The science, summarized fairly

- **What works**: circuit discovery + surgical slot translation with frozen donor weights is mechanically sound; ablation drops (2-node nucleus: +20 pp val / +15 pp held-out in 7.8.2), perfect restore bookkeeping, gradient code verified against finite differences (max rel. err 0.0072 — and the 7.8.1 MLP-bridge defect really is fixed in 7.8.2: all four declared adapters now receive analytical gradients with recorded update norms), strict disjoint splits (characterization/val/held-out, seeds 777/888/999, automated leakage test), pre-registered controls and thresholds.
- **What fails (and is admitted)**: donor-identity acquisition — exact agreement 25% vs pre-registered 90% ⇒ `final_classification.json: verdict = FAIL`. Donor activation patching does not generalize (+10 pp val → −10 pp held-out). Random matched injection reaches *higher* held-out accuracy (45% vs 40%) though with its own ablation sensitivity. One of five replication seeds is negative. Higher-layer donor routing (L1–L3) never engages; the recipient compresses everything into its residual highway.
- **Program arc**: E4 claimed "DEMONSTRATED AT ONE DEPTH, not solved in general" (+3.33 pp over a 21.67 baseline, with interference); E6.5's own correction log then walked back nine overclaims (agreement 38% not 95%, "outperformed" when it underperformed, non-monotonic depth labeled monotonic, …) — a genuinely creditable forensic pass; 7.7–7.8.1 showed causal activity ≠ transfer; 7.8.2 formalized the optimizer and shipped the FAIL verdict. The GitHub description matches the artifacts.
- **Scope caveat for any reader**: these are 100–500k-parameter toy transformers on templated corpora and 20-item suites. Nothing here speaks directly to transfer between real LLMs; treat it as a well-instrumented sandbox study of the *hypothesis*, whose answer so far is: *causal transplant ≠ capability transplant*.

---

## 8. Recommendations (if this work continues)

1. **Ship the data or the recipe**: commit `fusionlab_data/models/*.json/*.npz` (they are small — the report lists ~0.9 MB per model) or add `build_all_models.py` that constructs **both** e4 and e6 families (the e6 builder is currently missing). Make `reproduce.sh` invoke it when artifacts are absent.
2. **Purge hardcoded numbers from `report_generation.py`** (§5 table): interpolate every figure from `results.json`, and extend `validate_report.py` to (a) parse *all* numeric literals in prose and compare to the data, (b) validate `claim_provenance.json` claims against results (it currently blesses false claims as `"verified": true`), (c) ban "statistically significant" unless the CI excludes zero, and "identical/increased/decreased" unless computed.
3. **Fix `discover_recipient_pathway`** to return "none detected" when max drop ≤ 0.
4. **Version the result files** (`results_e7_8_2.json`, or per-milestone output dirs) so milestones stop overwriting each other's evidence; drop root-level duplicates and the `(2)` files.
5. **Raise evaluation power** (n ≥ 100 items, report binomial CIs per condition) before drawing sign conclusions from ±5 pp moves; or pre-register a paired item-level test.
6. Minor: fix `load_model` path misuse in tests, parameterize `/home/user/...` hardcodes, decide whether G3 should train.

---

## 9. One-paragraph bottom line

This is a small, clean, surprisingly self-critical ML-research workspace: a well-built numpy micro-transformer lab plus a chain of "EQUYLAPTA" transfer experiments whose central claim — plug-and-play cross-architecture capability transfer — **the repo itself ultimately disproves** (Level C causal activity only, FAIL vs its own 90% criterion, and honest reporting that a random control outperforms the transplant on raw accuracy). The code is safe and tidy and the committed 7.8.2 numbers are internally consistent in tabular form, but the project is **not reproducible from GitHub** (gitignored checkpoints plus a genuinely missing model-builder), and the newest report still repeats the exact sin its own audits keep catching — hardcoded stale figures in the narrative/Q&A and "verified" provenance claims that contradict its own `results.json`. Fix those two classes of problems and this becomes a credible (if toy-scale) negative-result package.
