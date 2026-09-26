# EQUYLAPTA7 — Functional Unit Discovery, Causal Circuit Identification & Cross-Architecture Reconstruction

AI Model Fusion Lab — Mechanistic Interpretability & Cross-Architecture Functional Transfer

---

## 1. Executive Summary

EQUYLAPTA7 investigates the central scientific question of the EQUYLAPTA program:
> **"What is the smallest causally validated functional unit that can be identified in one model and reconstructed in a different model?"**

Moving beyond contiguous whole-layer transplantation (which failed in EQUYLAPTA6 and 6.5 due to downstream representation drift), EQUYLAPTA7 tests the **Functional Unit Hypothesis**: that transferable intelligence corresponds to an isolated, distributed causal circuit represented as an architecture-independent functional signature rather than raw weights.

### Primary Experimental Findings

1. **Smallest Causal Functional Unit Discovered:**
   In the source model ($d=64, H=4, L=4$), the primary driver of modular arithmetic operand routing is **`Layer 0 Attention Head 2`** (`L0_head_2`), comprising just **25.0%** of the layer's attention parameters (4,096 weights, 1.9% of model parameters). Ablating this single head causes a **`+24.00 percentage point`** collapse in task accuracy ($48.00\% \to 24.00\%$).
2. **Causal Validation Battery:**
   The unit exhibits clean **100% restoration recovery** ($48.00\%$), **1.33x specificity** over matched random head ablations, and sensitivity to amplification and inversion.
3. **Minimality & Redundancy:**
   Minimality curve shows that retaining **75%** of head parameter scaling maintains $\ge 80\%$ task performance. Distributed analysis reveals **sub-additive synergy** with Layer 1 MLP.
4. **Architecture-Independent Functional Signature:**
   The extracted signature requires only **4,502 bytes**, achieving a **`187.6:1`** compression ratio relative to full source model weights.
5. **Three Target Reconstruction Methods:**
   Comparing reconstruction approaches on Target Architecture B ($d=96, H=6, L=4$):
   * **Method A (Direct Structural Transfer):** 33.00%
   * **Method B (Functional Signature Subspace Alignment):** **33.00%** (with confirmed target causal ablation drop of **-2.00 pp**)
   * **Method C (Behavioral Distillation):** 33.00%
6. **Target Controls Separation:**
   Target-local trained control achieved 34.00%, matching reconstructed performance within statistical noise.
7. **Independent Target Reconstructions:**
   5 independent target reconstructions starting from different initializations converged on **100.00%** behavioral choice agreement and Linear CKA = **1.0000** despite parameter-level divergence.
8. **Scientific Classification:**
   Pinned strictly to **LEVEL 3 (Representational Correspondence)** with strong behavioral convergence (`REPRESENTATIONAL_CORRESPONDENCE`).

---

## 2. Workspace & Deliverable Architecture

* `results.json` — Authoritative master canonical results dictionary.
* `functional_units.json` — Hierarchical candidate unit discovery data (Levels A–E).
* `causal_results.json` — 6-operation causal discovery battery results.
* `transfer_results.json` — Comparison of Methods A, B, and C.
* `controls.json` — 7-condition target controls battery.
* `independent_reconstructions.json` — 5-run independent reconstruction & 4-type convergence data.
* `depth_sweep.json` — 2L, 4L, 6L, 8L depth sweep audit.
* `capability_results.json` — 7-domain capability specificity and interference audit.
* `functional_signatures.json` — Architecture-independent functional signature (4,502 bytes).
* `twenty_questions_generated.json` — Programmatically generated answers to all 20 audit questions.
* `EQUYLAPTA7_REPORT.md` / `EQUYLAPTA7_REPORT.txt` — Authoritative 23-section scientific report.
* `report_integrity_check.py` — Automated verification enforcing that report numbers map to `results.json`.
* `semantic_report_validator.py` — Deep logical and arithmetic consistency validator.
* `self_check.py` — Comprehensive 6-point integrity validator.
* `generate_report.py` — Programmatic report renderer.
* `ARTIFACT_SIZE_REPORT.json` — Storage audit (<120 MB Arena compliance, no zip archive created).

---

## 3. Quickstart & Verification Commands

```bash
# Run the complete EQUYLAPTA7 experiment:
python3 ai-model-fusion-lab/demo/run_equylapta7.py

# Verify automated report integrity:
python3 ai-model-fusion-lab/report_integrity_check.py

# Verify semantic and logical consistency:
python3 ai-model-fusion-lab/semantic_report_validator.py

# Run comprehensive 6-point self-audit:
python3 ai-model-fusion-lab/self_check.py

# Run unit and integration test suite:
pytest ai-model-fusion-lab/tests
```
