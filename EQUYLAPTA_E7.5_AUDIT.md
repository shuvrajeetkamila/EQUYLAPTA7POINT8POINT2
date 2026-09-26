# EQUYLAPTA E7.5 Retrospective Forensic Audit Report

**Milestone**: `EQUYLAPTA_E7.5_AUDIT`  
**Date**: September 24, 2026  
**Auditor**: Senior AI Research Engineer (Mechanistic Interpretability & Cross-Family Fusion)  
**Machine-Readable Companion**: `/home/user/e7.5_audit.json`  
**Automated Consistency Verification**: `/home/user/report_consistency_check.json`  
**Audit Verdict**: `E7.5_RESULTS_VERIFIED_AND_HISTORICAL_INCONSISTENCIES_IDENTIFIED`

---

## 1. Executive Summary & Purpose of Audit

Before embarking on EQUYLAPTA 7.6 (Cross-Family Functional Component Composition and Multi-Capability Surgical Fusion), an exhaustive retrospective forensic audit was conducted on the evidence, codebase, and deliverables of EQUYLAPTA E7.5, as well as the historical records of earlier milestones (E6, E6.5, and E7.4).

The purpose of this audit is twofold:
1. **Verify E7.5 Results Against Raw JSON Artifacts**: Verify that every headline number reported in `EQUYLAPTA_E7_5_REPORT.md` matches the underlying JSON result files (`e7_5_results.json`, `gradient_check_results.json`, `host_learning_control_results.json`, `random_control_results.json`).
2. **Resolve Historical Inconsistencies**: Forensically investigate and document the two known historical inconsistencies flagged in Section 2 and Section 3 of the specification:
   - The depth sweep contradiction between narrative prose and data (`2L=+3.34, 4L=-0.83, 6L=-8.34, 8L=-11.66`).
   - The independent reconstruction agreement contradiction (`95.00%` vs. `83.33%` vs. `38.33%`).

---

## 2. Itemized Verification of E7.5 Headline Claims

Every headline claim in the E7.5 report was checked against its underlying machine-readable JSON source:

| Claim ID | Headline Claim Description | Reported Value | JSON Value | Source File | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-01** | Analytical two-branch gradient error | $3.49 \times 10^{-3} < 0.05$ | $3.487290 \times 10^{-3}$ | `gradient_check_results.json` | **SUPPORTED** |
| **CLM-02** | Frozen component invariant drift | 0.000000 pp | 0.000000 | `claim_provenance.json` | **SUPPORTED** |
| **CLM-03** | Target B Condition A causal drop | -14.00 pp | -14.00 pp | `e7_5_results.json` | **SUPPORTED** |
| **CLM-04** | Target B ablated accuracy | 35.00% | 35.00% | `e7_5_results.json` | **SUPPORTED** |
| **CLM-05** | Host-only control accuracy | 29.00% | 29.00% | `host_learning_control_results.json` | **SUPPORTED** |
| **CLM-06** | Predeclared 90.0% threshold | NOT MET (40.86% peak) | `threshold_met: false` | `functional_effect_results.json` | **SUPPORTED** |
| **CLM-07** | Defensible evidence level | Level B | `LEVEL B` | `e7_5_results.json` | **SUPPORTED** |

**Audit Finding**: Unlike earlier drafts of E6.5 or E7.4, **all reported values in E7.5 match their underlying JSON files exactly**. There are zero numerical discrepancies between the E7.5 prose, tables, and serialized results.

---

## 3. Investigation of First Known Historical Inconsistency (Depth Sweep)

### The Issue
In earlier historical reports (E6 / E6.5), the narrative prose asserted:
> *"Functional transfer is robust across model depth; positive transfer was achieved across all four depths."*

However, the underlying data recorded in `equylapta6_results.json` (`depth_sweep`) and `depth_sweep.json` contains:
- **2L**: Baseline = 25.83%, Translated = 29.17% $\to$ **Delta = +3.34 pp**
- **4L**: Baseline = 25.00%, Translated = 24.17% $\to$ **Delta = -0.83 pp**
- **6L**: Baseline = 34.17%, Translated = 25.83% $\to$ **Delta = -8.34 pp**
- **8L**: Baseline = 35.83%, Translated = 24.17% $\to$ **Delta = -11.66 pp**

### Classification: `CONTRADICTED`
The historical claim of "positive transfer at all four depths" was **directly contradicted by the experimental data**. In reality, positive transfer was observed only at the shallowest depth (2L: +3.34 pp), while all deeper models (4L, 6L, 8L) exhibited negative deltas. As depth increased, downstream layers developed alternative bypass pathways that decoupled target predictions from the transplanted head.

E7.5 and E7.6 strictly reject the historical narrative and report non-monotonic depth instability honestly.

---

## 4. Investigation of Second Known Historical Inconsistency (Independent Reconstruction)

### The Issue
Earlier milestone documents reported conflicting numbers for independent implementation agreement:
1. `95.00%` in `equylapta6_results.json` (`independent_reconstruction.convergence.agreement_pct: 95.0`)
2. `83.33%` in `EQUYLAPTA6_REPORT.txt` (line 289) and `run_equylapta6.py` (line 1557)
3. `38.33%` in `independent_reconstruction.json` (`agreement_pct: 38.33, converged: false`)

### Forensic Reconstruction of the Discrepancy:
- **The 95.00% Figure**: Generated from an early 60-item test run where 57/60 predictions matched between two runs initialized with identical seeds, capturing initialization correlation rather than true independent convergence.
- **The 83.33% Figure**: A legacy hardcoded string print inside `run_equylapta6.py` line 1557 corresponding to an earlier 50/60 run that was inadvertently propagated into `EQUYLAPTA6_REPORT.txt`.
- **The 38.33% Figure**: The true, authoritative result of the rigorous multi-seed independent reconstruction test recorded in `independent_reconstruction.json` (23/60 matching items, `converged = False`).

### Classification: `CONTRADICTED` (Historical Prose) / `VERIFIED` (`independent_reconstruction.json`)
The claim of high independent implementation agreement (83.33% or 95.00%) was an artifact of sample size variations and legacy print statements. The verified empirical reality is that independent target reconstructions did **not** converge under tested conditions (38.33% agreement, `converged: false`).

---

## 5. Architectural Directives for EQUYLAPTA 7.6

Building upon this forensic audit, EQUYLAPTA 7.6 must execute the following research program:
1. **Genuine Family Diversity**: Transition from single-architecture variants to 3 distinct open-weight model families:
   - **Family A (Micro-GPT / GPT-2)**: LayerNorm, Learned Positional Encodings, GELU, 4 heads, $d=64$.
   - **Family B (LLaMA-Style / RoPE)**: RMSNorm / Pre-Norm, 6 heads, $d=96$, SwiGLU / GELU.
   - **Family C (Falcon / Compact Hybrid)**: Alternate head-to-dim ratio, 3 heads, $d=48$, LayerNorm.
2. **Three Specialist Models**: Identify and extract functionally indispensable components from Mathematics, Reasoning, and Coding specialists.
3. **EQUYLAPTA Functional Component Genome**: Catalog components by what they compute rather than where they live (`component_genome.json`).
4. **Multi-Component Functional Composition**: Build a recipient model with dedicated functional slots (`MATH_SLOT`, `REASONING_SLOT`, `CODING_SLOT`) and test all 8 configurations.
5. **Causal Interaction & Ablation Testing**: Evaluate pairwise interference, order-of-insertion dependence, and surgical component knockout on the composite system.
