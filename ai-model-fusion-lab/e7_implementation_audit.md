# EQUYLAPTA7 Implementation Audit Report

## Forensic Review of E7 Experimental Code, Claims, and Missing Scientific Mechanics

This audit documents the critical implementation defects identified in the initial EQUYLAPTA7 draft, establishing the required corrections for **EQUYLAPTA7.1**.

---

### Audit Register Summary

| ID | Module / Function | Claimed Behavior | Actual Code Execution | Discrepancy Severity | Required E7.1 Correction |
|:---|:---|:---|:---|:---:|:---|
| **AUDIT-01** | `reconstruct_method_b_subspace` | Functional signature reconstruction using extracted PCA subspace basis and covariance. | Generated random normal noise scaled by error: `grad[:16,:16] += err * rng.normal()`. Ignored `top_subspace_basis`, covariance, and eigenvalues. | **CRITICAL** | Implement `METHOD_B_FUNCTIONAL_SIGNATURE_RECONSTRUCTION_V2`: Align target-space projections with the extracted functional subspace basis using Procrustes/least-squares; enforce hard assertion that basis is consumed. |
| **AUDIT-02** | `reconstruct_method_c_distillation` | Behavioral/task distillation from source teacher into target student circuit. | Added random Gaussian noise: `weights + rng.normal(0, 0.005)`. Zero teacher evaluation, zero loss, zero gradients. | **CRITICAL** | Implement `METHOD_C_BEHAVIORAL_DISTILLATION_V2`: Genuine gradient descent optimizing target-native student parameters to match teacher soft targets; save loss curves and parameter deltas. |
| **AUDIT-03** | Target Causal Ablation (`main`) | Target ablation tests causal dependence of the reconstructed functional unit. | Modified QKV slice `[:16, :16]` but blindly set `tgt_b.head_mask[0, 0] = 0` (ablating Head 0) without ensuring Head 0 was the modified unit. | **CRITICAL** | Define `target_functional_unit_id` explicitly (Target Layer 0 Head 0); parameterize that exact head, and ablate/restore that exact unit. |
| **AUDIT-04** | `run_independent_reconstructions` | 5 independent reconstructions initialized from fresh target models converge. | Executed loop on single target model instance with superficial seed variations, producing artificial 100% agreement and Linear CKA = 1.0000. | **CRITICAL** | Instantiate fresh target models per seed, compute SHA-256 target initialization hashes to prove independence, execute genuine training, and report real convergence metrics. |
| **AUDIT-05** | `extract_e7_functional_signature` | Head-specific functional signature for causal unit `L0_head_2`. | Collected `acts['hiddens'][0][0, -1]`, capturing the whole Layer 0 residual state rather than Head 2. | **HIGH** | Extract activations specifically from Head 2 (`e=16`), along with its attention routing matrix and a 10-point intervention strength curve (-1.0 to +1.25). |
| **AUDIT-06** | `run_target_controls` (Control 3) | Control 3 is a genuinely target-local trained component. | Added random noise: `v + rng.normal(0, 0.008)`. Zero training steps were executed. | **CRITICAL** | Genuinely train target-local control on target training texts for the identical step budget (30 steps). |
| **AUDIT-07** | `run_depth_sweep_e7` | Independent measurement of transfer across 2L, 4L, 6L, 8L. | Evaluated to degenerate `[0.0, 0.0, 0.0, 0.0]` pp because reconstructed payloads were not correctly connected to evaluation forward passes. | **HIGH** | Properly inject reconstructed parameters into each depth model, evaluate baseline vs reconstructed vs ablated, and measure real depth degradation slope. |
| **AUDIT-08** | `run_minimality_curve` | Minimal functional unit discovery. | Scaled head scalar magnitude fraction from 1.0 down to 0.01 (magnitude scaling), not parameter subset selection. | **MEDIUM** | Label magnitude scaling as intervention-strength curve, and evaluate parameter subsets and activation subspace ranks. |

---

### Non-Negotiable Correction Principles for EQUYLAPTA7.1

1. **No Pseudo-Optimization:** Every method claiming to train or reconstruct must perform real forward/backward passes and save actual loss curves and parameter updates.
2. **Hard Assertions on Basis Consumption:** Method B must mathematically fail if the functional signature's `top_subspace_basis` is not consumed.
3. **True Independent Reconstructions:** Target models must be freshly loaded with verified distinct SHA-256 initialization hashes.
4. **Exact Target-Unit Alignment:** The parameters modified by the reconstruction must be the exact parameters ablated and restored during causal testing.
5. **Truthful Scientific Reporting:** If transfer fails or scores remain below target-local controls, the report must state that without equivocation.
