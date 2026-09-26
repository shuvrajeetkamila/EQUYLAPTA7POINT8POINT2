# EQUYLAPTA 7.8.2 CRITICAL CODEBASE & EXPERIMENTAL AUDIT
## Thorough Verification of EQUYLAPTA 7.8.1 Implementation & Corrective Mandate

**Date**: 2026-09-26  
**Auditor**: Senior AI Research Engineering Lead  
**Scope**: Code-level and evidence-integrity audit of all components in `EQUYLAPTA7POINT8_1`.

---

## 1. What EQUYLAPTA 7.8.1 Successfully Corrected

1. **Eliminated Handcrafted Gradient Proxy**: Successfully removed the synthetic update proxy `(p * 0.005) + ...` from `architecture_translator.py` and implemented analytical two-branch backpropagation for the attention head bridge.
2. **Two-Branch Loss Formalization**: Correctly formalized the mathematical objective $L_{\text{functional}} = 0.5 \cdot \text{mean}( ((Y_{\text{intact}} - Y_{\text{ablated}}) - \Delta_{\text{target}})^2 )$.
3. **Automated Leakage Detection**: Built `tests/test_leakage.py` and implemented prompt deduplication across characterization (`seed=777`), validation (`seed=888`), and held-out (`seed=999`) splits, eliminating the 3-item overlap from 7.8.
4. **Pre-Registered Structured Controls**: Replaced the flawed 7.8 shuffled control with distinct categories in `control_registry.json` (Shuffled-Node, Shuffled-Path, Parameter-Shuffle, Location-Matched).

---

## 2. What Remains Incorrect in 7.8.1

Despite significant progress in 7.8.1, deep code-level inspection reveals severe structural and narrative discrepancies:

1. **Declared Trainable Parameters Not Optimized (The MLP Bridge Defect)**:
   - In `training_objective.json`, four parameters were declared trainable:
     `head_bridges.L0_head_2.W_in`, `head_bridges.L0_head_2.W_out`, `mlp_bridges.L0_mlp.W_in`, `mlp_bridges.L0_mlp.W_out`.
   - In `src/transfer.py` (lines 180–235), the training loop **only retrieved and updated `head_bridges["L0_head_2"]`**!
   - `mlp_bridges["L0_mlp"]` was never passed through backpropagation, received zero gradients, and was never updated!
2. **Gradient Validation Incomplete**:
   - `src/gradient_validation.py` only validated finite differences for `head_bridges["L0_head_2"]`. It never checked `MLPSlotBridge` parameters, leaving half the declared trainable parameters unverified.
3. **Direct Contradiction on Activation Patching**:
   - In `results.json`: `heldout_seed_999.donor_lift_over_baseline_pp = -10.0 pp` (baseline 35.0%, donor patch 25.0%).
   - In `claim_provenance.json`: Claim stated that donor activation patching improved accuracy by `+10.0 pp` on both validation and held-out test splits.
   - This was a flat contradiction: validation improved by +10 pp, but held-out dropped by -10 pp. The report must state: *"Validation improvement was not retained on held-out data."*
4. **Oversized Circuit Narrative Contradiction**:
   - In 7.8.1 report Section 18: Claimed expanding from 4 to 8 nodes provides `0.0 pp additional causal benefit`.
   - In `results.json`: `Condition_H_Oversized_Circuit` validation causal drop was `+25.0 pp`, compared to `+5.0 pp` for the 2-node nucleus!
   - The narrative blindly repeated a legacy 7.7/7.8 claim that contradicted the empirical 7.8.1 data.
5. **Misleading Five-Seed Description**:
   - For `Condition_E_Recipient_Reconstructed_FDC`, the raw seed drops were `[-5.0, 10.0, 15.0, 35.0, 10.0]`.
   - Seed 9001 had a negative drop (`-5.0 pp`).
   - The report narrative repeatedly used phrasing implying consistent positive replication across all seeds, rather than acknowledging seed-level sign variability.
6. **Random Control Superiority Glossed Over**:
   - On the strictly held-out split, `Condition_F_Random_Matched` achieved an active accuracy of **45.0%**, outperforming the transplanted Condition E (**30.0%**).
   - The report focused entirely on causal drop (+13.0 pp vs -8.0 pp) to declare Condition E superior, without honestly evaluating the raw held-out capability comparison.

---

## 3. Discrepancies Between Code and Report

| Item | Actual Code / results.json | Stated in Report | Audit Verdict |
|---|---|---|---|
| **MLP Adapter Training** | Only head adapters updated; MLP bridges untouched (`dW_in=0, dW_out=0`) | Claimed all declared adapters were optimized | **CODE DEFECT** |
| **Held-Out Activation Patch Lift** | `-10.0 pp` (25.0% vs 35.0% baseline) | `+10.0 pp lift on held-out split` | **REPORT CONTRADICTION** |
| **Oversized Causal Effect** | `+25.0 pp drop` | `0.0 pp additional gain` | **REPORT CONTRADICTION** |
| **Five-Seed Replication** | 1 out of 5 seeds was negative (`-5.0 pp`) | "Replicated across seeds" without disclosing negative seed | **REPORT OMISSION** |
| **Random Control Held-Out Acc** | 45.0% active accuracy | Focused solely on causal drop (-8.0 pp) | **SELECTIVE REPORTING** |

---

## 4. Declared Parameters Not Actually Optimized

- `mlp_bridges.L0_mlp.W_in` (Shape: [96, 64], 6,144 parameters): Declared trainable, received NO gradient, parameter change = 0.0.
- `mlp_bridges.L0_mlp.W_out` (Shape: [64, 96], 6,144 parameters): Declared trainable, received NO gradient, parameter change = 0.0.

---

## 5. Required Corrective Actions in EQUYLAPTA 7.8.2

1. **Implement True Multi-Branch Optimization**: Derive and execute exact analytical two-branch gradients for both `head_bridges` AND `mlp_bridges` simultaneously.
2. **Audit Parameter Updates**: Create `adapter_update_audit.json` tracking initial norm, final norm, update norm, and gradient existence for every declared trainable parameter.
3. **Automated Trainable Parameter Consistency Test**: Implement `tests/test_trainable_parameter_consistency.py` requiring 100% agreement between Declared, Optimizer, Gradient-Bearing, and Changed parameters.
4. **Complete Finite-Difference Gradient Validation**: Verify analytical gradients against finite differences for both head bridge AND MLP bridge (minimum 10 coordinates each).
5. **Truthful Activation Patch Reporting**: Implement automated prose generation based strictly on machine-readable deltas, explicitly stating when held-out effects fail to replicate validation gains.
6. **Pre-Registered Replication Criteria**: Define `replication_criteria.json` prior to evaluation and classify seed results mechanically.
7. **Rigorous Random Competitor Treatment**: Honestly report both capability score and causal drop for Random Matched and Shuffled controls.
8. **Automated Classification Engine**: Create `src/classification_engine.py` to generate `final_classification.json` deterministically from evidence rules.
