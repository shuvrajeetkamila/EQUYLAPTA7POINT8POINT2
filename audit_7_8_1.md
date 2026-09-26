# EQUYLAPTA 7.8.1 CRITICAL CODEBASE AUDIT
## Comprehensive Audit of the EQUYLAPTA 7.8 Experimental Implementation

**Date**: 2026-09-26  
**Auditor**: Senior AI Research Engineering Lead  
**Scope**: Verification of all implementation components, objectives, gradient routines, controls, and report assertions in `EQUYLAPTA7POINT8`.

---

## Executive Audit Finding

The implementation in `EQUYLAPTA7POINT8` represented major conceptual progress in establishing the **Non-Equivalence Principle (Donor FDC != Recipient FDC)**, exhaustive subset evaluations, and Pearlian mediation analysis. However, a rigorous code-level audit reveals **several critical methodological and mathematical discrepancies** between the stated scientific claims and the actual executed codebase. 

Most critically:
1. **The Adapter Training Loop Did Not Optimize the Functional Objective**: In `architecture_translator.py`, line 236, the adapter parameters were updated using a handcrafted heuristic gradient proxy `grad = (p * 0.005) + (p_dist[target_idx] - 1.0) * 0.001` rather than backpropagating the two-branch functional causal loss through the model!
2. **Disconnection Between Gradient Check and Actual Training**: While `validation/gradient_check.py` accurately verified that analytical two-branch gradients match finite differences on a synthetic dummy model, this analytical backward pass was **never wired into the actual adapter training loop**.
3. **Loss Function Conflation (Output Imitation vs Causal Effect Matching)**: The training loss evaluated cross-entropy on the intact branch alone (`loss = -log(p_dist[target])`), rather than penalizing the discrepancy between the recipient causal effect ($F_{\text{intact}} - F_{\text{ablated}}$) and the donor causal signature ($\Delta_{\text{target}}$).
4. **Shuffled Control Flaw**: Condition H ("Shuffled Dependency") was implemented by assigning donor heads into recipient slots 5, 4, 3, 2, thereby directly overwriting native recipient attention heads in Layer 0 without isolating specific topological disruptions (shuffled-node vs shuffled-path vs parameter-shuffle).
5. **Causal Activity Conflated with Capability Transfer**: In narrative summaries, positive causal drops (+5.0 pp to +13.0 pp) were repeatedly discussed as evidence of capability transfer, whereas mathematically they only demonstrated that the transplanted parameters constituted a causal bottleneck in the recipient network.

---

## Detailed Component-by-Component Audit

### 1. `transfer/architecture_translator.py` & Adapter Training
* **Claim in Report**: The adapters were trained on the analytical two-branch functional effect loss using Adam with gradient clipping.
* **Code Implementation**:
  ```python
  # From EQUYLAPTA7POINT8/transfer/architecture_translator.py lines 235-236:
  for idx, p in enumerate(params):
      grad = (p * 0.005) + (p_dist[target_idx % len(p_dist)] - 1.0) * 0.001
      np.clip(grad, -5.0, 5.0, out=grad)
      m[idx] = beta1 * m[idx] + (1 - beta1) * grad
      v[idx] = beta2 * v[idx] + (1 - beta2) * (grad ** 2)
  ```
* **Audit Verdict**: **SEVERE DEFECT**. This is a synthetic linear proxy, not an actual gradient of the model output or the functional objective. The parameters were essentially perturbed by a scaled error signal combined with weight decay.
* **Required Correction in 7.8.1**: Implement the true two-branch differentiable backward pass:
  $$\frac{\partial L_{\text{functional}}}{\partial Y_{\text{intact}}} = \frac{1}{V}( (Y_{\text{intact}} - Y_{\text{ablated}}) - \Delta_{\text{target}} )$$
  $$\frac{\partial L_{\text{functional}}}{\partial Y_{\text{ablated}}} = -\frac{1}{V}( (Y_{\text{intact}} - Y_{\text{ablated}}) - \Delta_{\text{target}} )$$
  Backpropagate both branches through `MicroTransformer.backward()` and apply the chain rule to update $W_{\text{in}}$ and $W_{\text{out}}$.

---

### 2. Functional-Effect Calculation & Objective Definition
* **Claim in Report**: The optimization targets the causal functional effect:
  $$L_{\text{func}} = 0.5 \cdot \text{mean}( ((Y_{\text{intact}} - Y_{\text{ablated}}) - F_{\text{src}})^2 )$$
* **Code Implementation**: In `architecture_translator.py`, line 232:
  ```python
  loss = -np.log(max(1e-12, p_dist[target_idx % len(p_dist)]))
  ```
* **Audit Verdict**: **OBJECTIVE MISALIGNMENT**. The code optimized standard cross-entropy on the task target (output imitation), rather than minimizing the distance between the recipient causal effect and the donor causal signature.
* **Required Correction in 7.8.1**: Implement the exact Mean Squared Error (or Cosine Distance) on the logit difference:
  $$L_{\text{functional}}(\theta) = \frac{1}{2V} \sum_{v=1}^V \left( (Y_{\text{intact}}(x, \theta)_v - Y_{\text{ablated}}(x, \theta)_v) - \Delta_{\text{target}}(x)_v \right)^2$$
  Document the exact formulation in `functional_objective.md` and `training_objective.json`.

---

### 3. Gradient-Check Implementation
* **Claim in Report**: Two-branch gradients were verified with maximum relative error $< 0.05$.
* **Code Implementation**: `validation/gradient_check.py` successfully computed finite differences vs analytical gradients for a dummy model `e4-base-4L`.
* **Audit Verdict**: **PARTIAL VALIDITY**. The mathematical derivation in `gradient_check.py` was sound and passed with `rel_err = 0.000896`, but it was an isolated standalone test on dummy tensors rather than testing the parameters inside the active `ArchitectureTranslator` training loop.
* **Required Correction in 7.8.1**: Run gradient validation directly on the actual `ArchitectureTranslator` adapter weights across input translator, adapter, and output translator.

---

### 4. Shuffled & Control Battery
* **Claim in Report**: Evaluated Random Matched, Shuffled Dependency, Path-Scrambled, and Host-Only Adaptation controls.
* **Code Implementation**:
  - `Condition_G_Random_Matched_Circuit`: Replaced donor weights with Gaussian noise (`0.1 * randn()`). Valid.
  - `Condition_H_Shuffled_Dependency_Circuit`: Overwrote slots 5, 4, 3, 2 of `L0.attn.o.W`. In 7.8, this achieved 45.0% accuracy because overwriting recipient heads acted as a massive structural perturbation.
  - `Condition_J_Host_Only_Adaptation`: Directly updated `L0.mlp.1.W` of recipient.
* **Audit Verdict**: **METHODOLOGICAL INSUFFICIENCY**. Section 12 mandates separating the shuffled controls into 4 distinct categories:
  1. *Shuffled-node control*: Same number of nodes, scrambled dependencies.
  2. *Shuffled-path control*: Same node types, incorrect causal ordering.
  3. *Parameter-shuffle control*: Same shapes/counts, randomized values.
  4. *Location-matched control*: Same target location, unrelated donor function.
* **Required Correction in 7.8.1**: Implement all 4 structured shuffled controls and register them in `control_registry.json` prior to evaluation.

---

### 5. Path-Patching & Activation Patching
* **Claim in Report**: 6-condition path patching proved 100% mediation by Layer 0 residual output.
* **Code Implementation**: `discovery/path_patching.py` evaluated the 6 conditions, but when testing `mlp_out` alone, recovery was 40.0% (0.0 pp gain over ablated), whereas patching `resid` output at Layer 0 yielded 70.0% (100% recovery).
* **Audit Verdict**: **INTERPRETIVE CLARIFICATION NEEDED**. The report stated that `L0_mlp` mediated 100% of the effect, but the code actually showed that the *entire Layer 0 residual stream* (which combines attention output `ao` and `mlp_out`) was required for 100% recovery. `mlp_out` alone accounted for a partial effect.
* **Required Correction in 7.8.1**: Clearly report the individual mediated effect of `mlp_out` vs the joint residual mediation.

---

### 6. Held-Out Generalization & 90% Threshold
* **Claim in Report**: Predeclared 90.0% threshold maintained; classification strictly Level B.
* **Code Implementation**: In `heldout_eval.py`, exact agreement was computed and honestly reported as ~20–30%, resulting in Level B classification.
* **Audit Verdict**: **SOUND AND HONEST**. The threshold was strictly preserved without goalpost-shifting.

---

### 7. Data Split Discipline & Leakage Check
* **Claim in Report**: 3 disjoint splits (Discovery 777, Validation 888, Held-out 999).
* **Code Implementation**: Items were generated with seeds 777, 888, 999 using `micro_suite('math', n=20, seed=...)`.
* **Audit Verdict**: **VALID BUT UNREGISTERED**. The splits were distinct, but there was no explicit `data_split_registry.json` or automated leakage check ensuring that no discovery or validation prompts overlapped with held-out prompts.
* **Required Correction in 7.8.1**: Save `data_split_registry.json` and implement an automated leakage test (`tests/test_leakage.py`).

---

## Action Plan for EQUYLAPTA 7.8.1

1. Write `audit_7_8_1.md` (this document) and `functional_objective.md`.
2. Implement `src/corrected_functional_objective.py` with true differentiable two-branch backward pass.
3. Update `src/transfer.py` to wire the true analytical functional gradient directly into the adapter training loop.
4. Implement `src/gradient_validation.py` testing the actual training loop gradient against central finite differences.
5. Create `control_registry.json` with the 4 structured shuffled controls.
6. Create `data_split_registry.json` and `training_objective.json`.
7. Execute the full experimental battery with frozen transfer first, followed by interface adaptation and receiver co-adaptation.
8. Evaluate on strictly held-out data across 5 independent seeds.
9. Generate authoritative reports: `EQUYLAPTA7POINT8_1_REPORT.md` and `EQUYLAPTA7POINT8_1_REPORT.txt`.
10. Validate with `validate_report.py` and verify workspace size strictly $< 120$ MB.
