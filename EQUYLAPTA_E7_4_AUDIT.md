# EQUYLAPTA E7.4 Retrospective Forensic Audit Report

**Audit Identifier**: `EQUYLAPTA_E7_4_AUDIT`  
**Date**: September 24, 2026  
**Auditor**: Senior AI Research Engineer (Mechanistic Interpretability & Causal Mediation)  
**Subject**: EQUYLAPTA E7.4 Codebase, Execution Artifacts, and Scientific Claims  
**Machine-Readable Companion**: `/home/user/e7_4_audit.json`  
**Overall Verdict**: `OPTIMIZATION_OBJECTIVE_MISMATCH_IDENTIFIED`  
**Audited Scientific Classification**: `LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY`

---

## 1. Executive Summary & Purpose of Audit

Before implementing EQUYLAPTA E7.5, an exhaustive, unsparing forensic audit of the EQUYLAPTA E7.4 codebase (`run_equylapta7_4.py`, `generate_e7_4_reports.py`, and canonical JSON results) was conducted. The central goal of E7.4 was to test whether a bidirectional synthetic interface bridge ($W_{\text{in}} \in \mathbb{R}^{16 \times 16}, W_{\text{out}} \in \mathbb{R}^{64 \times d_{\text{tgt}}}$) wrapping a strictly frozen transplanted attention head projection ($W_{\text{comp}} \in \mathbb{R}^{16 \times 64}$) could transfer causal computation into an independent target network.

While E7.4 successfully resolved the catastrophic metric invalidity of E7.3 (replacing coordinate slicing with downstream behavioral effect vectors in 141-token logit space) and honestly reported that causal mediation failed (-5.00 pp causal drop in Target B), **this audit reveals a critical implementation flaw in how the bridge parameters were optimized**. Specifically, the bridge optimization did not optimize the pure functional-effect objective claimed, but instead trained primarily on standard cross-entropy language modeling loss, while computing an incomplete single-branch gradient.

---

## 2. Comprehensive Forensic Evaluation Across the 10 Audit Items

### Item 1: What E7.4 Actually Implemented
E7.4 implemented:
1. A **Bidirectional Synthetic Interface Bridge**: An input adapter $W_{\text{in}} \in \mathbb{R}^{16 \times 16}$ (256 params), a frozen component $W_{\text{comp}} \in \mathbb{R}^{16 \times 64}$ (1,024 weights from source `L0_head_2`), and an output adapter $W_{\text{out}} \in \mathbb{R}^{64 \times 96}$ (6,144 params), computing $W_{\text{eff}} = W_{\text{in}} @ W_{\text{comp}} @ W_{\text{out}}$.
2. A **Common Behavioral Functional Effect Space**: Downstream logit perturbations $F(x) = \text{Logits}_{\text{intact}}(x)[:, -1, :] - \text{Logits}_{\text{ablated}}(x)[:, -1, :] \in \mathbb{R}^{141}$.
3. Five interface conditions (Transplant only, Input adapter, Output adapter, Bidirectional bridge, Bridge + receiver co-adaptation) evaluated across 5 seeds, 4 depths (2L, 4L, 6L, 8L), and 4 capacities (25%, 50%, 100%, 150%).
4. A 7-intervention causal pathway battery (ablation, restoration, input bridge bypass, output bridge bypass, random noise control, amplification $\times 1.5$, inversion $-1.0\times$).

### Item 2: What E7.4 Claimed to Implement
E7.4 claimed in §10, §11, and §12 of its report that the bidirectional bridge was trained specifically to reproduce the source model's causal functional perturbation effect vector:
$$\min_{\theta} \mathcal{L}_{\text{func}} = \frac{1}{2} \| F_{\text{target}}(x, \theta) - F_{\text{source}}(x) \|^2$$
where $F_{\text{target}}(x, \theta) = Y_{\text{intact}}(x, \theta) - Y_{\text{ablated}}(x, \theta)$.

### Item 3: Whether Those Two Things Match (The Mismatch Finding)
**THEY DO NOT MATCH.**
Inspection of `run_equylapta7_4.py` (lines 280–305) reveals two critical defects in the optimization loop:
1. **Contamination by Cross-Entropy Task Loss**:
   ```python
   loss_task, grads_task = target_model.loss_and_grads(arr)
   ...
   G_eff = grads_task["L0.attn.o.W"][:16, :] + lambda_func * grads_func["L0.attn.o.W"][:16, :]
   ```
   The gradient used to update the bridge was dominated by `grads_task` (the ordinary autoregressive cross-entropy loss on token predictions), with the functional effect gradient `grads_func` added only as an auxiliary penalty with $\lambda = 0.5$. Thus, the bridge was trained primarily as an ordinary task adapter rather than a pure functional-effect translator.
2. **Incomplete Single-Branch Differentiation**:
   The gradient `grads_func` was obtained by calling `target_model.backward(arr, dlogits_func)` solely on the intact model state. The loss was mathematically defined as the difference between intact and ablated outputs, but the backward pass failed to differentiate through the ablated branch $-dY_{\text{ablated}}/d\theta$. In Condition 5, where the Layer 0 MLP receiver was adapted, receiver parameters were updated without accounting for their impact on the ablated output, producing an analytically incorrect gradient.

### Item 4: Which Experiments Genuinely Executed
The master execution run (`python3 demo/run_equylapta7_4.py`) executed all declared phases deterministically in 268.7 seconds:
- Phase 1: Source causal indispensability verified (+24.00 pp drop, 0.00 pp recovery error).
- Phase 2: 5 interface conditions evaluated across 5 seeds on Arch B.
- Phase 3: 7 causal pathway interventions executed on Condition 4.
- Phase 4: Comparative pipelines ($A \to A$ control, $A \to C$ pipeline).
- Phase 5: Corrected depth sweep across 2L, 4L, 6L, 8L.
- Phase 6: Capacity sweep across 25%, 50%, 100%, 150%.
- Phase 7: Collateral capability tests across 6 domains.

### Item 5: Which Reported Metrics Are Valid
1. **Causal Drop ($Y_{\text{active}} - Y_{\text{ablated}}$)**: Accurately measured the performance impact of physically zeroing $W_{\text{comp}}$.
2. **Restoration Recovery Error**: Validly verified that restoring $W_{\text{comp}}$ returned the model to its exact intact state (0.00 pp error).
3. **Task Accuracy**: Validly computed on held-out test items.
4. **Random Component Comparison**: Validly compared true $W_{\text{comp}}$ (+7.00 pp advantage) against Gaussian noise.
5. **Frozen Component Invariance**: Parameter norm checks confirmed zero weight drift in $W_{\text{comp}}$ during training.

### Item 6: Which Metrics Are Only Proxies
1. **Common Functional Effect Cosine Similarity**: Measures geometric orientation in 141-dimensional logit space. While an improvement over coordinate slicing, high cosine similarity does not prove that the underlying computation was causally necessary for recipient behavior.
2. **Pearson Correlation ($r$)**: Measures linear co-movement of logit perturbation patterns across vocabulary tokens; acts as a representational diagnostic.
3. **Linear CKA and Procrustes Distance**: Diagnostic measures of representation similarity, completely decoupled from causal execution.

### Item 7: Whether Causal Mediation Was Demonstrated
**NO.**
Causal mediation decisively failed in both target architectures:
- In Target B (`e6-base-4L`), ablating $W_{\text{comp}}$ caused accuracy to **increase** from 30.00% to 35.00% (causal drop: **-5.00 pp**).
- In Target A (`e4-base-4L`), ablating the bridged head caused accuracy to **increase** from 18.00% to 19.00% (causal drop: **-1.00 pp**).
Removing the transplanted component relieved representational interference and allowed the recipient model to achieve higher accuracy using its native pathways.

### Item 8: Whether the Transplanted Component Itself Caused the Observed Behavior
**NO.**
The target network bypassed the transplanted component. The presence of the active component acted as a distractor or impedance. When ablated, the target model's accuracy returned toward its unadapted baseline (34.00%).

### Item 9: Whether Receiver Adaptation Learned Around the Transplant
**YES.**
In Condition 5 (Bridge + Layer 0 MLP adaptation), functional agreement collapsed from 40.78% to **3.53%**, and the causal drop remained negative (**-10.00 pp**). The downstream receiver adapted to fit local training tokens by bypassing the frozen bridge rather than learning to decode its computational signals.

### Item 10: Whether Bridge Optimization Truly Optimized the Claimed Functional-Effect Objective
**NO.**
As demonstrated under Item 3, bridge optimization was contaminated by cross-entropy task loss and omitted the backward gradient through the ablated branch.

---

## 3. Scientific Downgrade & Classification

Because causal mediation was non-positive (-5.00 pp drop in Target B, -1.00 pp drop in Target A) and functional agreement reached only 40.78% (falling far short of the 90.0% predeclared threshold), E7.4 is scientifically classified as:

**`LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY`**

Any higher classification (Levels C, D, E, or F) is completely precluded by empirical evidence.

---

## 4. Remediation Mandates for EQUYLAPTA E7.5

To repair these foundational issues, EQUYLAPTA E7.5 must execute the following mandates:
1. **Pure Functional-Effect Optimization**: Eliminate cross-entropy task loss contamination from Condition A; optimize the bridge exclusively on functional-effect matching:
   $$\mathcal{L}_{\text{func}} = \frac{1}{2 V} \sum_{v=1}^V \left( (Y_{\text{intact}, v}(x, \theta) - Y_{\text{ablated}, v}(x, \theta)) - F_{\text{source}, v}(x) \right)^2$$
2. **Complete Two-Branch Differentiability**: Differentiate through both branches of the functional-effect expression:
   $$\nabla_\theta \mathcal{L}_{\text{func}} = \nabla_\theta Y_{\text{intact}}(\dots; +d\text{logits}) + \nabla_\theta Y_{\text{ablated}}(\dots; -d\text{logits})$$
3. **Mandatory Analytical vs. Finite-Difference Gradient Check**: Implement `tests/test_functional_effect_gradient.py` and verify that the analytical gradient matches central finite differences with relative error $< 0.05$ before running training experiments.
4. **Three Separate Optimization Conditions**:
   - Condition A: Pure Functional-Effect Bridge (train $W_{\text{in}}, W_{\text{out}}$; freeze $W_{\text{comp}}$ and backbone; objective: $\mathcal{L}_{\text{func}}$).
   - Condition B: Intact-Output Control (train equivalent bridge using ordinary intact-output matching).
   - Condition C: Random Component Control (replace $W_{\text{comp}}$ with matched Gaussian noise, train with $\mathcal{L}_{\text{func}}$).
5. **Mandatory Host-Learning Control**: Evaluate a target model given equivalent parameter capacity and training budget with **no transplanted functional component** to formally prove whether observed improvements stem from host learning vs. transplant mediation.
6. **Preserve Strict Component Freezing**: Verify that $W_{\text{comp}}$ remains 100% frozen (0 modified parameters, exact hash matching).
