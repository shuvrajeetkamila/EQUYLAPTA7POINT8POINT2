# EQUYLAPTA E7.3 Audit & Forensic Review
## Independent Methodological and Causal Evaluation

**Milestone Audited**: `EQUYLAPTA E7.3`  
**Auditor**: Senior AI Research Engineer (Mechanistic Interpretability & Model Fusion)  
**Overall Verdict**: **CLASSIFICATION DOWNGRADE REQUIRED (Level E $\to$ Level B)**  
**Corrected Status**: `LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY`  

---

### Executive Summary

Before implementing EQUYLAPTA E7.4, an exhaustive audit of the E7.3 experimental pipeline, code artifacts, and numerical records was conducted. While E7.3 correctly resolved the validation Procrustes refitting bug from E7.2 and rigorously enforced frozen parameter invariants on the transplanted head, its scientific classification and primary metrics suffered from several critical vulnerabilities:
1. **Premature Assignment of Level E**: Level E is defined as *"Cross-architecture functional effect with localized adaptation AND causal mediation by the transplanted component."* The empirical data shows a causal drop of exactly **+0.00 pp** in Target B and **-2.00 pp** in Target A. The component exerted zero causal mediation on task success.
2. **Invalid Functional Agreement Metric**: E7.3 compared raw coordinate slices `out_s[:16]` (coordinates 0..15 of a 64-dimensional vector) against `out_t[:16]` (coordinates 0..15 of a 96-dimensional vector). Comparing arbitrary coordinate indices across non-isomorphic latent spaces lacks mathematical justification.
3. **Conflation of Local Target Learning with Functional Transfer**: Expanding downstream receiver radius to Radius 3 in Target A raised accuracy from 13.00% to 21.00%, but ablating the transplanted head yielded 23.00% accuracy. The host learned to perform the task independently, bypassing the head.

---

### 1. Evaluation of Core Questions

#### Question 1: Was $A \to A$ causal transfer actually demonstrated?
**Evaluation**: **NO.**
- Target A baseline accuracy: **13.00%**.
- Transplant without adaptation: **11.00%** (causal drop under ablation: **-2.00 pp**).
- After Radius 3 co-adaptation, accuracy reached **21.00%** (+8.00 pp gain).
- However, when the transplanted head `L0_head_2` was ablated in this adapted host, accuracy was **23.00%** (causal drop: **-2.00 pp**; intact 21% vs ablated 23%).
- **Conclusion**: The host model learned the domain tokens autonomously through its downstream layers; removing the transplanted unit actually improved accuracy slightly. There was zero causal functional transfer.

#### Question 2: Was $A \to B$ causal transfer actually demonstrated?
**Evaluation**: **NO.**
- Target B baseline accuracy: **34.00%**.
- Method 2 (Functional Procrustes) without adaptation: **35.00%** (causal drop under ablation: **+0.00 pp**; intact 35% vs ablated 35%).
- Method 2 with Radius 1 co-adaptation: accuracy declined to **26.00%** (causal drop under ablation: **+0.00 pp**; intact 26% vs ablated 26%).
- Radius 2 co-adaptation: **25.00%** (causal drop: **+0.00 pp**).
- Radius 3 co-adaptation: **23.00%** (causal drop: **+0.00 pp**).
- **Conclusion**: The transplanted unit had zero causal influence on downstream predictions.

#### Question 3: Did co-adaptation cause the transplanted component to become necessary?
**Evaluation**: **NO.**
- Necessity requires that disabling the component causes downstream task collapse (as seen in source `L0_head_2`: +24.00 pp collapse).
- In both $A \to A$ and $A \to B$, disabling the transplanted unit caused no performance degradation whatsoever. Co-adaptation did not induce host dependency.

#### Question 4: Was the functional-agreement metric architecture-independent?
**Evaluation**: **NO.**
- The E7.3 code calculated:
  ```python
  cos_sim = float(np.dot(out_s[:16], out_t[:16]) / (np.linalg.norm(out_s[:16]) * np.linalg.norm(out_t[:16]) + 1e-9))
  ```
- Truncating residual representations to their first 16 dimensions assumes that basis vectors $e_0 \dots e_{15}$ represent the same semantic directions in both models. This assumption is mathematically invalid in unaligned latent spaces.

#### Question 5: Was the depth sweep measuring the intended construct?
**Evaluation**: **PARTIALLY FLAWED.**
- In the 2L model, co-adapting Layer 0 MLP produced an apparent accuracy surge (21.00% $\to$ 32.00%), but functional agreement was **-3.49%**. Because 2L models have only one layer before logits, the Layer 0 MLP was able to directly memorize target training examples, confounding local parameter tuning with functional transfer.

#### Question 6: Were any conclusions stronger than their underlying evidence?
**Evaluation**: **YES.**
- The report conclusion *"Local downstream receiver adaptation is sufficient to initiate signal interface"* is not supported by the evidence. The receiver adapted to the task corpus, but it ignored or bypassed the transplanted head.

---

### 2. Formal Classification Downgrade

Per the strict definitions in the EQUYLAPTA evidence ladder:
- **Level E** requires: *Cross-architecture functional effect with localized adaptation AND causal mediation by the transplanted component.*
- Because causal mediation was **0.00 pp** and functional agreement was **11.50%** (against the predeclared 90% threshold), E7.3 cannot defensibly claim Level E.
- **Audited Level**: **LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY** (Orthogonal Procrustes achieved geometric alignment on training representations, but no causal functional transfer was established).

---

### 3. Concrete Architectural Directives for EQUYLAPTA E7.4

To resolve these defects, EQUYLAPTA E7.4 must implement:
1. **Common Functional Effect Space**:
   Compare the downstream causal perturbation vectors in vocabulary/logit space:
   $$F_{\text{source}}(x) = \text{Logits}(M_{\text{src}}, x) - \text{Logits}(M_{\text{src,abl}}, x)$$
   $$F_{\text{target}}(x) = \text{Logits}(M_{\text{tgt}}, x) - \text{Logits}(M_{\text{tgt,abl}}, x)$$
   This creates a 100% architecture-independent metric.
2. **Bidirectional Synthetic Interface Bridge**:
   Explicitly model both directions of the interface:
   $$\text{Target Context } x \xrightarrow{W_{\text{in}}} \text{Transplanted Head } (100\% \text{ frozen}) \xrightarrow{W_{\text{out}}} \text{Target Receiver}$$
3. **Controlled Component Disentanglement (5 Conditions)**:
   Test transplant-only, input-adapter-only, output-adapter-only, bidirectional-bridge, and bridge-plus-coadaptation to localize the exact point of interface failure.
4. **Full Causal Mediation Pathway Test**:
   Intervene on the complete chain: bypass input adapter, ablate component, bypass output adapter, restore component, and test against random component controls.
