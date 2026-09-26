# EQUYLAPTA7.2 Preflight Forensic Implementation Audit

## 1. Context and Objective
Before executing **EQUYLAPTA7.2: True Functional Translation Bridge**, this preflight audit critically examines the codebase and empirical findings of **EQUYLAPTA7.1** to ensure that all prior defects are understood, partially remediated components are completed, and architectural limitations are rigorously addressed.

The central goal of EQUYLAPTA7.2 is to determine:
> *Is the functional translation mechanism itself valid?*

Before testing cross-architecture transfer from **Architecture A** (`e4-math-4L`, $d=64$, 4 layers, 4 heads) to **Architecture B** (`e6-base-4L`, $d=96$, 4 layers, 6 heads), we **must** first validate functional reconstruction within the **same architecture** (**Architecture A $\to$ Architecture A**, using independently initialized `e4-base-4L`). If A $\to$ A fails, failure of A $\to$ B cannot be attributed to cross-architecture incompatibility.

---

## 2. Audit Matrix of E7.1 Implementation vs E7.2 Requirements

| Audit ID | Subsystem | Status in E7.1 | Forensic Evaluation | Required E7.2 Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **PREFLIGHT-01** | Source Signature Specificity | Remediated | Extracted from 16-dimensional Head 2 output representations. | Preserve `L0_head_2`; build Functional Signature V3 strictly separating implementation representation from functional representation. |
| **PREFLIGHT-02** | Independent Target Initialization | Remediated | 5 fresh target models with logged distinct SHA-256 hashes. | Maintain 5 fresh instances with logged initialization, alignment, and final hashes. |
| **PREFLIGHT-03** | Functional Signature Usage & Coordinate Assumption | **Critical Defect Persisting** | E7.1 applied source projection $P = B B^T$ directly to target coordinates, assuming identical coordinate semantics. | **MANDATORY**: Implement true learned alignment bridge (Source Functional Space $\to$ Paired Probes $\to$ Learned Map via Orthogonal Procrustes $\to$ Target Functional Space). |
| **PREFLIGHT-04** | Optimization Gradient Integrity | **Critical Defect Persisting** | E7.1 used heuristic directional updates rather than mathematical derivatives. | **MANDATORY**: Implement true loss derivatives via central finite-difference gradients. Enforce formal gradient check against numerical reference with error $< 5\%$. |
| **PREFLIGHT-05** | Exact Target-Unit Causality | Remediated | Defined `target_functional_unit_id: L0_head_0` with 4-step sequence. | Retain exact unit ID, parameter mask, and 4-step sequence for both A $\to$ A and A $\to$ B. |
| **PREFLIGHT-06** | Target-Local Trained Control | Partially Remediated | Trained for 30 steps, but used approximate gradient steps. | Train genuine target-local circuit on target data using true finite-difference gradients. |
| **PREFLIGHT-07** | Same-Architecture Control Bridge | **Missing in E7.1** | Jumped directly to cross-architecture transfer (A $\to$ B) without testing A $\to$ A. | **MANDATORY**: Execute Phase B (A $\to$ A) on separately initialized `e4-base-4L` before Phase C (A $\to$ B). |

---

## 3. Key Methodological Innovations for EQUYLAPTA7.2

### 3.1 Learned Alignment Bridge (Orthogonal Procrustes)
Rather than projecting target weights with the source basis directly ($B_{\text{src}} B_{\text{src}}^T$), E7.2 gathers paired functional activations across probe expressions:
$$X \in \mathbb{R}^{N \times 16} \quad (\text{source head output}), \quad Y \in \mathbb{R}^{N \times 16} \quad (\text{target baseline head output})$$
We solve the classical Orthogonal Procrustes problem:
$$R^* = \arg\min_{R, \, R^T R = I} \| X R - Y \|_F^2 = U V^T \quad \text{where} \quad \text{SVD}(X^T Y) = U \Sigma V^T$$
We also evaluate a **Random Alignment Null Control** ($R_{\text{rand}}$). If learned alignment does not outperform random alignment, the bridge transmits no functional signal.

### 3.2 Real Optimization with Central Finite-Difference Gradients
Target head output weights $W_o^{(\text{tgt})} \in \mathbb{R}^{e \times d}$ are optimized to minimize:
$$\mathcal{L}_{\text{function}} = \lambda_{\text{align}} \mathcal{L}_{\text{align}} + \lambda_{\text{task}} \mathcal{L}_{\text{task}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$
The gradient $\nabla_{W} \mathcal{L}$ is computed using true central finite differences:
$$\left[\nabla_W \mathcal{L}\right]_{i,j} = \frac{\mathcal{L}(W + \epsilon E_{i,j}) - \mathcal{L}(W - \epsilon E_{i,j})}{2\epsilon}$$
A dedicated unit test compares this gradient against single-sided numerical estimates and validates relative error $< 5\%$, saved in `gradient_check.json`.

### 3.3 Strict Scientific Order
1. **Phase A**: Source Characterization & Functional Signature V3 (`functional_signature_v3.json`)
2. **Phase B**: Architecture A $\to$ Architecture A Functional Reconstruction (`aa_results.json`)
3. **Phase C**: Architecture A $\to$ Architecture B Cross-Architecture Translation (`ab_results.json`)
4. **Phase D**: Exact Causal Validation (`causal_results.json`)
5. **Phase E**: Robustness Sweeps (Depth 2L, 4L, 6L, 8L & Size 100% to 1%)
6. **Phase F**: Decision Tree & Evidence Ladder Assessment
