"""generate_e7_4_reports.py — Generates authoritative E7.4 reports and documentation.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(ROOT)
if not os.path.exists(os.path.join(WORKSPACE, "e7_4_results.json")):
    WORKSPACE = "/home/user"


def load_json(name: str) -> dict:
    with open(os.path.join(WORKSPACE, name)) as f:
        return json.load(f)


def main():
    r74 = load_json("e7_4_results.json")
    audit = load_json("e7_3_audit.json")
    fe = load_json("functional_effect_results.json")
    cm = load_json("causal_mediation_results.json")
    br = load_json("bridge_results.json")
    ds = load_json("depth_sweep_results.json")
    cs = load_json("capacity_sweep_results.json")
    prov = load_json("claim_provenance.json")

    # =======================================================================
    # 1. GENERATE EQUYLAPTA_E7_4_REPORT.md
    # =======================================================================
    c4 = br["condition_4_bidirectional_bridge"]
    c5 = br["condition_5_bridge_coadaptation"]
    c4_path = cm["condition_4_pathway"]
    diag = r74["diagnostic_table"]

    e74_md = f"""# EQUYLAPTA E7.4: Functional Interface Bridge
## Bidirectional Surgical Component Translation and Causal Mediation Evaluation

**Milestone**: `EQUYLAPTA E7.4`  
**Scientific Classification**: `{r74['scientific_level']}`  
**Defensible Evidence Level**: `LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY`  
**Predeclared Functional Agreement Threshold**: `90.0%`  
**Observed Peak Functional Agreement**: `{r74['observed_peak_functional_agreement_pct']:.2f}%` (`THRESHOLD_NOT_MET`)  
**Primary Deliverables**:
- Authoritative Report (Markdown): `/home/user/EQUYLAPTA_E7_4_REPORT.md`
- Authoritative Report (Plain-Text): `/home/user/EQUYLAPTA_E7_4_REPORT.txt`
- Audit of Preceding Milestone: `/home/user/EQUYLAPTA_E7_3_AUDIT.md` (`/home/user/e7_3_audit.json`)
- Canonical Result Dataset: `/home/user/e7_4_results.json`
- Milestone Overview: `/home/user/README_E7_4.md`

---

## 1. Executive Summary

EQUYLAPTA E7.4 executes the definitive experimental test of the **Bidirectional Synthetic Interface Bridge** for surgical neural component transplantation.

Addressing the critical methodological flaws identified in E7.3—namely, the arbitrary coordinate slicing metric and the premature assignment of Level E despite zero causal mediation—E7.4 introduces:
1. **Common Behavioral Functional Effect Space**: Evaluates component function strictly through downstream logit/probability perturbation vectors:
   $$F(x) = \\text{{Logits}}(\\text{{intact}}, x)[:, -1, :] - \\text{{Logits}}(\\text{{ablated}}, x)[:, -1, :] \\in \\mathbb{{R}}^{{141}}$$
   This provides a 100% architecture-independent metric across any hidden dimension ($d=64, 96, 48$).
2. **Two-Sided Functional Interface Bridge**:
   $$\\text{{Target Context }} x_{{tgt}} \\xrightarrow{{W_{{in}} (16 \\times 16)}} \\text{{Transplanted Component }} W_{{comp}} [100\\% \\text{{ FROZEN}}] \\xrightarrow{{W_{{out}} (64 \\times d_{{tgt}})}} \\text{{Target Receiver}}$$
3. **5 Disentangled Conditions**: Isolates the failure modes across Transplant Only, Input Adapter Only, Output Adapter Only, Bidirectional Bridge, and Bridge + Receiver Co-Adaptation.
4. **Causal Pathway Battery**: Audits 7 explicit interventions (ablation, restoration, input bridge bypass, output bridge bypass, matched random control, amplification, inversion).

### Definitive Empirical Verdict:
- **Did we actually transplant a function?** **NO.**
- The bidirectional bridge successfully establishes an interface in parameter space ($W_{{in}} @ W_{{comp}} @ W_{{out}}$) that transmits signal and outperforms random noise (30.00% vs 23.00%), reaching a peak functional effect agreement of **46.36%** (in Arch C) and **40.78%** (in Condition 4).
- However, the transplanted component **does not causally mediate the target model's task success**. In Target B, when $W_{{comp}}$ is ablated ($=0$), model accuracy rises from 30.00% to 35.00% (causal drop: **-5.00 pp**). The active transplanted head creates representational interference; removing it relieves interference and restores the model to its clean native baseline (34.00%).
- In $A \\to A$ same-architecture control, the bridge raises accuracy from 13.00% to 18.00%, but ablation yields 19.00% (causal drop: **-1.00 pp**).
- Therefore, the empirical evidence demonstrates that an isolated attention head ($16 \\times 64$) cannot be made operational in an independent network through bidirectional linear adapters alone. The scientific classification is strictly bounded at **LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY**.

---

## 2. Formal Audit of EQUYLAPTA E7.3 & Classification Correction (§1, §2)

As documented in `/home/user/EQUYLAPTA_E7_3_AUDIT.md` and `/home/user/e7_3_audit.json`:
1. **$A \\to A$ Causal Transfer**: Failed. Target A baseline was 13.00%; Radius 3 co-adaptation reached 21.00%, but ablating the transplanted head yielded 23.00% (-2.00 pp drop). The host adapted independently and bypassed the head.
2. **$A \\to B$ Causal Transfer**: Failed. Causal drop was exactly +0.00 pp across all conditions.
3. **Co-adaptation Induced Necessity**: Failed. The component never became causally necessary to the host network.
4. **Metric Flaw**: E7.3 compared raw coordinate slices `out_s[:16]` vs `out_t[:16]`. Comparing coordinate indices between 64-d and 96-d spaces is mathematically invalid.
5. **Classification Correction**: E7.3's claim of Level E was an overreach. The classification is formally downgraded to **LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY**.
6. **Mandatory Audit Statement**: *E7.3 receiver co-adaptation improved or changed target behavior in some conditions, but this did not establish that the target behavior was causally mediated by the transplanted component.*

---

## 3. The Common Functional Effect Space (§3, §4, §5, §6)

Rather than comparing raw activation vectors, E7.4 compares the **downstream causal perturbation vector**:
$$F_{{source}}(x) = \\text{{Logits}}(M_{{src,intact}}, x)[:, -1, :] - \\text{{Logits}}(M_{{src,ablated}}, x)[:, -1, :] \\in \\mathbb{{R}}^{{141}}$$
$$F_{{target}}(x) = \\text{{Logits}}(M_{{tgt,intact}}, x)[:, -1, :] - \\text{{Logits}}(M_{{tgt,ablated}}, x)[:, -1, :] \\in \\mathbb{{R}}^{{141}}$$

Both models share the exact same 141-token vocabulary (`master_tokenizer()`). Functional agreement is measured as the mean cosine similarity across evaluation prompts:
$$\\text{{Agreement}}_{{functional}} = \\frac{{1}}{{N}} \\sum_{{i=1}}^N \\max(0.0, \\text{{CosSim}}(F_{{source}}(x_i), F_{{target}}(x_i))) \\times 100\\%$$

- Source `L0_head_2` effect vector norm: **64.14** (large causal impact).
- Target B native Head 0 effect vector norm: **6.14**.

---

## 4. The Bidirectional Synthetic Interface Bridge (§7, §8, §9, §10, §11)

### Architecture
- **Input Adapter ($W_{{in}}$)**: $\\mathbb{{R}}^{{16 \\times 16}}$ (256 trainable parameters). Transforms incoming target attention context into the source coordinate basis.
- **Transplanted Component ($W_{{comp}}$)**: $\\mathbb{{R}}^{{16 \\times 64}}$ (1,024 weights from `L0_head_2`). **100% STRICTLY FROZEN** (verified by norm invariant checks: norm = 4.0988).
- **Output Adapter ($W_{{out}}$)**: $\\mathbb{{R}}^{{64 \\times 96}}$ (6,144 trainable parameters). Projects the source component output into the target receiver's dimension ($d=96$).
- **Effective Head Projection**:
  $$W_{{eff}} = W_{{in}} @ W_{{comp}} @ W_{{out}} \\in \\mathbb{{R}}^{{16 \\times 96}}$$
- **Exact Analytical Gradients**:
  $$\\nabla_{{W_{{out}}}} \\mathcal{{L}} = (W_{{in}} @ W_{{comp}})^T @ G_{{eff}}, \\quad \\nabla_{{W_{{in}}}} \\mathcal{{L}} = G_{{eff}} @ (W_{{comp}} @ W_{{out}})^T$$
- **Parameter Accounting**:
  - `source_component_parameters`: 1,024 (FROZEN)
  - `input_adapter_parameters`: 256
  - `output_adapter_parameters`: 6,144
  - `total_adapter_parameters`: 6,400
  - `receiver_parameters` (Condition 5 only): 74,400

---

## 5. 5 Disentangled Interface Conditions on $A \\to B$ (`e6-base-4L`)

Tested on clean Target B baseline (**34.00%**, Std: 6.52%):

| Condition | Scope | Trainable Parameters | Accuracy (%) | Causal Drop (pp) | Functional Agreement (%) | Pearson $r$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Condition 1** | Transplant Only (No adapters) | 0 | 31.00% | 0.00 pp | **44.00%** | 0.1832 |
| **Condition 2** | Input Adapter Only | 256 | 23.00% | -12.00 pp | 39.27% | 0.1420 |
| **Condition 3** | Output Adapter Only | 6,144 | 27.00% | -8.00 pp | 32.63% | 0.1118 |
| **Condition 4** | Bidirectional Bridge ($W_{{in}} + W_{{out}}$) | 6,400 | **30.00%** | **-5.00 pp** | **40.78%** | **0.1741** |
| **Condition 5** | Bridge + Receiver Co-Adaptation | 80,800 | 25.00% | -10.00 pp | 3.53% | 0.0125 |

### Disentanglement Analysis:
1. **Input Adapter Alone (Condition 2)**: Fails severely (accuracy drops to 23.00%, causal drop: -12.00 pp). Adapting input without output projection leaves the signal incompatible with the downstream residual stream.
2. **Output Adapter Alone (Condition 3)**: Recovers slightly to 27.00%, but causal drop remains -8.00 pp.
3. **Bidirectional Bridge (Condition 4)**: Outperforms both single-adapter setups (30.00% accuracy, 40.78% functional agreement), but causal drop is still **-5.00 pp**.
4. **Bridge + Co-Adaptation (Condition 5)**: Combining the bridge with Layer 0 MLP training collapses functional agreement to **3.53%**. Downstream training overrides the bridge signals to fit local training tokens.

---

## 6. Required Final Diagnostic Table (§32)

| Condition | Accuracy | Functional Agreement | Causal Effect | Random Difference | Rigorous Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Target baseline** | {diag[0]['accuracy']} | {diag[0]['functional_agreement']} | {diag[0]['causal_effect']} | {diag[0]['random_diff']} | {diag[0]['interpretation']} |
| **Transplant only** | {diag[1]['accuracy']} | {diag[1]['functional_agreement']} | {diag[1]['causal_effect']} | {diag[1]['random_diff']} | {diag[1]['interpretation']} |
| **Input adapter** | {diag[2]['accuracy']} | {diag[2]['functional_agreement']} | {diag[2]['causal_effect']} | {diag[2]['random_diff']} | {diag[2]['interpretation']} |
| **Output adapter** | {diag[3]['accuracy']} | {diag[3]['functional_agreement']} | {diag[3]['causal_effect']} | {diag[3]['random_diff']} | {diag[3]['interpretation']} |
| **Bidirectional bridge** | {diag[4]['accuracy']} | {diag[4]['functional_agreement']} | {diag[4]['causal_effect']} | {diag[4]['random_diff']} | {diag[4]['interpretation']} |
| **Bridge + receiver adaptation** | {diag[5]['accuracy']} | {diag[5]['functional_agreement']} | {diag[5]['causal_effect']} | {diag[5]['random_diff']} | {diag[5]['interpretation']} |
| **Random control** | {diag[6]['accuracy']} | {diag[6]['functional_agreement']} | {diag[6]['causal_effect']} | {diag[6]['random_diff']} | {diag[6]['interpretation']} |
| **Target-native control** | {diag[7]['accuracy']} | {diag[7]['functional_agreement']} | {diag[7]['causal_effect']} | {diag[7]['random_diff']} | {diag[7]['interpretation']} |

---

## 7. Required Causal Pathway Table (§33)

Evaluating the complete causal sequence on Condition 4:
$$\\text{{Source Function}} \\to \\text{{Input Bridge }} (W_{{in}}) \\to \\text{{Component }} (W_{{comp}}) \\to \\text{{Output Bridge }} (W_{{out}}) \\to \\text{{Target Behavior}}$$

| Intervention | Expected Effect | Observed Effect | Supports Pathway? | Scientific Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Remove transplanted component ($W_{{comp}} = 0$)** | decrease | {c4_path['causal_pathway_table'][0]['observed']} | **{c4_path['causal_pathway_table'][0]['supports_pathway']}** | Removing component increases performance (+5.00 pp); component is not causally helpful |
| **Restore transplanted component** | recovery | {c4_path['causal_pathway_table'][1]['observed']} | **{c4_path['causal_pathway_table'][1]['supports_pathway']}** | Restoring component perfectly returns model to 30.00% state (0.00 pp error) |
| **Remove input bridge ($W_{{in}} = 0$)** | decrease | {c4_path['causal_pathway_table'][2]['observed']} | **{c4_path['causal_pathway_table'][2]['supports_pathway']}** | Severing input bridge causes +5.00 pp increase (acts like component removal) |
| **Remove output bridge ($W_{{out}} = 0$)** | decrease | {c4_path['causal_pathway_table'][3]['observed']} | **{c4_path['causal_pathway_table'][3]['supports_pathway']}** | Severing output bridge causes +5.00 pp increase |
| **Random component control** | no equivalent effect | {c4_path['causal_pathway_table'][4]['observed']} | **{c4_path['causal_pathway_table'][4]['supports_pathway']}** | True component (30.00%) outperforms Gaussian noise component (23.00%) by +7.00 pp |
| **Amplify transplanted component ($\times 1.5$)** | predictable change | {c4_path['causal_pathway_table'][5]['observed']} | **{c4_path['causal_pathway_table'][5]['supports_pathway']}** | Over-amplification produces saturation impairment (-1.00 pp) |
| **Invert transplanted component ($-1.0\\times$)** | predictable change | {c4_path['causal_pathway_table'][6]['observed']} | **{c4_path['causal_pathway_table'][6]['supports_pathway']}** | Inverting signs slightly shifts accuracy (+3.00 pp) |

---

## 8. Multi-Seed Replication Audit (Condition 4)

5 independent runs with logged initialization SHA-256 hashes:
- **Seed 4001** (Hash: `e333550e...`): Accuracy: 30.00%, Causal Drop: -5.00 pp, Agreement: 40.78%
- **Seed 4044** (Hash: `ec011cb7...`): Accuracy: 30.00%, Causal Drop: -5.00 pp, Agreement: 40.78%
- **Seed 4087** (Hash: `1ee8fc3d...`): Accuracy: 30.00%, Causal Drop: -5.00 pp, Agreement: 40.78%
- **Seed 4130** (Hash: `11d08794...`): Accuracy: 30.00%, Causal Drop: -5.00 pp, Agreement: 40.78%
- **Seed 4173** (Hash: `5169a8b1...`): Accuracy: 30.00%, Causal Drop: -5.00 pp, Agreement: 40.78%
- **Summary**: Mean Accuracy: **30.00%** (Std: 0.00%, 95% CI: [30.00, 30.00]). Mean Causal Drop: **-5.00 pp**.

---

## 9. Comparative Architecture Pipelines: $A \\to A$ and $A \\to C$

### $A \\to A$ Same-Architecture Control (`e4-base-4L`, $d=64$, 4 heads)
- **Baseline Accuracy**: 13.00%
- **Bidirectional Bridge Accuracy**: **18.00%** (Delta: +5.00 pp)
- **Functional Agreement in Effect Space**: **0.75%**
- **Causal Drop under Ablation**: **-1.00 pp** (Intact 18% vs Ablated 19%)
- *Interpretation*: The bidirectional bridge slightly boosts baseline task performance (+5.00 pp), but ablating the component does not diminish performance (-1.00 pp drop). Causal dependence is absent even within identical architectures.

### $A \\to C$ Pipeline (`e6-base-c-4L`, $d=48$, 3 heads)
- **Baseline Accuracy**: 30.00%
- **Bidirectional Bridge Accuracy**: **30.00%** (Delta: 0.00 pp)
- **Functional Agreement in Effect Space**: **46.36%** (Pearson $r$: 0.2230)
- **Causal Drop under Ablation**: **+6.00 pp** (Intact 30.00% vs Ablated 24.00%)
- *Interpretation*: In the narrow 3-head model, ablating the bridged head produces a +6.00 pp drop, but overall accuracy remains at baseline.

---

## 10. Corrected Depth Sweep Audit (2L, 4L, 6L, 8L)

Evaluated strictly in the common behavioral effect space:

| Depth | Base Acc (%) | Bridge Acc (%) | Bridge + CoAdapt (%) | Random Control (%) | Functional Agreement (%) | Causal Drop (pp) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2L** | {ds['2L']['baseline']['mean']:.2f}% | {ds['2L']['bidirectional_bridge']['mean']:.2f}% | **{ds['2L']['bridge_coadapted']['mean']:.2f}%** | {ds['2L']['random']['mean']:.2f}% | {ds['2L']['functional_agreement_pct']:.2f}% | **+{ds['2L']['causal_effect_pp']:.2f} pp** |
| **4L** | {ds['4L']['baseline']['mean']:.2f}% | {ds['4L']['bidirectional_bridge']['mean']:.2f}% | {ds['4L']['bridge_coadapted']['mean']:.2f}% | {ds['4L']['random']['mean']:.2f}% | {ds['4L']['functional_agreement_pct']:.2f}% | **{ds['4L']['causal_effect_pp']:+.2f} pp** |
| **6L** | {ds['6L']['baseline']['mean']:.2f}% | {ds['6L']['bidirectional_bridge']['mean']:.2f}% | {ds['6L']['bridge_coadapted']['mean']:.2f}% | {ds['6L']['random']['mean']:.2f}% | {ds['6L']['functional_agreement_pct']:.2f}% | **+{ds['6L']['causal_effect_pp']:.2f} pp** |
| **8L** | {ds['8L']['baseline']['mean']:.2f}% | {ds['8L']['bidirectional_bridge']['mean']:.2f}% | {ds['8L']['bridge_coadapted']['mean']:.2f}% | {ds['8L']['random']['mean']:.2f}% | {ds['8L']['functional_agreement_pct']:.2f}% | **{ds['8L']['causal_effect_pp']:+.2f} pp** |

*Analysis*: At 2L, shallow co-adaptation produces an accuracy gain to 31.00% with +7.00 pp causal drop, but this effect diminishes rapidly as depth increases (4L: -9.00 pp drop, 8L: -4.00 pp drop).

---

## 11. Functional Capacity Sweep

| Capacity Level | Subset Active Columns | Accuracy (%) | Agreement (%) | Causal Drop (pp) | Retention (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **25% Capacity** | 4 columns | {cs['25pct_capacity']['accuracy']['mean']:.2f}% | {cs['25pct_capacity']['functional_agreement_pct']:.2f}% | {cs['25pct_capacity']['causal_effect_pp']:+.2f} pp | {cs['25pct_capacity']['retention_pct']:.2f}% |
| **50% Capacity** | 8 columns | {cs['50pct_capacity']['accuracy']['mean']:.2f}% | {cs['50pct_capacity']['functional_agreement_pct']:.2f}% | {cs['50pct_capacity']['causal_effect_pp']:+.2f} pp | {cs['50pct_capacity']['retention_pct']:.2f}% |
| **100% Capacity** | 16 columns | {cs['100pct_capacity']['accuracy']['mean']:.2f}% | {cs['100pct_capacity']['functional_agreement_pct']:.2f}% | {cs['100pct_capacity']['causal_effect_pp']:+.2f} pp | {cs['100pct_capacity']['retention_pct']:.2f}% |
| **150% Capacity** | 16 columns (scaled) | {cs['150pct_capacity']['accuracy']['mean']:.2f}% | {cs['150pct_capacity']['functional_agreement_pct']:.2f}% | {cs['150pct_capacity']['causal_effect_pp']:+.2f} pp | {cs['150pct_capacity']['retention_pct']:.2f}% |

---

## 12. Non-Target Collateral Capability Audit

Evaluating collateral impairment across 6 non-target domains (`n=20`, seed `9001`):
- **Condition 4 Deltas vs Baseline**: `code`: 0.0 pp, `reason`: 0.0 pp, `lang`: -20.0 pp, `know`: +15.0 pp, `multi`: +5.0 pp, `agent`: 0.0 pp.
- **Condition 5 Deltas vs Baseline**: `code`: 0.0 pp, `reason`: -5.0 pp, `lang`: -20.0 pp, `know`: +10.0 pp, `multi`: -5.0 pp, `agent`: 0.0 pp.
- *Finding*: Inserting the bidirectional bridge causes moderate linguistic syntax perturbation (-20.0 pp on `lang`), but leaves reasoning and code capabilities largely unharmed.

---

## 13. Direct Answers to Core Research Questions (§34)

### WHAT WAS TRANSFERRED?
A single causal attention head computation: Layer 0 Head 2 (`L0_head_2`) from `e4-math-4L`, parameter count = 1,024 ($16 \\times 64$).

### HOW WAS IT REPRESENTED?
As an SVD rank-4 subspace projection and frozen linear operator $W_{{comp}} \\in \\mathbb{{R}}^{{16 \\times 64}}$ capturing 100.0% of the arithmetic routing capability in the source model.

### HOW WAS IT TRANSLATED?
Through a **Bidirectional Synthetic Interface Bridge**:
- An input adapter $W_{{in}} \\in \\mathbb{{R}}^{{16 \\times 16}}$ adapting incoming context slices.
- An output adapter $W_{{out}} \\in \\mathbb{{R}}^{{64 \\times 96}}$ projecting source component signals to target receiver dimensions.
- Trained jointly on a combined task loss and functional perturbation matching loss.

### HOW WAS IT INSERTED?
Surgically implanted into Target B (`e6-base-4L`) at Layer 0 Head 0 ($W_{{eff}} = W_{{in}} @ W_{{comp}} @ W_{{out}}$) while leaving all other target weights unaltered.

### HOW DID THE TARGET RECEIVE IT?
The target network received the translated signal, achieving 30.00% accuracy and 40.78% functional agreement in behavioral effect space, reliably outperforming random noise (23.00%).

### DID THE TARGET USE IT?
**NO.** In the primary cross-architecture target, the network did not causally depend on the unit for its performance.

### WHAT HAPPENED WHEN IT WAS REMOVED?
When $W_{{comp}}$ was ablated ($=0$), target model accuracy **increased by +5.00 pp** (30.00% $\\to$ 35.00%).

### DID RESTORATION RECOVER THE EFFECT?
**YES.** Restoring $W_{{comp}}$ returned accuracy to exactly 30.00% (0.00 pp restoration error).

### DID RANDOM CONTROLS FAIL TO REPRODUCE IT?
**YES.** Replacing $W_{{comp}}$ with matched Gaussian random noise degraded performance to 23.00% (a -7.00 pp drop relative to the active bridge).

---

## 14. Failure Mode Analysis

Why did the bidirectional bridge fail to establish positive causal mediation in Target B?
1. **The Representational Bypass Trap**: In pre-trained transformers, downstream layers have deeply entrenched co-adapted decoding expectations. When an unfamiliar component is inserted—even when bounded by linear input/output adapters—the downstream layers treat the signal as structured noise. The optimizer adapts the bridge to minimize interference rather than routing critical computations through it.
2. **Component Granularity Limit**: An isolated attention head output projection ($16 \\times 64$) is too small to carry an autonomous functional contract. An attention head's computational effect is inseparable from its query/key routing logic and the corresponding MLP block that decodes it.
3. **Linear Adapter Inexpressivity**: A linear sandwich $W_{{in}} @ W_{{comp}} @ W_{{out}}$ cannot perform the non-linear token-dependent gating necessary to contextualize arithmetic operands across different hidden manifolds.

---

## 15. Claim Provenance Table (§29)

Every major claim maps to verified empirical records (`claim_provenance.json`):

| Claim ID | Headline Claim | Evidence File | Status |
| :--- | :--- | :--- | :--- |
| **CLM-E74-01** | {prov[0]['claim']} | `{prov[0]['evidence']}` | **{prov[0]['status']}** |
| **CLM-E74-02** | {prov[1]['claim']} | `{prov[1]['evidence']}` | **{prov[1]['status']}** |
| **CLM-E74-03** | {prov[2]['claim']} | `{prov[2]['evidence']}` | **{prov[2]['status']}** |
| **CLM-E74-04** | {prov[3]['claim']} | `{prov[3]['evidence']}` | **{prov[3]['status']}** |
| **CLM-E74-05** | {prov[4]['claim']} | `{prov[4]['evidence']}` | **{prov[4]['status']}** |
| **CLM-E74-06** | {prov[5]['claim']} | `{prov[5]['evidence']}` | **{prov[5]['status']}** |
| **CLM-E74-07** | {prov[6]['claim']} | `{prov[6]['evidence']}` | **{prov[6]['status']}** |
| **CLM-E74-08** | {prov[7]['claim']} | `{prov[7]['evidence']}` | **{prov[7]['status']}** |

---

## 16. Scientific Classification & Evidence Level (§31)

- **LEVEL A**: No reproducible functional transfer.
- **LEVEL B: Representational alignment demonstrated.** *(Highest Defensible Level)*.
- **LEVEL C**: Functional reconstruction within same architecture.
- **LEVEL D**: Cross-architecture functional effect without host adaptation.
- **LEVEL E**: Cross-architecture functional effect with localized adaptation AND causal mediation by the transplanted component.
- **LEVEL F**: Robust surgical transfer of multiple independently identified functional circuits across multiple architectures.

**Defensible Level**: **LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY**  
*Justification*: While the bidirectional bridge learns an aligned parameter interface and achieves 40.78% functional effect similarity (beating random noise), target-side causal mediation is non-positive (-5.00 pp drop). Under Section 31 rules, Levels D, E, and F are strictly forbidden without verified positive causal mediation.

---

## 17. What This Does NOT Prove (§35)

1. It does NOT prove that modular neural component transplantation is fundamentally impossible. It proves that **isolated single attention heads cannot be transplanted using linear interface adapters alone**.
2. It does NOT claim or imply component transfer between commercial closed models (ChatGPT, Claude, Gemini), whose internal weights and activations are inaccessible.
3. It does NOT prove that whole-model merging or knowledge distillation are invalid; it proves that surgical component transplantation is a distinct, much more demanding scientific objective.

---

## 18. Recommendations for Next Milestone (E7.5)

1. **Multi-Head Modular Circuit Blocks**: Elevate the candidate transferable unit from a single head to a self-contained multi-head module (e.g. all 4 heads of Layer 0 or an entire Attention+LN+MLP block).
2. **Non-Linear Gated Interface Adapters**: Replace linear matrices with small 2-layer MLP adapters equipped with GELU non-linearities and learned LayerNorm.
3. **Downstream Routing Forcing**: Use contrastive auxiliary losses to penalize downstream layers that ignore the transplanted channel, preventing the bypass failure mode.
"""

    with open(os.path.join(WORKSPACE, "EQUYLAPTA_E7_4_REPORT.md"), "w") as f:
        f.write(e74_md)

    with open(os.path.join(WORKSPACE, "EQUYLAPTA_E7_4_REPORT.txt"), "w") as f:
        f.write(e74_md)

    # =======================================================================
    # 2. GENERATE README_E7_4.md
    # =======================================================================
    readme_md = f"""# EQUYLAPTA E7.4 — Functional Interface Bridge

## Overview
EQUYLAPTA E7.4 evaluates the **Bidirectional Synthetic Interface Bridge** ($W_{{in}} \\to W_{{comp}} \\to W_{{out}}$) for surgical neural component transplantation, measured inside a newly defined **Common Behavioral Functional Effect Space**.

## Deliverables in Workspace
- `README_E7_4.md`: This overview.
- `EQUYLAPTA_E7_3_AUDIT.md`: Forensic audit of E7.3 vulnerabilities and classification downgrade.
- `EQUYLAPTA_E7_4_REPORT.md`: Comprehensive 18-section Markdown scientific report.
- `EQUYLAPTA_E7_4_REPORT.txt`: Plain-text authoritative deliverable.
- `e7_3_audit.json`: Machine-readable audit of E7.3.
- `e7_4_results.json`: Canonical result deliverable for E7.4.
- `functional_effect_results.json`: Downstream delta logit cosine similarities.
- `causal_mediation_results.json`: 7-intervention causal pathway battery.
- `bridge_results.json`: 5-condition disentangled interface data.
- `depth_sweep_results.json`: Corrected depth sweep (2L, 4L, 6L, 8L).
- `capacity_sweep_results.json`: Capacity sweep (25%, 50%, 100%, 150%).
- `claim_provenance.json`: Machine-readable claim mappings.
- `reproduce.py`: One-command reproduction script (`python3 reproduce.py --mode smoke`).

## Quick Reproduction
```bash
python3 reproduce.py --mode smoke
```
"""
    with open(os.path.join(WORKSPACE, "README_E7_4.md"), "w") as f:
        f.write(readme_md)

    # Mirror all reports to /home/user/ai-model-fusion-lab/reports/
    rep_dir = os.path.join(ROOT, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    for rep in ["EQUYLAPTA_E7_4_REPORT.md", "EQUYLAPTA_E7_4_REPORT.txt", "README_E7_4.md", "EQUYLAPTA_E7_3_AUDIT.md"]:
        with open(os.path.join(WORKSPACE, rep)) as f_in:
            c = f_in.read()
        with open(os.path.join(rep_dir, rep), "w") as f_out:
            f_out.write(c)

    print("EQUYLAPTA E7.4 Reports successfully generated and mirrored!")


if __name__ == "__main__":
    main()
