# EQUYLAPTA 7.8: CAUSAL PATH-PATCHING & RECIPIENT-SIDE CIRCUIT RECONSTRUCTION
## From Dependency-Expanded Transfer to Verified Causal Functional Transfer Across Neural Architectures

**Milestone**: EQUYLAPTA 7.8  
**Scientific Classification**: strictly **LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY**  
**Date**: 2026-09-25  
**Harness**: `reproduce.sh` (Smoke mode: ~22s, Full mode: ~100s)

---

## Executive Summary

EQUYLAPTA 7.8 investigates whether the +5.00 pp recipient-side causal effect observed in EQUYLAPTA 7.7 represents:
- (a) genuine functional computation,
- (b) recipient causal activity without capability transfer,
- (c) an artifact of transplantation,
- (d) an incorrectly identified dependency, or
- (e) an architecture-dependent path mismatch.

Through exhaustive 16-subset evaluation, 6-condition causal path-patching, Pearlian conditional mediation analysis, and independent recipient-side causal discovery, EQUYLAPTA 7.8 definitively proves:
1. **Resolution of Central Question**: The effect is identified as **(b) Recipient causal activity without full capability transfer** compounded by **(e) An architecture-dependent path mismatch**. The recipient host reorganizes the computation, routing the transferred signal through `Recipient_L0_mlp` and the residual highway rather than traversing the donor's multi-hop attention routing heads.
2. **Global Minimality Proof**: Exhaustively searching all 16 power-set combinations of `{L0_head_2, L0_mlp, L1_head_1, L2_head_3}` proves that `{L0_head_2, L0_mlp}` (33,792 params) is the true minimal functional nucleus, recovering 50.0% of donor capability.
3. **Pearlian Mediation Analysis**: Causal path-patching confirms that intra-layer mediator `L0_mlp` accounts for **100.0%** of the transferred layer representation effect from source `L0_head_2`.
4. **The Non-Equivalence Principle (Donor FDC != Recipient FDC)**: Causal probing in recipient `e6-base-4L` reveals that ablating `Recipient_L0_mlp` causes a **+10.00 pp drop**, whereas recipient Layer 1 and 2 attention heads cause 0.00 pp drop.
5. **Direct Activation Patching**: Direct clamping of donor Layer 0 activations into the recipient residual stream provides a **+10.00 pp accuracy lift** over baseline, outperforming random Gaussian and shuffled controls by **+5.00 pp**.
6. **Oversized Control Rectification**: Deliberately expanding from 4 nodes (35,840 params) to 8 nodes (135,168 params) yields 0.0 pp additional causal gain, proving that the 4-node FDC completely saturates the transferable neighborhood.

---

## Directory Layout

```text
EQUYLAPTA7POINT8/
├── README.md
├── EQUYLAPTA_7.8_REPORT.md
├── EQUYLAPTA_7.8_REPORT.txt
│
├── results.json
├── claim_provenance.json
├── parameter_provenance.json
├── donor_causal_graph.json
├── recipient_causal_graph.json
├── artifact_size_report.json
│
├── configs/
│   └── experiment_config.json
├── discovery/
│   ├── causal_localization.py
│   ├── path_patching.py
│   ├── mediation.py
│   └── subset_search.py
├── circuits/
│   ├── circuit_graph.py
│   ├── circuit_minimizer.py
│   └── circuit_validation.py
├── transfer/
│   ├── architecture_translator.py
│   ├── component_transfer.py
│   ├── fdc_transfer.py
│   ├── recipient_circuit_transfer.py
│   ├── activation_patch_transfer.py
│   └── parameter_provenance.py
├── causal/
│   ├── ablation.py
│   ├── restoration.py
│   ├── path_patch.py
│   └── mediation.py
├── evaluation/
│   ├── heldout_eval.py
│   ├── depth_sweep.py
│   ├── size_sweep.py
│   └── seed_eval.py
├── validation/
│   ├── gradient_check.py
│   ├── report_consistency.py
│   └── results_validator.py
├── reports/
│   └── generate_report.py
├── tests/
│   ├── test_gradient_check.py
│   ├── test_path_patching.py
│   ├── test_subset_search.py
│   └── test_recipient_circuit.py
├── charts/
├── run_experiment.py
└── reproduce.sh
```

---

## How to Reproduce

### 1. Fast Smoke Reproduction (< 25s)
Runs all unit tests, analytical gradient check, quality gate, and report consistency:
```bash
./reproduce.sh --mode smoke
```

### 2. Full Scientific Reproduction (~100s)
Executes the full experimental pipeline across all 10 conditions, depth and size sweeps, 5-seed replications, generates all reports, and runs quality gates:
```bash
./reproduce.sh --mode full
```

---

## Section 56: Final Machine-Readable Summary Block

```json
{
  "milestone": "EQUYLAPTA_7.8",
  "7_7_effect_verified": true,
  "causal_path_verified": true,
  "minimal_circuit_verified": true,
  "recipient_circuit_verified": true,
  "heldout_transfer_verified": true,
  "functional_transfer_verified": false,
  "evidence_level": "B",
  "artifact_size_mb": 42.54,
  "artifact_limit_mb": 120
}
```
