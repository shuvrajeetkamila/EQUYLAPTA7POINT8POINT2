# EQUYLAPTA E7.3 — Surgical Functional Component Transplant with Target-Side Co-Adaptation

## Overview
EQUYLAPTA E7.3 establishes the first controlled investigation of **target-side receiver co-adaptation** for cross-architecture surgical neural component transplants.

## Key Deliverables in Workspace
- `README_E7_3.md`: This executive overview.
- `EQUYLAPTA_E7_2_1_REPORT.md`: Forensic report on E7.2.1 integrity corrections (frozen validation Procrustes).
- `EQUYLAPTA_E7_3_REPORT.md`: Authoritative comprehensive Markdown report (24 sections).
- `EQUYLAPTA_E7_3_REPORT.txt`: Plain-text authoritative deliverable.
- `source_component_record.json`: Complete causal, architectural, and capability record of `L0_head_2`.
- `results_e7_2_1.json`: Integrity correction benchmarks.
- `results_e7_3.json`: Authoritative canonical results file.
- `coadaptation_results.json`: Complete training trajectories across Radii 1, 2, and 3.
- `causal_results.json`: 5-step causal sequence data (`Baseline -> Transplant -> Ablated -> Restored -> Destroyed`).
- `depth_sweep_results.json`: Sweeps across 2L, 4L, 6L, 8L.
- `capacity_sweep_results.json`: Sweeps across 25%, 50%, 100%, 150% capacity.
- `claim_provenance.json`: Verified provenance mapping for every headline claim.
- `experiment_config.json`: Experimental hyperparameters and seeds.
- `reproduce.py`: Standalone reproduction script (`python3 reproduce.py --mode smoke`).

## Reproduction Command
```bash
python3 reproduce.py --mode smoke
```
