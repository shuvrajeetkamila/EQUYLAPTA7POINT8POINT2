PATTERN GENOME RESULTS — Milestone 4
==========================================================================

Patterns registered: 100 (hypotheses, never facts)
Patterns tested: 89   statuses: {'PROMISING': 10, 'UNSUPPORTED': 49, 'NOT_TESTED': 30}
Evidence ladder histogram (tested patterns): L0 NOT_TESTED: 43, L1 PATTERN_DETECTED: 27, L2 CORRELATED_WITH_CAPABILITY: 9, L3 ABLATION_EFFECT: 2, L5 POSITIVE_INTERVENTION: 8

-- Circuits --
A-math-convergence-01 [convergence] model=A-math capability=math members=[{'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 0}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 1}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 2}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 3}, {'level': 'MODULE', 'layer': 0, 'module': 'attn'}] status=PROMISING L5
    MINIMAL UNIT: ['L0.h3'] retains 100.0% of full-circuit effect (25.0 of 25.0) verified=True
A-math-convergence-02 [convergence] model=A-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 0}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 1}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 2}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 3}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}] status=PROMISING L5
    MINIMAL UNIT: ['L0.h3'] retains 120.0% of full-circuit effect (25.0 of 20.83) verified=True
A-math-divergence-03 [divergence] model=A-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'mlp'}, {'level': 'MODULE', 'layer': 1, 'module': 'attn'}] status=NOT_TESTED L0
A-math-divergence-04 [divergence] model=A-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}] status=UNSUPPORTED L1
    MINIMAL UNIT: ['L0.attn'] retains 120.0% of full-circuit effect (25.0 of 20.83) verified=True
A-math-hub_and_spokes-05 [hub_and_spokes] model=A-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}, {'level': 'MODULE', 'layer': 1, 'module': 'attn'}] status=UNSUPPORTED L1
    MINIMAL UNIT: ['L0.attn'] retains 85.7% of full-circuit effect (25.0 of 29.17) verified=True
A-math-redundant_paths-06 [redundant_paths] model=A-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}] status=NOT_TESTED L0
A-math-fork_recombine-07 [fork_recombine] model=A-math capability=math members=[{'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 0}, {'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}] status=PROMISING L5
    MINIMAL UNIT: ['L0.h0'] retains 160.0% of full-circuit effect (33.33 of 20.83) verified=True
A-math-community-08 [community] model=A-math capability=math members=[{'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 0}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 1}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 2}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 3}] status=NOT_TESTED L0
    MINIMAL UNIT: ['L0.h1'] retains 266.6% of full-circuit effect (33.33 of 12.5) verified=True
A-math-routing-09 [routing] model=A-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'attn'}] status=UNSUPPORTED L1
    MINIMAL UNIT: ['L0.attn', 'L0.attn'] retains 100.0% of full-circuit effect (25.0 of 25.0) verified=True
deep-math-convergence-01 [convergence] model=deep-math capability=math members=[{'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 0}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 1}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 2}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 3}, {'level': 'MODULE', 'layer': 0, 'module': 'attn'}] status=PROMISING L5
    MINIMAL UNIT: ['L0.h2'] retains 118.2% of full-circuit effect (54.16 of 45.83) verified=True
deep-math-convergence-02 [convergence] model=deep-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 0}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 1}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 2}, {'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 3}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}] status=PROMISING L4
    MINIMAL UNIT: ['L0.h2'] retains 100.0% of full-circuit effect (54.16 of 54.16) verified=True
deep-math-divergence-03 [divergence] model=deep-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'mlp'}, {'level': 'MODULE', 'layer': 1, 'module': 'attn'}] status=UNSUPPORTED L1
    MINIMAL UNIT: ['L0.mlp', 'L1.attn'] retains 100.0% of full-circuit effect (50.0 of 50.0) verified=True
deep-math-divergence-04 [divergence] model=deep-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}] status=UNSUPPORTED L1
    MINIMAL UNIT: ['L0.attn'] retains 84.6% of full-circuit effect (45.83 of 54.16) verified=True
deep-math-hub_and_spokes-05 [hub_and_spokes] model=deep-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}, {'level': 'MODULE', 'layer': 1, 'module': 'attn'}] status=UNSUPPORTED L1
    MINIMAL UNIT: ['L0.attn'] retains 110.0% of full-circuit effect (45.83 of 41.66) verified=True
deep-math-redundant_paths-06 [redundant_paths] model=deep-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 2, 'module': 'attn'}] status=NOT_TESTED L0
    MINIMAL UNIT: ['L0.attn', 'L2.attn'] retains 100.0% of full-circuit effect (58.33 of 58.33) verified=True
deep-math-fork_recombine-07 [fork_recombine] model=deep-math capability=math members=[{'level': 'HEAD', 'layer': 0, 'module': 'attn', 'index': 0}, {'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}] status=PROMISING L4
    MINIMAL UNIT: ['L0.attn'] retains 84.6% of full-circuit effect (45.83 of 54.16) verified=True
deep-math-community-08 [community] model=deep-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}, {'level': 'MODULE', 'layer': 0, 'module': 'mlp'}, {'level': 'HEAD', 'layer': 1, 'module': 'attn', 'index': 0}, {'level': 'HEAD', 'layer': 1, 'module': 'attn', 'index': 1}, {'level': 'HEAD', 'layer': 1, 'module': 'attn', 'index': 2}, {'level': 'HEAD', 'layer': 1, 'module': 'attn', 'index': 3}] status=PROMISING L3
    MINIMAL UNIT: ['L0.attn'] retains 122.2% of full-circuit effect (45.83 of 37.5) verified=True
deep-math-bridge-09 [bridge] model=deep-math capability=math members=[{'level': 'MODULE', 'layer': 0, 'module': 'attn'}] status=UNSUPPORTED L1
compound-A-math-convergence-01+A-math-convergence-02 [compound] model=A-math capability=math members=[{'layer': 0, 'level': 'MODULE', 'module': 'mlp'}, {'index': 3, 'layer': 0, 'level': 'HEAD', 'module': 'attn'}, {'index': 0, 'layer': 0, 'level': 'HEAD', 'module': 'attn'}, {'index': 1, 'layer': 0, 'level': 'HEAD', 'module': 'attn'}, {'layer': 0, 'level': 'MODULE', 'module': 'attn'}, {'index': 2, 'layer': 0, 'level': 'HEAD', 'module': 'attn'}] status=PROMISING L3

-- Pattern evidence (tested only) --
P001 Fibonacci spacing [B] -> L0 UNSUPPORTED
    reason: gap structure of significant layers [0, 1] not distinguishable from random at this depth (best fit fibonacci, p=1.0)
    detection: {'significant': False, 'p': 1.0, 'stat': None}
P002 Golden ratio [B] -> L0 UNSUPPORTED
    reason: phi-bin fraction 0.0 not above random null (p=1.0)
    detection: {'significant': False, 'p': 1.0, 'stat': None}
P003 Golden-angle spacing [B] -> L0 NOT_TESTED
    reason: golden-angle angular analysis needs >=8 principal directions with task links; deferred (honest underpowerment)
P004 Prime spacing [B] -> L0 UNSUPPORTED
    reason: gap structure of significant layers [0, 1] not distinguishable from random at this depth (best fit fibonacci, p=1.0)
    detection: {'significant': False, 'p': 1.0, 'stat': None}
P005 Power-law contribution [B] -> L0 UNSUPPORTED
    reason: magnitude family best fit = exponential (power-law r2=0.7057); no heavy-tail evidence
    detection: {'significant': False, 'p': 0.5, 'stat': 0.7057}
P006 Exponential depth growth [B] -> L0 UNSUPPORTED
    reason: depth-curve best fit = insufficient (needs exp/log/sqrt); contribution profile is flat or linear at this scale
    detection: {'significant': False, 'p': None, 'stat': None}
P007 Logarithmic scaling [B] -> L0 UNSUPPORTED
    reason: depth-curve best fit = insufficient (needs exp/log/sqrt); contribution profile is flat or linear at this scale
    detection: {'significant': False, 'p': None, 'stat': None}
P008 Square-root scaling [B] -> L0 UNSUPPORTED
    reason: depth-curve best fit = insufficient (needs exp/log/sqrt); contribution profile is flat or linear at this scale
    detection: {'significant': False, 'p': None, 'stat': None}
P009 Inverse-square decay [B] -> L1 UNSUPPORTED
    reason: pairwise attention-activation coupling vs layer distance: inverse-square-normalized coupling CV=0.00; with <=3 distance bins this cannot distinguish 1/d2 from exponential decay (underpowered, honest)
P010 Geometric progression [B] -> L0 UNSUPPORTED
    reason: gap structure of significant layers [0, 1] not distinguishable from random at this depth (best fit fibonacci, p=1.0)
    detection: {'significant': False, 'p': 1.0, 'stat': None}
P011 Arithmetic progression [B] -> L0 UNSUPPORTED
    reason: gap structure of significant layers [0, 1] not distinguishable from random at this depth (best fit fibonacci, p=1.0)
    detection: {'significant': False, 'p': 1.0, 'stat': None}
P012 Harmonic series [B] -> L0 UNSUPPORTED
    reason: magnitude family best fit = exponential (power-law r2=0.7057); no heavy-tail evidence
    detection: {'significant': False, 'p': 0.5, 'stat': 0.7057}
P013 Lucas sequence [B] -> L0 UNSUPPORTED
    reason: gap structure of significant layers [0, 1] not distinguishable from random at this depth (best fit fibonacci, p=1.0)
    detection: {'significant': False, 'p': 1.0, 'stat': None}
P014 Pascal branching [D] -> L0 NOT_TESTED
    reason: Phase D compound machinery (combinatorial branching needs validated pathways first)
P015 Binomial structure [D] -> L0 NOT_TESTED
    reason: Phase D compound machinery (combinatorial branching needs validated pathways first)
P016 Fractal self-similarity [B] -> L0 NOT_TESTED
    reason: needs >=4 layers for cross-scale comparison
P017 Recursion [B] -> L0 NOT_TESTED
    reason: needs >=4 layers for cross-scale comparison
P018 Symmetry [B] -> L1 UNSUPPORTED
    reason: layer-swap symmetry of causal profiles: layer effects ['45.8', '37.5', '12.5', '37.5'] are not exchangeable (first vs last differ); no symmetry evidence
P019 Rotational symmetry [B] -> L0 NOT_TESTED
    reason: subspace-geometry rig deferred (Phase B low priority at this scale; primitives exist)
P020 Reflection symmetry [B] -> L0 NOT_TESTED
    reason: subspace-geometry rig deferred (Phase B low priority at this scale; primitives exist)
P023 Series-parallel network [A] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.03, 'stat': 2}
    task_correlation: {'significant': True, 'selectivity': 27.08}
    ablation: {'effect': 20.83, 'significant': True, 'effect_specific': False, 'specificity': -32.3, 'max_single': 33.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 41.66}
P024 Convergence (all-roads-to-Rome) [A] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.02, 'stat': 4}
    task_correlation: {'significant': True, 'selectivity': 38.54}
    ablation: {'effect': 45.83, 'significant': True, 'effect_specific': True, 'specificity': 8.33, 'max_single': 58.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P025 Divergence / branching [A] -> L1 UNSUPPORTED
    detection: {'significant': True, 'p': 0.02, 'stat': 2}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 54.16, 'significant': True, 'effect_specific': True, 'specificity': 22.91, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 20.83}
P026 Fan-in combination [A] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.02, 'stat': 4}
    task_correlation: {'significant': True, 'selectivity': 38.54}
    ablation: {'effect': 45.83, 'significant': True, 'effect_specific': True, 'specificity': 8.33, 'max_single': 58.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P029 Recurrent loop (depth recurrence) [B] -> L0 NOT_TESTED
    reason: needs >=4 layers for cross-scale comparison
P030 Hub-and-spoke [A] -> L0 UNSUPPORTED
    reason: hub candidate attn0 weighted-degree=3.02, degree-preserving null p=1.000; detection-only (no causal battery beyond circuit above)
    detection: {'significant': True, 'p': 0.03, 'stat': 1}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 41.66, 'significant': True, 'effect_specific': True, 'specificity': 10.41, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 16.67}
P031 Mesh network [B] -> L1 UNSUPPORTED
    reason: graph density 0.170 vs random-DAG nulls; no capability association without a causal battery
P032 Tree hierarchy [B] -> L1 UNSUPPORTED
    reason: hierarchical clustering over <=8 graph nodes is trivial (any set clusters); no capability association without branch-removal effects, which the circuit batteries cover
P033 DAG of transformations [B] -> L1 UNSUPPORTED
    reason: the dependency graph is a DAG by architectural construction; DAG-ness itself is not capability evidence (§22 anti-pattern)
P034 Small-world network [B] -> L1 UNSUPPORTED
    reason: density=0.170, mean embed->final path length 8.6; architecture is a shallow chain by construction — small-world structure is not expressible at this depth (honest architectural limit)
P035 Scale-free network [B] -> L1 UNSUPPORTED
    detection: {'significant': True, 'p': 0.03, 'stat': 1}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 41.66, 'significant': True, 'effect_specific': True, 'specificity': 10.41, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 16.67}
P036 Community / modular network [A] -> L3 PROMISING
    detection: {'significant': True, 'p': 0.05, 'stat': 1}
    task_correlation: {'significant': True, 'selectivity': 8.85}
    ablation: {'effect': 37.5, 'significant': True, 'effect_specific': False, 'specificity': -6.25, 'max_single': 45.83}
    restoration: {'recovered_pct': 28.6}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P037 Bottleneck [A] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.02, 'stat': 4}
    task_correlation: {'significant': True, 'selectivity': 38.54}
    ablation: {'effect': 45.83, 'significant': True, 'effect_specific': True, 'specificity': 8.33, 'max_single': 58.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P038 Bridge / connector [A] -> L1 UNSUPPORTED
    detection: {'significant': True, 'p': 0.05, 'stat': 1.0}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 45.83, 'significant': True, 'effect_specific': True, 'specificity': 8.33, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 8.34}
P040 Redundant parallel paths [A] -> L0 NOT_TESTED
    detection: {'significant': False, 'p': 0.04, 'stat': 1}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 58.33, 'significant': True, 'effect_specific': True, 'specificity': 20.83, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 16.67}
P041 Neural hierarchy [C] -> L1 UNSUPPORTED
    reason: task-vs-general residual decodability by depth [1.0, 1.0]; no monotone abstraction trend test passed at this depth
P042 Lateral inhibition [C] -> L1 UNSUPPORTED
    reason: most-negative node-pair activation correlation = -0.647; without a competition-intervention rig this stays detection-tier
P043 Excitation/inhibition balance [C] -> L1 UNSUPPORTED
    reason: sign balance measurable only via signed contribution batteries (P055 covers the negative side)
P044 Sparse coding [A] -> L2 UNSUPPORTED
    reason: mean Hoyer sparsity task=0.225 vs general=0.238; sparse-coding CAUSAL tier delegated to the sparse_window circuit battery (channel ablation)
P046 Cellular specialization [C] -> L1 UNSUPPORTED
    reason: role clusters degenerate at <=26 nodes (communities reported in Phase A circuits)
P047 Functional redundancy [C] -> L2 UNSUPPORTED
    reason: leave-one-out redundancy captured in circuit batteries (leave_one_out_effect)
P048 Modularity [A] -> L3 PROMISING
    detection: {'significant': True, 'p': 0.05, 'stat': 1}
    task_correlation: {'significant': True, 'selectivity': 8.85}
    ablation: {'effect': 37.5, 'significant': True, 'effect_specific': False, 'specificity': -6.25, 'max_single': 45.83}
    restoration: {'recovered_pct': 28.6}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P049 Hierarchical specialization [C] -> L0 NOT_TESTED
    reason: needs >=2 clustering levels; degenerate at this size
P050 Neural ensemble [C] -> L2 UNSUPPORTED
    reason: ensemble candidates = multi-member circuits in Phase A batteries
P051 Population coding [C] -> L2 UNSUPPORTED
    detection: {'significant': True, 'p': 0.0244, 'stat': 1.0}
    task_correlation: {'significant': True, 'selectivity': 50.0}
P052 Competitive selection [C] -> L2 UNSUPPORTED
    reason: competition mapped to anti-correlation P042 (detection-tier)
P053 Winner-take-all [C] -> L2 UNSUPPORTED
    reason: top-8 activation concentration task/general ratio = 0.952; concentration alone is not capability evidence (§22)
P054 Cooperative activation [C] -> L2 UNSUPPORTED
    reason: cooperation = factorial interaction; measured inside Phase A batteries (joint_vs_sum)
P056 Signal amplification [C] -> L1 UNSUPPORTED
    reason: mlp/resid gain profile [0.472, 0.548]; gain>1 stages exist but no causal amplifier test beyond the circuit batteries
P057 Signal attenuation [C] -> L1 UNSUPPORTED
    reason: attenuation stages in [0.472, 0.548] (gain<1); detection-tier
P058 Synaptic plasticity locus [C] -> L0 NOT_TESTED
    reason: needs training-loop gradient instrumentation (plasticity); deferred
P059 Hebbian association [C] -> L2 UNSUPPORTED
    reason: split-half coactivation stability = 0.979 (associations exist; causal break-test not run this pass)
P060 Homeostasis [C] -> L2 UNSUPPORTED
    reason: activation statistics stable across inputs by construction (pre-LN); no perturbation rig this pass
P061 Series resistance [C] -> L1 UNSUPPORTED
    reason: process metaphor mapped to measurable depth profiles (gain/selectivity above); no independent causal evidence this pass
P062 Parallel resistance [C] -> L0 NOT_TESTED
    detection: {'significant': False, 'p': 0.04, 'stat': 1}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 58.33, 'significant': True, 'effect_specific': True, 'specificity': 20.83, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 16.67}
P063 Current splitting [C] -> L1 UNSUPPORTED
    detection: {'significant': True, 'p': 0.02, 'stat': 2}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 54.16, 'significant': True, 'effect_specific': True, 'specificity': 22.91, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 20.83}
P064 Current convergence [C] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.02, 'stat': 4}
    task_correlation: {'significant': True, 'selectivity': 38.54}
    ablation: {'effect': 45.83, 'significant': True, 'effect_specific': True, 'specificity': 8.33, 'max_single': 58.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P065 Voltage divider [B] -> L0 UNSUPPORTED
    reason: phi-bin fraction 0.0 not above random null (p=1.0)
    detection: {'significant': False, 'p': 1.0, 'stat': None}
P066 Feedback control [C] -> L0 NOT_TESTED
    reason: perturbation rig failed: cannot import name '_forward_cache' from 'pattern_genome.intervention' (/home/user/ai-model-fusion-lab/pattern_genome/intervention.py)
P067 Negative feedback [C] -> L0 NOT_TESTED
    reason: perturbation rig failed: cannot import name '_forward_cache' from 'pattern_genome.intervention' (/home/user/ai-model-fusion-lab/pattern_genome/intervention.py)
P068 Positive feedback [C] -> L0 NOT_TESTED
    reason: perturbation rig failed: cannot import name '_forward_cache' from 'pattern_genome.intervention' (/home/user/ai-model-fusion-lab/pattern_genome/intervention.py)
P069 Resonance [B] -> L0 NOT_TESTED
    reason: needs repeated-input response rig; deferred
P070 Phase alignment [B] -> L0 NOT_TESTED
    reason: ordering perturbation not expressible without weight surgery; deferred
P071 Impedance matching [C] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.02, 'stat': 4}
    task_correlation: {'significant': True, 'selectivity': 38.54}
    ablation: {'effect': 45.83, 'significant': True, 'effect_specific': True, 'specificity': 8.33, 'max_single': 58.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P072 Signal filtering [C] -> L0 NOT_TESTED
    reason: information-survival rig deferred (decode-through-node)
P073 Low-pass filtering [B] -> L0 NOT_TESTED
    reason: no frequency-ordered basis in residual channels; honest architectural mismatch (channels are not spectral bins)
P074 High-pass filtering [B] -> L0 NOT_TESTED
    reason: same as P073
P075 Band-pass filtering [B] -> L0 NOT_TESTED
    reason: same as P073
P076 Amplifier chain [C] -> L1 UNSUPPORTED
    reason: process metaphor mapped to measurable depth profiles (gain/selectivity above); no independent causal evidence this pass
P077 Representation transformer [B] -> L0 NOT_TESTED
    reason: mapping quality = linear map between node representations; covered by alignment subsystem (M3) — detection deferred
P078 Bridge circuit (comparison) [C] -> L0 NOT_TESTED
    reason: difference-stream decodability deferred
P079 Switch / gating [A] -> L1 UNSUPPORTED
    detection: {'significant': True, 'p': 0.04, 'stat': 0.358}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 25.0, 'significant': True, 'effect_specific': False, 'specificity': -12.5, 'max_single': 25.0}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 25.0}
P080 Multiplexer / routing [A] -> L1 UNSUPPORTED
    detection: {'significant': True, 'p': 0.04, 'stat': 0.358}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 25.0, 'significant': True, 'effect_specific': False, 'specificity': -12.5, 'max_single': 25.0}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 25.0}
P082 Raw->refined->product [C] -> L1 UNSUPPORTED
    reason: process metaphor mapped to measurable depth profiles (gain/selectivity above); no independent causal evidence this pass
P083 Filtering->sorting->assembly [C] -> L0 NOT_TESTED
    reason: entropy-stage rig deferred
P084 Mining->refining->manufacturing [C] -> L1 UNSUPPORTED
    reason: process metaphor mapped to measurable depth profiles (gain/selectivity above); no independent causal evidence this pass
P085 Seed->plant->fruit [C] -> L1 UNSUPPORTED
    reason: process metaphor mapped to measurable depth profiles (gain/selectivity above); no independent causal evidence this pass
P086 Digestion pipeline [C] -> L0 NOT_TESTED
    reason: decomposition-reconstruction rig deferred
P089 Assembly line [C] -> L1 UNSUPPORTED
    reason: per-layer specialization profile [7.3, 22.92] (selectivity points); stations exist but causal per-station effects are in the module batteries
P090 Quality-control gate [C] -> L1 UNSUPPORTED
    reason: activation-correctness correlations {'embed': 0.611, 'final': 0.323} (null p=0.010); detection-tier only
P091 Fork->process->recombine [A] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.03, 'stat': 2}
    task_correlation: {'significant': True, 'selectivity': 27.08}
    ablation: {'effect': 20.83, 'significant': True, 'effect_specific': False, 'specificity': -32.3, 'max_single': 33.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 41.66}
P092 Map->route->destination [C] -> L0 NOT_TESTED
    reason: trajectory-structure rig deferred
P093 Multiple paths, common destination [A] -> L0 NOT_TESTED
    detection: {'significant': False, 'p': 0.04, 'stat': 1}
    task_correlation: {'significant': False, 'selectivity': 0.0}
    ablation: {'effect': 58.33, 'significant': True, 'effect_specific': True, 'specificity': 20.83, 'max_single': 45.83}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 16.67}
P094 Evolution->selection->recombination [D] -> L0 NOT_TESTED
    reason: evolution is gated on a validated transfer existing (standing M3 rule); no validated transfer this run — machinery deliberately dormant
P095 Mutation->selection->retention [D] -> L0 NOT_TESTED
    reason: evolution is gated on a validated transfer existing (standing M3 rule); no validated transfer this run — machinery deliberately dormant
P096 Ant-colony path reinforcement [D] -> L0 NOT_TESTED
    reason: evolution is gated on a validated transfer existing (standing M3 rule); no validated transfer this run — machinery deliberately dormant
P097 Bee-swarm specialization [C] -> L1 UNSUPPORTED
    reason: role clusters degenerate at <=26 nodes (communities reported in Phase A circuits)
P098 River branching & convergence [A] -> L5 PROMISING
    detection: {'significant': True, 'p': 0.02, 'stat': 4}
    task_correlation: {'significant': True, 'selectivity': 38.54}
    ablation: {'effect': 45.83, 'significant': True, 'effect_specific': True, 'specificity': 8.33, 'max_single': 58.33}
    restoration: {'recovered_pct': 100.0}
    positive_intervention: {'significant': True, 'max_abs_change': 12.5}
P100 Cellular differentiation [C] -> L1 UNSUPPORTED
    reason: process metaphor mapped to measurable depth profiles (gain/selectivity above); no independent causal evidence this pass