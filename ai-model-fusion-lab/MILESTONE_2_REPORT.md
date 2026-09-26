# MILESTONE 2 REPORT — EQUYLAPTA [AI MODEL]

Generated from run artifacts (no hand-typed numbers).
- experiments recorded: 646 (genome-ablation:520, evolution:71, merge:32, synthesis:13, merge-same-base:4, distillation:2, alignment:2, ablation:1, transplant:1)
- genomes on disk: code-smith.json, generalist.json, gpt2.json, logic-owl.json, math-wiz.json, weak-base.json

## 1-2. Fixed / added
1. Distillation shape-contract assertions with EXPECTED/RECEIVED/REASON errors.
2. HF head ablation via attention-output-slice hooks (+ per-arch attention
   module discovery + shape assertions); regression tests on gpt2 + pythia.
3. activation_autopsy for HF via hooks (residual/attn/mlp per layer).
4. SuperModel.stats() fix + route-once fix.
5. eval_suite single-pass scoring.
6. resume_stages.py dead-code removal.
7. Arch gauntlet: tiny-random Llama/Qwen2/Mistral/Gemma + real gpt2/pythia must
   pass LOAD→AUTOPSY→FORWARD→BENCHMARK→ABLATION before "supported".

All fixes verified by regression tests (23 tests, `python -m unittest tests.test_core`).

## 3-4. Real models tested / architectures supported
- gpt2 [GPT-2] -> **SUPPORTED** (LOAD:PASS, AUTOPSY:PASS, FORWARD:PASS, BENCHMARK:PASS, ABLATION:PASS)
- EleutherAI/pythia-70m [GPT-NeoX] -> **SUPPORTED** (LOAD:PASS, AUTOPSY:PASS, FORWARD:PASS, BENCHMARK:PASS, ABLATION:PASS)
- hf-internal-testing/tiny-random-LlamaForCausalLM [Llama-style] (random weights: mechanics only) -> **SUPPORTED** (LOAD:PASS, AUTOPSY:PASS, FORWARD:PASS, BENCHMARK:PASS, ABLATION:PASS)
- trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 [Qwen-style] (random weights: mechanics only) -> **SUPPORTED** (LOAD:PASS, AUTOPSY:PASS, FORWARD:PASS, BENCHMARK:PASS, ABLATION:PASS)
- hf-internal-testing/tiny-random-MistralForCausalLM [Mistral-style] (random weights: mechanics only) -> **SUPPORTED** (LOAD:PASS, AUTOPSY:PASS, FORWARD:PASS, BENCHMARK:PASS, ABLATION:PASS)
- hmellor/tiny-random-Gemma2ForCausalLM [Gemma-style] (random weights: mechanics only) -> **SUPPORTED** (LOAD:PASS, AUTOPSY:PASS, FORWARD:PASS, BENCHMARK:PASS, ABLATION:PASS)
- hf-internal-testing/tiny-random-GPTNeoXForCausalLM [GPT-NeoX (tiny)] (random weights: mechanics only) -> **SUPPORTED** (LOAD:PASS, AUTOPSY:PASS, FORWARD:PASS, BENCHMARK:PASS, ABLATION:PASS)

## 5-6. Components investigated / regions discovered
- code-smith: CHANNEL_GROUP:32, HEAD:16, LAYER:6, MODULE:10
- generalist: CHANNEL_GROUP:48, HEAD:24, LAYER:10, MODULE:12
- gpt2: HEAD:16
- logic-owl: CHANNEL_GROUP:40, HEAD:24, LAYER:6, MODULE:12
- math-wiz: CHANNEL_GROUP:40, HEAD:20, LAYER:6, MODULE:10
- weak-base: CHANNEL_GROUP:40, HEAD:24, LAYER:10, MODULE:14

## 7-10. Transfers, specialists (full details in reports/)
- EQUYLAPTA-math: status **REJECTED** — specialist 75.0% vs best parent 75.0% (best attempt did not beat best parent)
- EQUYLAPTA-coding: status **REJECTED** — specialist 100.0% vs best parent 100.0% (best attempt did not beat best parent)
- EQUYLAPTA-reasoning: status **REJECTED** — specialist 45.0% vs best parent 45.0% (best attempt did not beat best parent)
- evolution search best fitness 64.75 -> {'components': [['math-wiz', 'L1.mlp.chan[32:64]'], ['math-wiz', 'L1.attn.head1']], 'capability': 'micro-math', 'coefficient': 0.79, 'blend_with': 'generalist'} (metrics {'primary': 45.0, 'secondary': {'micro-code': 100.0, 'micro-lang': 70.0}, 'latency_ms': 0.5, 'fitness': 64.75})

## 11-12. Benchmarks
- see SPECIALIST_REGISTRY.json + per-specialist construction reports; held-out discipline: frozen private suites, train/val/test disjoint, chance + parent baselines recorded in each genome.

## 13. Computational requirements
- CPU-only sandbox, 2GB RAM: full M2 run ~35-45 min; quick mode ~8 min. Real-model phase needs ~1.5GB free for pythia-70m, ~1.7GB for gpt2.

## 14. License status
- micro sources Apache-2.0, gpt2 MIT -> COMPATIBLE; proprietary-closed sources -> NOT DISTRIBUTABLE (license engine enforced).

## 15. Known limitations
- micro suites measure the microcosm, not MMLU/HumanEval; gpt2 head scan is subsampled + low-n; alignment metrics saturated at n≈d; component refit/calibration adapters = future work; synthesis 'ADAPTER' mechanism is gate-only in this milestone.

## 16. Reproduce
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors huggingface_hub
python cli.py m2-demo            # full run
python cli.py m2-demo --quick --no-real   # micro-only quick
python cli.py gauntlet
python cli.py genome
python demo/make_m2_reports.py```