# MODEL COMPATIBILITY REPORT (EQUYLAPTA M2)

Pairwise compatibility from MEASURED facts (arch, shapes, tokenizer, license) — never name similarity.

## Micro model family (shared arch + tokenizer)
- models: code-smith, generalist, logic-owl, math-wiz, polyglot, weak-base
- merge compatibility: **HIGH** (identical shapes; verified by shape gate). Note from M1/M2 experiments: merging independent models still LOSES capability; only same-base fine-tunes merged successfully — see experiment DB.

## Real models
- gpt2: gpt2 L12 d768 H12 V50257

## Cross-family matrix (micro vs real)
- gpt2 + logic-owl: shape_ok=False compat=0.15 -> DISTILLATION [WORKING micro] or ROUTING [WORKING]
- gpt2 + math-wiz: shape_ok=False compat=0.15 -> DISTILLATION [WORKING micro] or ROUTING [WORKING]
- gpt2 + polyglot: shape_ok=False compat=0.15 -> DISTILLATION [WORKING micro] or ROUTING [WORKING]
- gpt2 + weak-base: shape_ok=False compat=0.15 -> DISTILLATION [WORKING micro] or ROUTING [WORKING]

## Real-real pairs
- gpt2 + EleutherAI/pythia-70m: shape_ok=False (GPT-2 vs GPT-NeoX, d768 vs d512, different tokenizers) -> WEIGHT MERGE IMPOSSIBLE; fallback = text-level distillation or ROUTING (verified in M2 run)
- gauntlet-verified architectures (see models/arch_gauntlet.py): GPT-2, GPT-NeoX, Llama-style, Qwen-style, Mistral-style, Gemma-style — mechanics verified; capability work on tiny models only