"""tests/test_trainable_parameter_consistency.py

Verifies 100% mathematical consistency among:
1. Declared trainable parameters in training_objective.json
2. Optimizer registered parameters in ArchitectureTranslator
3. Gradient-bearing parameters during backward pass
4. Parameter values that actually changed after training
"""
import json
import os
import sys
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.synthetic import load_model, gen_suite
from src.transfer import ArchitectureTranslator


def test_parameter_consistency():
    obj_path = os.path.join(os.path.dirname(__file__), "..", "training_objective.json")
    with open(obj_path, "r") as f:
        obj_data = json.load(f)
    declared_params = set(obj_data["trainable_parameters"])

    donor = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e4-math-4L"))
    recip = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e6-base-4L"))

    # Instantiate Condition E which contains both head and mlp bridges
    translator = ArchitectureTranslator(
        donor, recip,
        active_elements=["L0_head_2", "L0_mlp"],
        condition_name="Condition_E_Recipient_Reconstructed_FDC",
        seed=42
    )

    optimizer_params = set(translator.get_trainable_parameter_names())

    # Initial parameter copy
    init_params = {k: v.copy() for k, v in translator.get_trainable_parameters().items()}

    # Run 1 epoch on 3 items to register gradients and updates
    items = gen_suite("math", n=3, seed=888)
    train_res = translator.train_adapters(items, epochs=1, lr=0.01)

    # Gradient-bearing parameters
    gradient_params = set(k for k, g in translator.last_gradients.items() if np.linalg.norm(g) > 1e-7)

    # Changed parameters
    final_params = translator.get_trainable_parameters()
    changed_params = set(k for k in optimizer_params if np.linalg.norm(final_params[k] - init_params[k]) > 1e-7)

    print("Declared:", declared_params)
    print("Optimizer:", optimizer_params)
    print("Gradient-bearing:", gradient_params)
    print("Changed:", changed_params)

    # Assert 100% agreement
    assert declared_params == optimizer_params, f"Mismatch declared vs optimizer: {declared_params ^ optimizer_params}"
    assert declared_params == gradient_params, f"Mismatch declared vs gradient: {declared_params ^ gradient_params}"
    assert declared_params == changed_params, f"Mismatch declared vs changed: {declared_params ^ changed_params}"


if __name__ == "__main__":
    test_parameter_consistency()
    print("test_trainable_parameter_consistency passed successfully!")
