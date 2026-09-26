"""tests/test_parameter_update.py

Verifies ||theta_after - theta_before|| > 0 strictly holds for all declared trainable adapters.
"""
import os
import sys
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.synthetic import load_model, gen_suite
from src.transfer import ArchitectureTranslator


def test_parameter_update_norm():
    donor = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e4-math-4L"))
    recip = load_model(os.path.join(WORKSPACE, "fusionlab_data", "models", "e6-base-4L"))

    translator = ArchitectureTranslator(
        donor, recip,
        active_elements=["L0_head_2", "L0_mlp"],
        condition_name="Condition_E_Recipient_Reconstructed_FDC",
        seed=42
    )

    init_params = {k: v.copy() for k, v in translator.get_trainable_parameters().items()}
    items = gen_suite("math", n=5, seed=888)
    train_res = translator.train_adapters(items, epochs=2, lr=0.005)

    final_params = translator.get_trainable_parameters()

    for p_name in translator.get_trainable_parameter_names():
        update_norm = float(np.linalg.norm(final_params[p_name] - init_params[p_name]))
        print(f"Parameter: {p_name} | Update Norm: {update_norm:.8f}")
        assert update_norm > 1e-6, f"Parameter {p_name} did not update! (norm={update_norm})"


if __name__ == "__main__":
    test_parameter_update_norm()
    print("test_parameter_update passed successfully!")
