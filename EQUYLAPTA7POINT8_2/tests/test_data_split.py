"""tests/test_data_split.py

Unit test confirming zero data leakage across characterization, validation, and held-out splits.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_splits import get_disjoint_splits


def test_strict_disjoint_splits():
    splits = get_disjoint_splits("math", n=20)
    char_prompts = set(it.prompt for it in splits["characterization"].items)
    val_prompts = set(it.prompt for it in splits["validation"].items)
    held_prompts = set(it.prompt for it in splits["heldout"].items)

    assert len(char_prompts) == 20
    assert len(val_prompts) == 20
    assert len(held_prompts) == 20

    assert len(char_prompts.intersection(val_prompts)) == 0, "Leakage between characterization and validation!"
    assert len(char_prompts.intersection(held_prompts)) == 0, "Leakage between characterization and held-out!"
    assert len(val_prompts.intersection(held_prompts)) == 0, "Leakage between validation and held-out!"


if __name__ == "__main__":
    test_strict_disjoint_splits()
    print("test_data_split passed!")
