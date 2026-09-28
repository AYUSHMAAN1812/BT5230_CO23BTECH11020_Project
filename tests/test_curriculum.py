import pandas as pd
import pytest

from ocsr_project.curriculum import select_short_molecules, training_phase


def test_curriculum_selects_training_only_in_original_order():
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c"],
            "canonical_smiles": ["CCO", "C" * 41, "ClCCl"],
            "split": ["train", "train", "train"],
        }
    )
    selected = select_short_molecules(frame, max_smiles_tokens=4)
    assert selected["compound_id"].tolist() == ["a", "c"]
    assert training_phase(1, 2) == "short"
    assert training_phase(2, 2) == "short"
    assert training_phase(3, 2) == "full"


def test_curriculum_rejects_validation_and_empty_selection():
    frame = pd.DataFrame(
        {"canonical_smiles": ["CCO"], "split": ["validation"]}
    )
    with pytest.raises(ValueError, match="only training"):
        select_short_molecules(frame, max_smiles_tokens=4)
    frame["split"] = "train"
    with pytest.raises(ValueError, match="no training molecules"):
        select_short_molecules(frame, max_smiles_tokens=1)
