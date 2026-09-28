import pandas as pd
import pytest

from ocsr_project.comparison import compare_evaluation_frames


def frame(ids, valid, tanimoto):
    return pd.DataFrame(
        {
            "image_id": ids,
            "reference_smiles": ["CCO", "CCN"],
            "valid": valid,
            "exact_isomeric_match": [False, False],
            "exact_nonisomeric_match": [False, False],
            "tanimoto": tanimoto,
        }
    )


def test_paired_comparison_aligns_ids_and_estimates_delta():
    baseline = frame(["a", "b"], [False, True], [0.0, 0.2])
    candidate = frame(["b", "a"], [True, True], [0.5, 0.1])
    candidate["reference_smiles"] = ["CCN", "CCO"]
    result = compare_evaluation_frames(baseline, candidate)
    assert result["count"] == 2
    assert result["metrics"]["valid"]["candidate_minus_baseline"] == 0.5
    assert result["metrics"]["tanimoto"]["candidate_minus_baseline"] == 0.2


def test_paired_comparison_rejects_mismatched_references():
    baseline = frame(["a", "b"], [False, True], [0.0, 0.2])
    candidate = frame(["a", "b"], [False, True], [0.0, 0.2])
    candidate.loc[0, "reference_smiles"] = "CCC"
    with pytest.raises(ValueError, match="identical references"):
        compare_evaluation_frames(baseline, candidate)
