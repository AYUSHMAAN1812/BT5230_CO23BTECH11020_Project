from ocsr_project.chemistry import canonicalize_smiles, tanimoto_similarity


def test_canonicalization_equivalence():
    assert canonicalize_smiles("C1=CC=CC=C1") == canonicalize_smiles("c1ccccc1")


def test_invalid_prediction_scores_zero():
    assert tanimoto_similarity("CCO", "not-smiles") == 0.0

