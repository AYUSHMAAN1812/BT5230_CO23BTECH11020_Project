from ocsr_project.splitting import assert_split_integrity, scaffold_key, scaffold_split


def test_scaffold_split_keeps_benzene_family_together():
    smiles = ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "C1CCCCC1", "CCO", "CCN"]
    assignments = scaffold_split(smiles, seed=7)
    assert_split_integrity(smiles, assignments)
    benzene_splits = {
        split for value, split in zip(smiles, assignments, strict=True) if "c1ccccc1" in value
    }
    assert len(benzene_splits) == 1
    assert scaffold_key("c1ccccc1") == scaffold_key("Oc1ccccc1")

