import pytest

from ocsr_project.tokenizer import SmilesTokenizer, tokenize_smiles


@pytest.mark.parametrize(
    "smiles",
    ["CCO", "c1ccccc1", "N[C@@H](C)C(=O)O", "[O-][N+](=O)c1ccccc1", "ClCBr"],
)
def test_lossless_tokenization(smiles):
    assert "".join(tokenize_smiles(smiles)) == smiles


def test_tokenizer_round_trip():
    values = ["CCO", "c1ccccc1", "N[C@@H](C)C(=O)O"]
    tokenizer = SmilesTokenizer.build(values)
    for value in values:
        assert tokenizer.decode(tokenizer.encode(value)) == value


def test_unsupported_character_fails():
    with pytest.raises(ValueError):
        tokenize_smiles("CCO_")


def test_tokenizer_save_load(tmp_path):
    tokenizer = SmilesTokenizer.build(["CCO", "ClCBr"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = SmilesTokenizer.load(path)
    assert restored.token_to_id == tokenizer.token_to_id
