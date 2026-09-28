"""The Colab import step requires complete, ID-aligned clean validation predictions."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from import_molscribe_predictions import import_predictions


def test_import_rejects_missing_prediction(tmp_path: Path) -> None:
    manifest = ROOT / "data/manifests/chembl37_stage_5000_clean.csv"
    reference = pd.read_csv(manifest, dtype=str)
    validation = reference.loc[reference["split"] == "validation"]
    predictions = pd.DataFrame(
        {"image_id": validation["compound_id"].iloc[:-1], "post_SMILES": "CCO"}
    )
    path = tmp_path / "prediction_validation.csv"
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError, match="all 470"):
        import_predictions(path, manifest, tmp_path / "out")


def test_import_preserves_invalid_predictions_in_denominator(tmp_path: Path) -> None:
    manifest = ROOT / "data/manifests/chembl37_stage_5000_clean.csv"
    reference = pd.read_csv(manifest, dtype=str)
    validation = reference.loc[reference["split"] == "validation"]
    predictions = pd.DataFrame(
        {"image_id": validation["compound_id"], "post_SMILES": "not-a-smiles"}
    )
    path = tmp_path / "prediction_validation.csv"
    predictions.to_csv(path, index=False)
    report = import_predictions(path, manifest, tmp_path / "out")
    assert report["summary"]["count"] == 470
    assert report["summary"]["valid_smiles_rate"] == 0
    assert report["synthetic_viability_gate_passed"] is False
