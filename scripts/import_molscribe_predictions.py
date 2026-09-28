#!/usr/bin/env python
"""Evaluate Colab MolScribe predictions on the unchanged clean synthetic validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.metrics import bootstrap_mean_interval, evaluate_pairs


def import_predictions(
    predictions_path: Path,
    manifest_path: Path,
    output_dir: Path,
    prediction_field: str = "post_SMILES",
) -> dict[str, object]:
    reference = pd.read_csv(manifest_path, dtype=str).fillna("")
    reference = reference.loc[(reference["split"] == "validation") & (reference["condition"] == "clean")]
    expected = reference[["compound_id", "canonical_smiles"]].rename(
        columns={"compound_id": "image_id", "canonical_smiles": "reference_smiles"}
    )
    if len(expected) != 470 or expected["image_id"].duplicated().any():
        raise ValueError("The source manifest is not the expected 470-image clean validation split")

    upstream = pd.read_csv(predictions_path, dtype=str).fillna("")
    if "image_id" not in upstream or prediction_field not in upstream:
        raise ValueError(f"Missing image_id or {prediction_field!r} in MolScribe output")
    if upstream["image_id"].duplicated().any():
        raise ValueError("Duplicate prediction image IDs")
    if set(upstream["image_id"]) != set(expected["image_id"]):
        raise ValueError("Predictions do not match all 470 clean validation image IDs")
    merged = expected.merge(
        upstream[["image_id", prediction_field]].rename(
            columns={prediction_field: "predicted_smiles"}
        ),
        on="image_id",
        validate="one_to_one",
        sort=False,
    )
    summary, rows = evaluate_pairs(
        merged["reference_smiles"].tolist(),
        merged["predicted_smiles"].tolist(),
        merged["image_id"].tolist(),
    )
    low, high = bootstrap_mean_interval([float(row["tanimoto"]) for row in rows])
    summary["mean_tanimoto_ci95_low"] = low
    summary["mean_tanimoto_ci95_high"] = high
    noniso_count = sum(bool(row["exact_nonisomeric_match"]) for row in rows)
    gate = (
        summary["valid_smiles_rate"] >= 0.50
        and noniso_count >= 5
        and summary["mean_tanimoto"] >= 0.10
    )
    report: dict[str, object] = {
        "evaluation_set": "clean synthetic ChEMBL 37 validation; real benchmark remains held out",
        "prediction_field": prediction_field,
        "summary": summary,
        "nonisomeric_exact_count": noniso_count,
        "synthetic_viability_gate_passed": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = predictions_path.stem
    merged.to_csv(output_dir / f"{stem}_native.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / f"{stem}_per_image.csv", index=False)
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/manifests/chembl37_stage_5000_clean.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/metrics/molscribe")
    parser.add_argument("--prediction-field", default="post_SMILES")
    args = parser.parse_args()
    report = import_predictions(
        args.predictions_csv, args.manifest, args.output_dir, args.prediction_field
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
