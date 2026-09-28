#!/usr/bin/env python
"""Evaluate saved MolScribe predictions on the held-out DECIMER HDM benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.metrics import bootstrap_mean_interval, evaluate_pairs

EXPECTED_MANIFEST_SHA256 = "bb88840799a54eadbaff7776a440744dec2cdeb7a84bde37d8cb4294e58e97f8"


def import_decimer_predictions(
    predictions_path: Path,
    manifest_path: Path,
    output_dir: Path,
    prediction_field: str = "post_SMILES",
    expected_count: int = 5088,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256,
) -> dict[str, object]:
    RDLogger.DisableLog("rdApp.*")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if expected_manifest_sha256 and manifest_digest != expected_manifest_sha256:
        raise ValueError("DECIMER benchmark manifest SHA-256 does not match the expected manifest")

    reference = pd.read_csv(manifest_path, dtype=str).fillna("")
    required_reference = {"image_id", "SMILES"}
    if not required_reference.issubset(reference.columns):
        raise ValueError(f"Benchmark manifest lacks fields: {sorted(required_reference)}")
    if len(reference) != expected_count or reference["image_id"].duplicated().any():
        raise ValueError(f"Expected {expected_count} unique benchmark image IDs")

    upstream = pd.read_csv(predictions_path, dtype=str).fillna("")
    if "image_id" not in upstream or prediction_field not in upstream:
        raise ValueError(f"Missing image_id or {prediction_field!r} in MolScribe output")
    if upstream["image_id"].duplicated().any():
        raise ValueError("Duplicate prediction image IDs")
    if set(upstream["image_id"]) != set(reference["image_id"]):
        raise ValueError("Predictions do not match every held-out benchmark image ID")

    merged = reference[["image_id", "SMILES"]].rename(
        columns={"SMILES": "reference_smiles"}
    ).merge(
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
        allow_invalid_references=True,
    )
    low, high = bootstrap_mean_interval([float(row["tanimoto"]) for row in rows])
    summary["mean_tanimoto_ci95_low"] = low
    summary["mean_tanimoto_ci95_high"] = high
    report: dict[str, object] = {
        "evaluation_set": "DECIMER Hand-Drawn Molecule Images v1.2; held-out external benchmark",
        "benchmark_manifest_sha256": manifest_digest,
        "prediction_field": prediction_field,
        "reference_handling": (
            "All 5,088 rows remain in the denominator. References unsupported by the local "
            "RDKit build receive zero for chemical-match and Tanimoto metrics."
        ),
        "summary": summary,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-field", default="post_SMILES")
    args = parser.parse_args()
    report = import_decimer_predictions(
        args.predictions_csv,
        args.manifest,
        args.output_dir,
        args.prediction_field,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
