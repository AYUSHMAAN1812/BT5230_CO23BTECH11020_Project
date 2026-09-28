#!/usr/bin/env python
"""Recompute the DECIMER results from saved prediction/reference pairs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.comparison import compare_evaluation_frames
from ocsr_project.metrics import bootstrap_mean_interval, evaluate_pairs

CLEAN = ROOT / "artifacts/metrics/molscribe/decimer_clean/prediction_decimer_hdm_benchmark_native.csv"
AUGMENTED = (
    ROOT
    / "artifacts/metrics/molscribe/decimer_augmented/augmented_prediction_decimer_hdm_benchmark_native.csv"
)
EXPECTED_SHA256 = {
    "clean": "80aef7636cfc7113c996f642ff57c93bd4cce87794bdbf5736e3bb3d8bcf1aa5",
    "augmented": "28a035fb8e114d961dea50eddab35020647cae043b6336fa8f11f6417b941ded",
}
EXPECTED = {
    "clean": {
        "valid_smiles_rate": 0.7651336477987422,
        "exact_isomeric_match_rate": 0.09689465408805031,
        "exact_nonisomeric_match_rate": 0.11163522012578617,
        "mean_tanimoto": 0.29980935441462625,
    },
    "augmented": {
        "valid_smiles_rate": 0.7682783018867925,
        "exact_isomeric_match_rate": 0.0972877358490566,
        "exact_nonisomeric_match_rate": 0.1104559748427673,
        "mean_tanimoto": 0.2941200774598872,
    },
}


def _evaluate(path: Path, expected_hash: str) -> tuple[dict[str, float | int], pd.DataFrame]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Saved prediction/reference file hash mismatch: {path}")
    data = pd.read_csv(path, dtype=str).fillna("")
    if len(data) != 5088 or data["image_id"].duplicated().any():
        raise ValueError(f"Expected 5,088 unique image IDs: {path}")
    summary, rows = evaluate_pairs(
        data["reference_smiles"].tolist(),
        data["predicted_smiles"].tolist(),
        data["image_id"].tolist(),
        allow_invalid_references=True,
    )
    low, high = bootstrap_mean_interval([float(row["tanimoto"]) for row in rows])
    summary["mean_tanimoto_ci95_low"] = low
    summary["mean_tanimoto_ci95_high"] = high
    return summary, pd.DataFrame(rows)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    summaries: dict[str, dict[str, float | int]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for condition, path in (("clean", CLEAN), ("augmented", AUGMENTED)):
        summaries[condition], frames[condition] = _evaluate(path, EXPECTED_SHA256[condition])
        for metric, expected_value in EXPECTED[condition].items():
            actual = float(summaries[condition][metric])
            if abs(actual - expected_value) > 1e-12:
                raise AssertionError(
                    f"{condition} {metric}: expected {expected_value}, recomputed {actual}"
                )

    if not frames["clean"]["image_id"].equals(frames["augmented"]["image_id"]):
        raise AssertionError("Clean and augmented image order differs")
    comparison = compare_evaluation_frames(frames["clean"], frames["augmented"])
    output = {"summaries": summaries, "paired_comparison": comparison, "verification": "PASS"}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
