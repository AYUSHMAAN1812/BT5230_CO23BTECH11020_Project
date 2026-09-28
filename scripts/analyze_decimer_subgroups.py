#!/usr/bin/env python
"""Create predefined chemical subgroup and paired error analyses for DECIMER HDM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.comparison import compare_evaluation_frames


def _labels(frame: pd.DataFrame) -> pd.DataFrame:
    labelled = frame.copy()
    labelled["smiles_length"] = labelled["reference_smiles"].astype(str).str.len()
    labelled["atom_group"] = pd.cut(
        labelled["atom_count"],
        bins=[0, 10, 20, 30, np.inf],
        labels=["2-10", "11-20", "21-30", "31+"],
        include_lowest=True,
    ).astype("string").fillna("Unprofiled")
    labelled["ring_group"] = np.select(
        [
            labelled["ring_count"].eq(0),
            labelled["ring_count"].eq(1),
            labelled["ring_count"].eq(2),
            labelled["ring_count"].ge(3),
        ],
        ["0", "1", "2", "3+"],
        default="Unprofiled",
    )
    labelled["stereocenter_group"] = np.select(
        [
            labelled["stereocenter_count"].eq(0),
            labelled["stereocenter_count"].eq(1),
            labelled["stereocenter_count"].ge(2),
        ],
        ["0", "1", "2+"],
        default="Unprofiled",
    )
    labelled["smiles_length_group"] = pd.cut(
        labelled["smiles_length"],
        bins=[0, 25, 50, 75, np.inf],
        labels=["1-25", "26-50", "51-75", "76+"],
        include_lowest=True,
    ).astype("string")
    return labelled


def analyze(clean_path: Path, augmented_path: Path, output_dir: Path) -> dict[str, object]:
    clean = _labels(pd.read_csv(clean_path))
    augmented = pd.read_csv(augmented_path)
    if set(clean["image_id"]) != set(augmented["image_id"]):
        raise ValueError("Clean and augmented evaluations do not contain identical image IDs")

    dimensions = {
        "Atom count": "atom_group",
        "Ring count": "ring_group",
        "Stereocenters": "stereocenter_group",
        "Reference SMILES length": "smiles_length_group",
    }
    subgroup_rows: list[dict[str, object]] = []
    subgroup_json: dict[str, object] = {}
    for dimension, column in dimensions.items():
        subgroup_json[dimension] = {}
        for group in clean[column].drop_duplicates().tolist():
            ids = clean.loc[clean[column] == group, "image_id"]
            clean_group = clean.loc[clean["image_id"].isin(ids)]
            augmented_group = augmented.loc[augmented["image_id"].isin(ids)]
            result = compare_evaluation_frames(clean_group, augmented_group)
            subgroup_json[dimension][str(group)] = result
            row: dict[str, object] = {
                "dimension": dimension,
                "group": str(group),
                "count": result["count"],
            }
            for metric, values in result["metrics"].items():
                prefix = metric.replace("exact_isomeric_match", "isomeric_exact").replace(
                    "exact_nonisomeric_match", "nonisomeric_exact"
                )
                row[f"clean_{prefix}"] = values["baseline_mean"]
                row[f"augmented_{prefix}"] = values["candidate_mean"]
                row[f"delta_{prefix}"] = values["candidate_minus_baseline"]
                row[f"delta_{prefix}_ci95_low"] = values["paired_delta_ci95_low"]
                row[f"delta_{prefix}_ci95_high"] = values["paired_delta_ci95_high"]
            subgroup_rows.append(row)

    paired = clean.merge(
        augmented[
            [
                "image_id",
                "predicted_smiles",
                "valid",
                "exact_isomeric_match",
                "exact_nonisomeric_match",
                "tanimoto",
            ]
        ],
        on="image_id",
        suffixes=("_clean", "_augmented"),
        validate="one_to_one",
    )
    paired["tanimoto_delta"] = paired["tanimoto_augmented"] - paired["tanimoto_clean"]
    paired["isomeric_transition"] = np.select(
        [
            paired["exact_isomeric_match_clean"] & paired["exact_isomeric_match_augmented"],
            paired["exact_isomeric_match_clean"] & ~paired["exact_isomeric_match_augmented"],
            ~paired["exact_isomeric_match_clean"] & paired["exact_isomeric_match_augmented"],
        ],
        ["Correct in both", "Lost after augmentation", "Corrected after augmentation"],
        default="Incorrect in both",
    )
    transition_counts = {
        str(key): int(value)
        for key, value in paired["isomeric_transition"].value_counts().items()
    }
    summary: dict[str, object] = {
        "count": len(paired),
        "subgroups": subgroup_json,
        "isomeric_transition_counts": transition_counts,
        "tanimoto_better_count": int((paired["tanimoto_delta"] > 0).sum()),
        "tanimoto_tied_count": int((paired["tanimoto_delta"] == 0).sum()),
        "tanimoto_worse_count": int((paired["tanimoto_delta"] < 0).sum()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subgroup_rows).to_csv(output_dir / "decimer_subgroup_metrics.csv", index=False)
    paired.to_csv(output_dir / "decimer_paired_error_analysis.csv", index=False)
    (output_dir / "decimer_subgroup_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.clean, args.augmented, args.output_dir)
    compact = {key: value for key, value in summary.items() if key != "subgroups"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
