"""Paired comparison of models evaluated on identical molecule images."""

from __future__ import annotations

import pandas as pd

from .metrics import bootstrap_mean_interval


def compare_evaluation_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    seed: int = 20260915,
) -> dict[str, object]:
    required = {
        "image_id",
        "reference_smiles",
        "valid",
        "exact_isomeric_match",
        "exact_nonisomeric_match",
        "tanimoto",
    }
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        if missing := required.difference(frame.columns):
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty or frame["image_id"].duplicated().any():
            raise ValueError(f"{name} must have nonempty, unique image IDs")

    left = baseline.sort_values("image_id").reset_index(drop=True)
    right = candidate.sort_values("image_id").reset_index(drop=True)
    if not left["image_id"].equals(right["image_id"]):
        raise ValueError("Paired evaluations must contain identical image IDs")
    if not left["reference_smiles"].equals(right["reference_smiles"]):
        raise ValueError("Paired evaluations must contain identical references")

    metrics = {}
    for column in (
        "valid",
        "exact_isomeric_match",
        "exact_nonisomeric_match",
        "tanimoto",
    ):
        left_values = pd.to_numeric(left[column], errors="raise").astype(float)
        right_values = pd.to_numeric(right[column], errors="raise").astype(float)
        delta = right_values - left_values
        low, high = bootstrap_mean_interval(delta.to_numpy(), seed=seed)
        metrics[column] = {
            "baseline_mean": float(left_values.mean()),
            "candidate_mean": float(right_values.mean()),
            "candidate_minus_baseline": float(delta.mean()),
            "paired_delta_ci95_low": low,
            "paired_delta_ci95_high": high,
        }
    return {"count": len(left), "metrics": metrics}
