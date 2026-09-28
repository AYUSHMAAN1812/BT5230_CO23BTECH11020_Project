"""Evaluation metrics that preserve invalid predictions in the denominator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .chemistry import canonicalize_smiles, molecular_profile, tanimoto_similarity


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class PairResult:
    image_id: str
    reference_smiles: str
    predicted_smiles: str
    canonical_reference: str | None
    canonical_prediction: str | None
    reference_parseable: bool
    valid: bool
    exact_isomeric_match: bool
    exact_nonisomeric_match: bool
    tanimoto: float
    token_edit_distance: int
    atom_count: int | None
    ring_count: int | None
    stereocenter_count: int | None


def evaluate_pairs(
    references: Sequence[str],
    predictions: Sequence[str],
    image_ids: Sequence[str] | None = None,
    *,
    allow_invalid_references: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction counts differ")
    if image_ids is None:
        image_ids = [str(index) for index in range(len(references))]
    if len(image_ids) != len(references):
        raise ValueError("Image id count differs from reference count")

    rows: list[PairResult] = []
    for image_id, reference, prediction in zip(image_ids, references, predictions, strict=True):
        canonical_reference = canonicalize_smiles(reference)
        if canonical_reference is None and not allow_invalid_references:
            raise ValueError(f"Invalid reference SMILES for {image_id}: {reference!r}")
        canonical_prediction = canonicalize_smiles(prediction)
        noniso_reference = canonicalize_smiles(reference, isomeric=False)
        noniso_prediction = canonicalize_smiles(prediction, isomeric=False)
        profile = (molecular_profile(canonical_reference) if canonical_reference else None) or {}
        rows.append(
            PairResult(
                image_id=str(image_id),
                reference_smiles=reference,
                predicted_smiles=prediction,
                canonical_reference=canonical_reference,
                canonical_prediction=canonical_prediction,
                reference_parseable=canonical_reference is not None,
                valid=canonical_prediction is not None,
                exact_isomeric_match=(
                    canonical_reference is not None
                    and canonical_prediction == canonical_reference
                ),
                exact_nonisomeric_match=(
                    noniso_reference is not None
                    and noniso_prediction is not None
                    and noniso_prediction == noniso_reference
                ),
                tanimoto=(
                    tanimoto_similarity(canonical_reference, prediction)
                    if canonical_reference is not None
                    else 0.0
                ),
                token_edit_distance=levenshtein_distance(reference, prediction),
                atom_count=profile.get("atom_count"),
                ring_count=profile.get("ring_count"),
                stereocenter_count=profile.get("stereocenter_count"),
            )
        )

    count = len(rows)
    tanimoto_values = np.asarray([row.tanimoto for row in rows], dtype=float)
    summary: dict[str, float | int] = {
        "count": count,
        "reference_parseable_rate": (
            float(np.mean([row.reference_parseable for row in rows])) if count else 0.0
        ),
        "valid_smiles_rate": float(np.mean([row.valid for row in rows])) if count else 0.0,
        "exact_isomeric_match_rate": (
            float(np.mean([row.exact_isomeric_match for row in rows])) if count else 0.0
        ),
        "exact_nonisomeric_match_rate": (
            float(np.mean([row.exact_nonisomeric_match for row in rows])) if count else 0.0
        ),
        "mean_tanimoto": float(np.mean(tanimoto_values)) if count else 0.0,
        "median_tanimoto": float(np.median(tanimoto_values)) if count else 0.0,
        "tanimoto_at_least_0_8": float(np.mean(tanimoto_values >= 0.8)) if count else 0.0,
        "tanimoto_at_least_0_9": float(np.mean(tanimoto_values >= 0.9)) if count else 0.0,
        "mean_edit_distance": (
            float(np.mean([row.token_edit_distance for row in rows])) if count else 0.0
        ),
    }
    return summary, [asdict(row) for row in rows]


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int = 20260915, resamples: int = 2000
) -> tuple[float, float]:
    """Return a percentile 95% bootstrap interval for a mean."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty sequence")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)
