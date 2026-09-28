"""Compare the fixed DECIMER reference with MolScribe on identical image IDs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECIMER_PATH = (
    ROOT
    / "artifacts/metrics/decimer_external_reference/decimer_100_image_sample.csv"
)
CLEAN_PATH = (
    ROOT
    / "artifacts/metrics/molscribe/decimer_clean/"
    "prediction_decimer_hdm_benchmark_per_image.csv"
)
AUGMENTED_PATH = (
    ROOT
    / "artifacts/metrics/molscribe/decimer_augmented/"
    "augmented_prediction_decimer_hdm_benchmark_per_image.csv"
)
OUTPUT_PATH = (
    ROOT
    / "artifacts/metrics/decimer_external_reference/"
    "decimer_vs_molscribe_same_100.json"
)

SEED = 20260925
BOOTSTRAP_SAMPLES = 2_000


def read_indexed(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate image IDs in {path}")
    return indexed


def as_bool(value: str) -> float:
    return float(value.strip().lower() == "true")


def metric_vectors(
    rows: dict[str, dict[str, str]],
    ids: list[str],
    *,
    decimer: bool,
) -> dict[str, list[float]]:
    exact_iso_field = "exact_isomeric" if decimer else "exact_isomeric_match"
    exact_noniso_field = (
        "exact_nonisomeric" if decimer else "exact_nonisomeric_match"
    )
    return {
        "valid": [as_bool(rows[image_id]["valid"]) for image_id in ids],
        "exact_isomeric": [
            as_bool(rows[image_id][exact_iso_field]) for image_id in ids
        ],
        "exact_nonisomeric": [
            as_bool(rows[image_id][exact_noniso_field]) for image_id in ids
        ],
        "tanimoto": [float(rows[image_id]["tanimoto"]) for image_id in ids],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def paired_interval(
    candidate: list[float], baseline: list[float]
) -> tuple[float, float]:
    differences = [a - b for a, b in zip(candidate, baseline, strict=True)]
    rng = random.Random(SEED)
    count = len(differences)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        estimates.append(mean([differences[rng.randrange(count)] for _ in range(count)]))
    estimates.sort()
    return estimates[int(0.025 * BOOTSTRAP_SAMPLES)], estimates[
        int(0.975 * BOOTSTRAP_SAMPLES) - 1
    ]


def summarize(vectors: dict[str, list[float]]) -> dict[str, float]:
    return {name: mean(values) for name, values in vectors.items()}


def main() -> None:
    decimer_rows = read_indexed(DECIMER_PATH)
    clean_rows = read_indexed(CLEAN_PATH)
    augmented_rows = read_indexed(AUGMENTED_PATH)
    ids = sorted(decimer_rows)

    if len(ids) != 100:
        raise ValueError(f"Expected 100 DECIMER sample rows, found {len(ids)}")
    for name, rows in (("clean", clean_rows), ("augmented", augmented_rows)):
        missing = set(ids) - set(rows)
        if missing:
            raise ValueError(f"{name} predictions are missing {len(missing)} sample IDs")

    vectors = {
        "decimer_external": metric_vectors(decimer_rows, ids, decimer=True),
        "molscribe_clean": metric_vectors(clean_rows, ids, decimer=False),
        "molscribe_augmented": metric_vectors(augmented_rows, ids, decimer=False),
    }
    result: dict[str, object] = {
        "count": len(ids),
        "sample_definition": (
            "Fixed seed-20260925 simple random sample assembled as 20 initial IDs "
            "plus 80 random IDs from the remaining benchmark images."
        ),
        "metrics": {name: summarize(values) for name, values in vectors.items()},
        "paired_decimer_minus_molscribe": {},
    }

    paired = result["paired_decimer_minus_molscribe"]
    assert isinstance(paired, dict)
    for baseline_name in ("molscribe_clean", "molscribe_augmented"):
        comparison = {}
        for metric_name, decimer_values in vectors["decimer_external"].items():
            baseline_values = vectors[baseline_name][metric_name]
            low, high = paired_interval(decimer_values, baseline_values)
            comparison[metric_name] = {
                "difference": mean(decimer_values) - mean(baseline_values),
                "paired_bootstrap_ci95_low": low,
                "paired_bootstrap_ci95_high": high,
            }
        paired[baseline_name] = comparison

    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
