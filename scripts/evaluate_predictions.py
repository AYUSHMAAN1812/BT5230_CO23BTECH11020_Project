#!/usr/bin/env python
"""Evaluate a prediction CSV with chemistry-aware metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.metrics import bootstrap_mean_interval, evaluate_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv).fillna("")
    required = {"image_id", "reference_smiles", "predicted_smiles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    summary, rows = evaluate_pairs(
        frame["reference_smiles"].astype(str).tolist(),
        frame["predicted_smiles"].astype(str).tolist(),
        frame["image_id"].astype(str).tolist(),
    )
    low, high = bootstrap_mean_interval([float(row["tanimoto"]) for row in rows], seed=args.seed)
    summary["mean_tanimoto_ci95_low"] = low
    summary["mean_tanimoto_ci95_high"] = high

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_csv.stem
    (args.output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(args.output_dir / f"{stem}_per_image.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

