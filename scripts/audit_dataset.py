#!/usr/bin/env python
"""Audit a rendered OCSR manifest and optionally verify a matched condition manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.chemistry import molecular_profile
from ocsr_project.splitting import assert_split_integrity
from ocsr_project.tokenizer import tokenize_smiles


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def resolve_image(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--compare-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/data_audit.json"))
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    required = {"compound_id", "canonical_smiles", "split", "image_path", "condition"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame["compound_id"].duplicated().any():
        raise AssertionError("Duplicate compound ids detected")
    if frame["canonical_smiles"].duplicated().any():
        raise AssertionError("Duplicate canonical molecules detected")
    assert_split_integrity(frame["canonical_smiles"].tolist(), frame["split"].tolist())

    token_lengths: list[float] = []
    atom_counts: list[float] = []
    ring_counts: list[float] = []
    stereocenter_counts: list[float] = []
    file_sizes: list[float] = []
    image_dimensions: set[tuple[int, int]] = set()
    missing_images: list[str] = []
    for row in frame.itertuples(index=False):
        token_lengths.append(len(tokenize_smiles(row.canonical_smiles)) + 2)
        profile = molecular_profile(row.canonical_smiles)
        if profile is None:
            raise AssertionError(f"Invalid canonical molecule: {row.compound_id}")
        atom_counts.append(profile["atom_count"])
        ring_counts.append(profile["ring_count"])
        stereocenter_counts.append(profile["stereocenter_count"])
        image_path = resolve_image(row.image_path)
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        file_sizes.append(image_path.stat().st_size)
        with Image.open(image_path) as image:
            image_dimensions.add(image.size)
    if missing_images:
        raise AssertionError(f"Missing {len(missing_images)} images; first: {missing_images[0]}")

    matched_conditions = None
    if args.compare_manifest:
        comparison = pd.read_csv(args.compare_manifest)
        left = frame[["compound_id", "canonical_smiles", "split"]].sort_values("compound_id")
        right = comparison[["compound_id", "canonical_smiles", "split"]].sort_values("compound_id")
        if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            raise AssertionError("Condition manifests do not contain identical molecules and splits")
        matched_conditions = True

    summary = {
        "status": "passed",
        "manifest": str(args.manifest),
        "condition": sorted(frame["condition"].unique().tolist()),
        "count": len(frame),
        "split_counts": frame["split"].value_counts().to_dict(),
        "matched_condition_manifest": matched_conditions,
        "unique_molecules": int(frame["canonical_smiles"].nunique()),
        "image_dimensions": [list(value) for value in sorted(image_dimensions)],
        "total_image_megabytes": float(sum(file_sizes) / 1_000_000),
        "image_bytes": describe(file_sizes),
        "sequence_tokens_including_special": describe(token_lengths),
        "atom_count": describe(atom_counts),
        "ring_count": describe(ring_counts),
        "stereocenter_count": describe(stereocenter_counts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

