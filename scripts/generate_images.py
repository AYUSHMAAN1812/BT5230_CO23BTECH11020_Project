#!/usr/bin/env python
"""Generate clean or augmented molecular depictions for an existing split manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.images import generate_depiction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--condition", choices=("clean", "handdrawn_augmented"), default="clean")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def stable_seed(global_seed: int, compound_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{compound_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_manifest)
    image_paths: list[str] = []
    for row in frame.itertuples(index=False):
        relative = Path(args.output_root) / args.condition / row.split / f"{row.compound_id}.png"
        absolute = relative if relative.is_absolute() else ROOT / relative
        generate_depiction(
            row.canonical_smiles,
            absolute,
            condition=args.condition,
            seed=stable_seed(args.seed, str(row.compound_id)),
            width=args.image_size,
            height=args.image_size,
        )
        image_paths.append(str(relative.as_posix()))
    frame["condition"] = args.condition
    frame["image_path"] = image_paths
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_manifest, index=False)
    print(f"Generated {len(frame)} {args.condition} images -> {args.output_manifest}")


if __name__ == "__main__":
    main()

