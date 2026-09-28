#!/usr/bin/env python
"""Canonicalize, filter, deduplicate, and split a molecular CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.chemistry import canonicalize_smiles
from ocsr_project.splitting import scaffold_split
from ocsr_project.tokenizer import tokenize_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default="compound_id")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    required = {args.smiles_column, args.id_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        raw = str(row[args.smiles_column]).strip()
        canonical = canonicalize_smiles(raw)
        reason = None
        if canonical is None:
            reason = "invalid_smiles"
        elif "." in canonical:
            reason = "multi_fragment"
        else:
            try:
                token_count = len(tokenize_smiles(canonical)) + 2
            except ValueError:
                reason = "unsupported_token"
            else:
                if token_count > args.max_tokens:
                    reason = "sequence_too_long"
                elif canonical in seen:
                    reason = "duplicate_canonical_smiles"
        if reason:
            rejected.append(
                {
                    "compound_id": str(row[args.id_column]),
                    "raw_smiles": raw,
                    "reason": reason,
                }
            )
            continue
        seen.add(canonical)
        accepted.append(
            {
                "compound_id": str(row[args.id_column]),
                "raw_smiles": raw,
                "canonical_smiles": canonical,
            }
        )

    assignments = scaffold_split(
        [str(row["canonical_smiles"]) for row in accepted], seed=args.seed
    )
    for row, split in zip(accepted, assignments, strict=True):
        row["split"] = split

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted).to_csv(args.output_manifest, index=False)
    rejected_path = args.output_manifest.with_name(args.output_manifest.stem + "_rejected.csv")
    pd.DataFrame(rejected).to_csv(rejected_path, index=False)
    counts = pd.Series(assignments).value_counts().to_dict()
    summary = {
        "input_count": len(frame),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "split_counts": counts,
        "seed": args.seed,
    }
    summary_path = args.output_manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

