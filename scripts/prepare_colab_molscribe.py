#!/usr/bin/env python
"""Package the clean, scaffold-separated ChEMBL train/validation images for Colab.

The synthetic test partition and held-out real benchmark are not exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/manifests/chembl37_stage_5000_clean.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/colab/molscribe_clean_train_validation.zip"
REQUIRED_FIELDS = {"compound_id", "canonical_smiles", "split", "condition", "image_path"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_add(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["image_id", "file_path", "SMILES"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def create_package(
    root: Path,
    manifest: Path,
    output: Path,
    expected_train: int,
    expected_validation: int,
    training_manifest: Path | None = None,
    training_condition: str = "clean",
) -> dict[str, object]:
    root = root.resolve()
    manifest = manifest.resolve()
    output = output.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Manifest lacks fields: {sorted(REQUIRED_FIELDS)}")
        validation_source_rows = list(reader)
    training_manifest = training_manifest.resolve() if training_manifest else manifest
    if not training_manifest.is_file():
        raise FileNotFoundError(training_manifest)
    if training_manifest == manifest:
        training_source_rows = validation_source_rows
    else:
        with training_manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                raise ValueError(f"Training manifest lacks fields: {sorted(REQUIRED_FIELDS)}")
            training_source_rows = list(reader)

    selected: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    image_sources: dict[str, Path] = {}
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    excluded_test = 0
    chosen_rows = [
        (row, training_condition)
        for row in training_source_rows
        if row["split"] == "train"
    ] + [
        (row, "clean") for row in validation_source_rows if row["split"] == "validation"
    ]
    excluded_test = sum(row["split"] == "test" for row in validation_source_rows)
    for row, required_condition in chosen_rows:
        split = row["split"]
        if row["condition"] != required_condition:
            raise ValueError(
                f"Expected {required_condition!r} images for {split}, got {row['condition']!r}"
            )
        image_id = row["compound_id"]
        smiles = row["canonical_smiles"]
        if not SAFE_ID.fullmatch(image_id) or not smiles:
            raise ValueError(f"Invalid image ID or empty SMILES: {image_id!r}")
        if image_id in seen_ids or smiles in seen_smiles:
            raise ValueError(f"Duplicate molecule or image ID: {image_id}")
        seen_ids.add(image_id)
        seen_smiles.add(smiles)
        source_relative = PurePosixPath(row["image_path"].replace("\\", "/"))
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ValueError(f"Unsafe image path: {source_relative}")
        source = root.joinpath(*source_relative.parts).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing or out-of-root image: {source}")
        target = f"images/{split}/{image_id}.png"
        selected[split].append({"image_id": image_id, "file_path": target, "SMILES": smiles})
        image_sources[target] = source

    counts = {split: len(rows) for split, rows in selected.items()}
    if counts != {"train": expected_train, "validation": expected_validation}:
        raise ValueError(f"Unexpected selected split counts: {counts}")
    if not excluded_test:
        raise ValueError("Expected a held-out synthetic test partition in the source manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for split in ("train", "validation"):
            name = f"{split}.csv"
            contents = _csv_bytes(selected[split])
            checksums[name] = _sha256(contents)
            _zip_add(archive, name, contents)
        for name, source in sorted(image_sources.items()):
            contents = source.read_bytes()
            with Image.open(io.BytesIO(contents)) as image:
                image.verify()
            checksums[name] = _sha256(contents)
            _zip_add(archive, name, contents)
        metadata: dict[str, object] = {
            "purpose": (
                f"MolScribe matched synthetic fine-tuning: {training_condition} training / "
                "clean validation"
            ),
            "source_manifest": manifest.name,
            "source_manifest_sha256": _sha256(manifest.read_bytes()),
            "training_manifest": training_manifest.name,
            "training_manifest_sha256": _sha256(training_manifest.read_bytes()),
            "training_condition": training_condition,
            "counts": counts,
            "excluded_synthetic_test_rows": excluded_test,
            "contains_augmented_images": training_condition != "clean",
            "contains_real_handdrawn_images": False,
            "coordinate_annotations_available": False,
            "bond_edge_annotations_available": False,
            "checksums": checksums,
        }
        _zip_add(archive, "package_metadata.json", json.dumps(metadata, indent=2).encode())
    metadata["package_sha256"] = _sha256(output.read_bytes())
    metadata["package_path"] = str(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-condition", default="clean")
    parser.add_argument("--expected-train", type=int, default=3758)
    parser.add_argument("--expected-validation", type=int, default=470)
    args = parser.parse_args()
    metadata = create_package(
        args.root,
        args.manifest,
        args.output,
        args.expected_train,
        args.expected_validation,
        args.training_manifest,
        args.training_condition,
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "checksums"}, indent=2))


if __name__ == "__main__":
    main()
