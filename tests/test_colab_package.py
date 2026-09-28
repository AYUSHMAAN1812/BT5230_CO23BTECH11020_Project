"""The Colab transfer bundle must not contain held-out or real benchmark data."""

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_colab_molscribe import create_package


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "white").save(path)


def _fixture(
    root: Path,
    *,
    bad_condition: bool = False,
    manifest_name: str = "manifest.csv",
    image_prefix: str = "",
) -> Path:
    rows = []
    for image_id, split, smiles in (
        ("CHEMBL1", "train", "CCO"),
        ("CHEMBL2", "validation", "c1ccccc1"),
        ("CHEMBL3", "test", "CCC"),
    ):
        relative = f"{image_prefix}data/processed/clean/{split}/{image_id}.png"
        _png(root / relative)
        rows.append(
            {
                "compound_id": image_id,
                "canonical_smiles": smiles,
                "split": split,
                "condition": "handdrawn_augmented" if bad_condition and split == "train" else "clean",
                "image_path": relative,
            }
        )
    manifest = root / manifest_name
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_package_contains_only_clean_train_and_validation(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    output = tmp_path / "out.zip"
    metadata = create_package(tmp_path, manifest, output, 1, 1)
    assert metadata["counts"] == {"train": 1, "validation": 1}
    assert metadata["excluded_synthetic_test_rows"] == 1
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert names == {
            "train.csv",
            "validation.csv",
            "images/train/CHEMBL1.png",
            "images/validation/CHEMBL2.png",
            "package_metadata.json",
        }
        saved = json.loads(archive.read("package_metadata.json"))
        assert not saved["contains_real_handdrawn_images"]
        assert not saved["contains_augmented_images"]
        rows = list(csv.DictReader(io.StringIO(archive.read("validation.csv").decode())))
        assert rows == [
            {
                "image_id": "CHEMBL2",
                "file_path": "images/validation/CHEMBL2.png",
                "SMILES": "c1ccccc1",
            }
        ]


def test_package_rejects_nonclean_training_row(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path, bad_condition=True)
    with pytest.raises(ValueError, match="Expected 'clean'"):
        create_package(tmp_path, manifest, tmp_path / "out.zip", 1, 1)


def test_package_can_pair_augmented_train_with_clean_validation(tmp_path: Path) -> None:
    clean_manifest = _fixture(tmp_path)
    augmented_manifest = _fixture(
        tmp_path,
        bad_condition=True,
        manifest_name="manifest_augmented.csv",
        image_prefix="augmented/",
    )
    output = tmp_path / "paired.zip"
    metadata = create_package(
        tmp_path,
        clean_manifest,
        output,
        1,
        1,
        training_manifest=augmented_manifest,
        training_condition="handdrawn_augmented",
    )
    assert metadata["training_condition"] == "handdrawn_augmented"
    assert metadata["contains_augmented_images"] is True
    with zipfile.ZipFile(output) as archive:
        assert "images/train/CHEMBL1.png" in archive.namelist()
        assert "images/validation/CHEMBL2.png" in archive.namelist()


def test_package_rejects_missing_image(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    (tmp_path / "data/processed/clean/train/CHEMBL1.png").unlink()
    with pytest.raises(ValueError, match="Missing or out-of-root"):
        create_package(tmp_path, manifest, tmp_path / "out.zip", 1, 1)
