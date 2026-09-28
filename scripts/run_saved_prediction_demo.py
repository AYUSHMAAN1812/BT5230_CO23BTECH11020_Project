#!/usr/bin/env python
"""Create a fast visual replay of one preserved synthetic-validation prediction.

This assessor-facing demonstration does not run model inference. It combines a
stored validation image with the corresponding saved prediction and chemistry
metrics so the result can be inspected without a GPU or checkpoint download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ID = "CHEMBL450375"
IMAGE_DIR = ROOT / "data/processed/chembl37_stage_5000/clean/validation"
PREDICTION_FILES = {
    "clean": (
        ROOT
        / "artifacts/metrics/molscribe/finetuned/finetuned_prediction_validation_per_image.csv"
    ),
    "augmented": (
        ROOT
        / "artifacts/metrics/molscribe/augmented/augmented_prediction_validation_per_image.csv"
    ),
}


def load_record(condition: str, image_id: str) -> dict[str, object]:
    path = PREDICTION_FILES[condition]
    data = pd.read_csv(path, dtype={"image_id": str}).fillna("")
    selected = data.loc[data["image_id"] == image_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {image_id!r} in {path}, found {len(selected)}")
    return selected.iloc[0].to_dict()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _molecule_image(smiles: str, size: tuple[int, int]) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, size[1] // 2), "Invalid SMILES", fill="#9C1C1C", font=_font(24, bold=True))
        return image
    return Draw.MolToImage(molecule, size=size).convert("RGB")


def _paste_contained(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    contained = ImageOps.contain(
        source.convert("RGB"),
        (right - left, bottom - top),
        method=Image.Resampling.LANCZOS,
    )
    x = left + (right - left - contained.width) // 2
    y = top + (bottom - top - contained.height) // 2
    canvas.paste(contained, (x, y))


def _wrap_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text by rendered pixel width, including strings without spaces."""
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current.rstrip())
            current = character.lstrip()
        else:
            current = candidate
    if current:
        lines.append(current.rstrip())
    return lines


def create_panel(
    record: dict[str, object],
    image_path: Path,
    output_path: Path,
    condition: str,
) -> dict[str, object]:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    reference = str(record["reference_smiles"])
    prediction = str(record["predicted_smiles"])
    width, height = 1440, 620
    canvas = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, bold=True)
    heading_font = _font(23, bold=True)
    body_font = _font(19)
    small_font = _font(17)

    draw.text(
        (40, 22),
        "Saved Prediction Replay - No Model Inference",
        fill="#17365D",
        font=title_font,
    )
    draw.text(
        (40, 66),
        f"Image ID: {record['image_id']} | Condition: {condition} fine-tuned MolScribe",
        fill="#30343B",
        font=body_font,
    )

    panel_width = 440
    starts = (30, 500, 970)
    headings = ("Input structure image", "Reference molecule", "Predicted molecule")
    images = (
        Image.open(image_path).convert("RGB"),
        _molecule_image(reference, (390, 300)),
        _molecule_image(prediction, (390, 300)),
    )
    for start, heading, panel_image in zip(starts, headings, images, strict=True):
        draw.rounded_rectangle(
            (start, 105, start + panel_width, 575),
            radius=10,
            fill="white",
            outline="#B7C4D6",
            width=2,
        )
        draw.text((start + 20, 124), heading, fill="#17365D", font=heading_font)
        _paste_contained(canvas, panel_image, (start + 20, 165, start + 420, 440))

    for line_number, line in enumerate(_wrap_to_width(draw, reference, small_font, 400)):
        draw.text((520, 458 + line_number * 22), line, fill="#30343B", font=small_font)
    for line_number, line in enumerate(_wrap_to_width(draw, prediction, small_font, 400)):
        draw.text((990, 458 + line_number * 22), line, fill="#30343B", font=small_font)

    exact = str(record["exact_isomeric_match"]).lower() == "true"
    valid = str(record["valid"]).lower() == "true"
    tanimoto = float(record["tanimoto"])
    metric_lines = (
        f"Valid SMILES: {valid}",
        f"Exact isomeric match: {exact}",
        f"Tanimoto similarity: {tanimoto:.4f}",
    )
    for line_number, line in enumerate(metric_lines):
        draw.text((50, 452 + line_number * 23), line, fill="#30343B", font=small_font)

    source_font = _font(15)
    source = "Replay source: preserved validation image and saved per-image prediction."
    for line_number, line in enumerate(_wrap_to_width(draw, source, source_font, 390)):
        draw.text((50, 530 + line_number * 18), line, fill="#5B6573", font=source_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    try:
        reported_output = str(output_path.relative_to(ROOT))
    except ValueError:
        reported_output = str(output_path)
    return {
        "mode": "saved_prediction_replay_no_inference",
        "condition": condition,
        "image_id": str(record["image_id"]),
        "image_path": str(image_path.relative_to(ROOT)),
        "reference_smiles": reference,
        "predicted_smiles": prediction,
        "valid": valid,
        "exact_isomeric_match": exact,
        "tanimoto": tanimoto,
        "output_path": reported_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(PREDICTION_FILES), default="augmented")
    parser.add_argument("--image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/demo/saved_prediction_demo.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = IMAGE_DIR / f"{args.image_id}.png"
    record = load_record(args.condition, args.image_id)
    summary = create_panel(record, image_path, args.output.resolve(), args.condition)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
