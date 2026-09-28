"""Deterministic molecular depiction and student-scale hand-drawn-style augmentation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .chemistry import parse_smiles


def render_molecule(smiles: str, *, width: int = 224, height: int = 224) -> Image.Image:
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot render invalid SMILES: {smiles!r}")
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit drawing support is required") from exc

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def augment_handdrawn_style(image: Image.Image, *, seed: int) -> Image.Image:
    """Apply conservative visual perturbations without changing chemical semantics."""
    rng = np.random.default_rng(seed)
    result = image.convert("RGB")

    angle = float(rng.uniform(-7.0, 7.0))
    result = result.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor="white")

    if rng.random() < 0.30:
        result = result.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8))))
    if rng.random() < 0.35:
        result = ImageEnhance.Contrast(result).enhance(float(rng.uniform(0.75, 1.25)))
    if rng.random() < 0.30:
        result = ImageEnhance.Sharpness(result).enhance(float(rng.uniform(0.6, 1.5)))
    if rng.random() < 0.35:
        pixels = np.asarray(result, dtype=np.int16)
        noise = rng.normal(0.0, float(rng.uniform(1.5, 5.0)), size=pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(pixels, mode="RGB")
    return result


def generate_depiction(
    smiles: str,
    output_path: str | Path,
    *,
    condition: str = "clean",
    seed: int = 20260915,
    width: int = 224,
    height: int = 224,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_molecule(smiles, width=width, height=height)
    if condition == "handdrawn_augmented":
        image = augment_handdrawn_style(image, seed=seed)
    elif condition != "clean":
        raise ValueError(f"Unknown image condition: {condition}")
    image.save(output, format="PNG", optimize=True)
    return output

