"""PyTorch dataset and collation utilities for image-to-SMILES training."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .tokenizer import SmilesTokenizer


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for model training") from exc
    return torch


class ImageSmilesDataset:
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        tokenizer: SmilesTokenizer,
        *,
        split: str | None = None,
        image_size: int = 224,
        max_sequence_length: int = 128,
        image_normalization: str = "symmetric",
    ) -> None:
        if image_normalization not in {"symmetric", "imagenet"}:
            raise ValueError(f"Unsupported image normalization: {image_normalization}")
        frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        required = {"compound_id", "canonical_smiles", "split", "image_path"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        if split is not None:
            frame = frame.loc[frame["split"] == split].copy()
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        self.image_normalization = image_normalization

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch = _torch()
        row = self.frame.iloc[index]
        with Image.open(Path(row["image_path"])) as raw_image:
            image = raw_image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.image_normalization == "imagenet":
            mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std
        else:
            tensor = (tensor - 0.5) / 0.5
        token_ids = self.tokenizer.encode(
            str(row["canonical_smiles"]), max_length=self.max_sequence_length
        )
        return {
            "image": tensor,
            "token_ids": token_ids,
            "compound_id": str(row["compound_id"]),
            "smiles": str(row["canonical_smiles"]),
        }


def make_collate_fn(tokenizer: SmilesTokenizer) -> Callable:
    def collate(batch: list[dict[str, object]]) -> dict[str, object]:
        torch = _torch()
        images = torch.stack([item["image"] for item in batch])
        max_length = max(len(item["token_ids"]) for item in batch)
        sequences = torch.full(
            (len(batch), max_length), tokenizer.pad_id, dtype=torch.long
        )
        for row_index, item in enumerate(batch):
            ids = torch.tensor(item["token_ids"], dtype=torch.long)
            sequences[row_index, : len(ids)] = ids
        return {
            "images": images,
            "sequences": sequences,
            "compound_ids": [item["compound_id"] for item in batch],
            "smiles": [item["smiles"] for item in batch],
        }

    return collate
