"""Training helpers shared by the controlled synthetic development experiments."""

from __future__ import annotations

import random

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for training") from exc
    return torch


def seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_cross_entropy(
    logits,
    target,
    *,
    pad_id: int,
    eos_id: int | None = None,
    eos_loss_weight: float = 1.0,
):
    torch = _torch()
    if eos_loss_weight <= 0:
        raise ValueError("eos_loss_weight must be positive")
    weights = None
    if eos_loss_weight != 1.0:
        if eos_id is None or eos_id == pad_id or not 0 <= eos_id < logits.shape[-1]:
            raise ValueError("A valid, non-padding eos_id is required for EOS weighting")
        weights = torch.ones(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        weights[eos_id] = eos_loss_weight
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        weight=weights,
        ignore_index=pad_id,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    *,
    pad_id: int,
    device: str,
    clip_norm: float = 1.0,
    eos_id: int | None = None,
    eos_loss_weight: float = 1.0,
):
    torch = _torch()
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        images = batch["images"].to(device)
        sequences = batch["sequences"].to(device)
        decoder_input = sequences[:, :-1]
        target = sequences[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, decoder_input)
        loss = token_cross_entropy(
            logits,
            target,
            pad_id=pad_id,
            eos_id=eos_id,
            eos_loss_weight=eos_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += float(loss.detach()) * images.shape[0]
        total_items += images.shape[0]
    return total_loss / max(total_items, 1)


@_torch().no_grad()
def evaluate_loss(
    model,
    loader,
    *,
    pad_id: int,
    device: str,
    eos_id: int | None = None,
    eos_loss_weight: float = 1.0,
):
    model.eval()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        images = batch["images"].to(device)
        sequences = batch["sequences"].to(device)
        logits = model(images, sequences[:, :-1])
        loss = token_cross_entropy(
            logits,
            sequences[:, 1:],
            pad_id=pad_id,
            eos_id=eos_id,
            eos_loss_weight=eos_loss_weight,
        )
        total_loss += float(loss) * images.shape[0]
        total_items += images.shape[0]
    return total_loss / max(total_items, 1)
