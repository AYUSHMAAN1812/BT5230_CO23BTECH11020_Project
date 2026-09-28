"""Deterministic short-structure training curriculum helpers."""

from __future__ import annotations

import pandas as pd

from .tokenizer import tokenize_smiles


def select_short_molecules(frame: pd.DataFrame, *, max_smiles_tokens: int) -> pd.DataFrame:
    """Return an ordered training-only subset without changing validation or vocabulary."""
    if max_smiles_tokens < 1:
        raise ValueError("max_smiles_tokens must be positive")
    if set(frame["split"]) != {"train"}:
        raise ValueError("Curriculum selection accepts only training molecules")
    lengths = frame["canonical_smiles"].map(lambda value: len(tokenize_smiles(str(value))))
    selected = frame.loc[lengths.le(max_smiles_tokens)].copy()
    if selected.empty:
        raise ValueError("Curriculum selected no training molecules")
    return selected


def training_phase(epoch: int, warmup_epochs: int) -> str:
    if epoch < 1 or warmup_epochs < 0:
        raise ValueError("Epoch and warm-up count must be nonnegative and epoch starts at one")
    return "short" if epoch <= warmup_epochs else "full"
