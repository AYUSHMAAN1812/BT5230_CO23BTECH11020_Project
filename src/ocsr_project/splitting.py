"""Deterministic molecule-first split generation with scaffold grouping."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from .chemistry import canonicalize_smiles, parse_smiles


def scaffold_key(smiles: str) -> str:
    """Return a chirality-aware Murcko scaffold key.

    Acyclic molecules have an empty Murcko scaffold. They receive a canonical-molecule
    key so that they remain deduplicated without collapsing every acyclic structure into
    one unusably large group.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot scaffold invalid SMILES: {smiles!r}")
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for scaffold splitting") from exc
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    return scaffold or f"ACYCLIC:{canonicalize_smiles(smiles)}"


def scaffold_split(
    smiles_values: Sequence[str],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 20260915,
) -> list[str]:
    """Assign each input molecule to train, validation, or test without scaffold overlap."""
    total_fraction = train_fraction + validation_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_fraction}")
    if not smiles_values:
        return []

    canonical = [canonicalize_smiles(value) for value in smiles_values]
    if any(value is None for value in canonical):
        bad = [index for index, value in enumerate(canonical) if value is None]
        raise ValueError(f"Invalid SMILES at indices: {bad}")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate canonical molecules must be removed before splitting")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(canonical):
        groups[scaffold_key(value)].append(index)

    rng = random.Random(seed)
    grouped_indices = list(groups.values())
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=len, reverse=True)

    target = {
        "train": len(smiles_values) * train_fraction,
        "validation": len(smiles_values) * validation_fraction,
        "test": len(smiles_values) * test_fraction,
    }
    current = {name: 0 for name in target}
    assignments = [""] * len(smiles_values)

    for group in grouped_indices:
        deficits = {name: target[name] - current[name] for name in target}
        destination = max(deficits, key=lambda name: (deficits[name], -current[name]))
        for index in group:
            assignments[index] = destination
        current[destination] += len(group)

    assert_split_integrity(canonical, assignments)
    return assignments


def assert_split_integrity(smiles_values: Sequence[str], assignments: Sequence[str]) -> None:
    if len(smiles_values) != len(assignments):
        raise AssertionError("Molecule and split assignment lengths differ")
    allowed = {"train", "validation", "test"}
    if not set(assignments).issubset(allowed):
        raise AssertionError(f"Unexpected split names: {set(assignments).difference(allowed)}")

    seen_molecules: dict[str, str] = {}
    seen_scaffolds: dict[str, str] = {}
    for smiles, split in zip(smiles_values, assignments, strict=True):
        canonical = canonicalize_smiles(smiles)
        if canonical in seen_molecules and seen_molecules[canonical] != split:
            raise AssertionError(f"Molecule leakage detected for {canonical}")
        seen_molecules[canonical] = split
        scaffold = scaffold_key(canonical)
        if scaffold in seen_scaffolds and seen_scaffolds[scaffold] != split:
            raise AssertionError(f"Scaffold leakage detected for {scaffold}")
        seen_scaffolds[scaffold] = split

