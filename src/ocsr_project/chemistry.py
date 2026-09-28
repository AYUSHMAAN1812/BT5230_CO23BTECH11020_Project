"""Chemistry-aware normalization, validation, fingerprints, and descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - exercised only in an incomplete environment
        raise RuntimeError(
            "RDKit is required. Install the project environment before running chemistry code."
        ) from exc
    return Chem, DataStructs, rdFingerprintGenerator


def parse_smiles(smiles: str):
    """Return a sanitized RDKit molecule, or ``None`` when parsing fails."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    Chem, _, _ = _rdkit()
    try:
        from rdkit import rdBase

        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    except (ValueError, RuntimeError):
        return None


def canonicalize_smiles(smiles: str, *, isomeric: bool = True) -> str | None:
    """Canonicalize a SMILES string while preserving stereochemistry by default."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    Chem, _, _ = _rdkit()
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def is_valid_smiles(smiles: str) -> bool:
    return canonicalize_smiles(smiles) is not None


def tanimoto_similarity(reference: str, prediction: str, *, radius: int = 2) -> float:
    """Calculate Morgan-fingerprint Tanimoto similarity; invalid predictions score zero."""
    reference_mol = parse_smiles(reference)
    prediction_mol = parse_smiles(prediction)
    if reference_mol is None or prediction_mol is None:
        return 0.0
    _, DataStructs, rdFingerprintGenerator = _rdkit()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(reference_mol),
            generator.GetFingerprint(prediction_mol),
        )
    )


@dataclass(frozen=True)
class MolecularProfile:
    atom_count: int
    heavy_atom_count: int
    ring_count: int
    heteroatom_count: int
    formal_charge: int
    stereocenter_count: int


def molecular_profile(smiles: str) -> dict[str, int] | None:
    """Return descriptors used for subgroup analysis in the evaluation report."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    Chem, _, _ = _rdkit()
    stereocenters = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, includeCIP=True, useLegacyImplementation=False
    )
    profile = MolecularProfile(
        atom_count=mol.GetNumAtoms(),
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        ring_count=mol.GetRingInfo().NumRings(),
        heteroatom_count=sum(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()),
        formal_charge=sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        stereocenter_count=len(stereocenters),
    )
    return asdict(profile)
