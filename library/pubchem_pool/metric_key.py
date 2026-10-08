"""The structure key of the competition metric: RDKit tautomer canonicalisation, then the InChIKey's first block.

Copied from `canon_key` in casmi26-sota-v40-dreams-champion.ipynb (engine code). The metric pins RDKit 2026.03.3,
which is the version installed in .venv: other versions may canonicalise some tautomers differently.
"""

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
METRIC_RDKIT_VERSION = "2026.03.3"

TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


def metric_key(smiles: str) -> str | None:
    """Tautomer-canonical InChIKey first block (14 characters), or None if RDKit fails on the SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(TAUTOMER_ENUMERATOR.Canonicalize(mol))[:14]
    except Exception:
        return None


def standard_key(smiles: str) -> str | None:
    """Standard InChIKey first block, without tautomer canonicalisation (much faster), or None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)  # can raise, e.g. KekulizeException on some aromatic systems
        return key[:14] if key else None
    except Exception:
        return None
