import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")  # silence RDKit's parsing warnings

N_BITS = 2048
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)

def fingerprint_from_mol(mol: Chem.Mol) -> np.ndarray:
    """Morgan fingerprint (radius 2, 2048 bits) of an RDKit molecule as a 0/1 uint8 array."""
    return MORGAN_GENERATOR.GetFingerprintAsNumPy(mol).astype(np.uint8)


def fingerprint(smiles: str) -> np.ndarray | None:
    """Morgan fingerprint (radius 2, 2048 bits) as a 0/1 uint8 array, or None if the SMILES can't be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return fingerprint_from_mol(mol)

